# best_first_crawling_strategy.py
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Set, Dict, List, Tuple
from urllib.parse import urlparse
import os
import csv

from ..models import TraversalStats
from .filters import FilterChain
from .scorers import URLScorer
from . import DeepCrawlStrategy

from .._types import AsyncWebCrawler, CrawlerRunConfig, CrawlResult, RunManyReturn
from ..utils import normalize_url_for_deep_crawl

from math import inf as infinity

# Configurable batch size for processing items from the priority queue
BATCH_SIZE = 10


class BestLinkFirstCrawlingStrategy(DeepCrawlStrategy):
    """
    Best-Link-First Crawling Strategy using a priority queue.
    
    This strategy prioritizes URLs based on their score, ensuring that higher-value
    pages are crawled first. It reimplements the core traversal loop to use a priority
    queue while keeping URL validation and link discovery consistent with our design.
    
    Core methods:
      - arun: Returns either a list (batch mode) or an async generator (stream mode).
      - _arun_best_first: Core generator that uses a priority queue to yield CrawlResults.
      - can_process_url: Validates URLs and applies filtering (inherited behavior).
      - link_discovery: Extracts and validates links from a CrawlResult.
    """
    def __init__(
        self,
        max_depth: int,
        filter_chain: FilterChain = FilterChain(),
        url_scorer: Optional[URLScorer] = None,
        include_external: bool = False,
        max_pages: int = infinity,
        batch_size: int = BATCH_SIZE,
        save_discovered_urls: bool = False,
        enforce_depth: bool = False, # Enable when using PathDepthFilter
        logger: Optional[logging.Logger] = None,
    ):
        self.max_depth = max_depth
        self.filter_chain = filter_chain
        self.url_scorer = url_scorer
        self.include_external = include_external
        self.max_pages = max_pages
        self.batch_size = batch_size
        self.save_discovered_urls = save_discovered_urls
        self.enforce_depth = enforce_depth
        self.logger = logger or logging.getLogger(__name__)
        self.stats = TraversalStats(start_time=datetime.now())
        self._cancel_event = asyncio.Event()
        self._pages_crawled = 0
        # Per-run cache to avoid rescoring the same URL multiple times
        self._score_cache: Dict[str, float] = {}
        # Per-run cache of filter decisions: True=allowed, False=rejected
        self._filter_cache: Dict[str, bool] = {}
        # Track crawl order index and per-URL total scores
        self._crawl_index_counter: int = 0
        self.crawl_order: Dict[str, int] = {}

    def _append_discovered_url(self, base_url: str, depth: int) -> None:
        """
        Append discovered URLs to a structured file with domain and path parts.
        File columns: domain, depth1, depth2, ...
        """
        parsed = urlparse(base_url)
        domain = parsed.netloc.replace("www.", "")
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        # Prepare the row (pad up to 2 depths for consistency)
        row = [domain] + path_parts
        while len(row) < 2:
            row.append("")

        # Ensure output directory
        output_file = "discovered_links2.csv"
        file_exists = os.path.isfile(output_file)

        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # write header if file is new
            if not file_exists:
                header = ["domain", "depth1", "depth2"]
                writer.writerow(header)
            writer.writerow(row)

    def _path_depth(self, url: str) -> int:
        """
        Compute depth as the number of path segments after the domain.
        Examples:
        - https://example.com/         -> 0
        - https://example.com/a        -> 1
        - https://example.com/a/b      -> 2
        """
        try:
            parsed = urlparse(url)
            # remove leading/trailing slashes and filter empty segments
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            return len(parts)
        except Exception:
            return 0

    def _is_sitemap_url(self, url: str) -> bool:
        """
        Check if the URL is a sitemap URL (/sitemap or /sitemap.xml).
        """
        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/").lower()
            return path in ["sitemap", "sitemap.xml"]
        except Exception:
            return False
            
    async def can_process_url(self, url: str, depth: int) -> bool:
        """
        Validate the URL format and apply filtering.
        For the starting URL (depth 0), filtering is bypassed.
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Missing scheme or netloc")
            if parsed.scheme not in ("http", "https"):
                raise ValueError("Invalid scheme")
            if "." not in parsed.netloc:
                raise ValueError("Invalid domain")
        except Exception as e:
            self.logger.warning(f"Invalid URL: {url}, error: {e}")
            return False

        # TODO: we want to filter for depth 0 too
        ###if depth != 0 and not await self.filter_chain.apply(url):
        if not await self.filter_chain.apply(url):
            self.logger.debug(f"Skipping {url} due to filter chain")
            return False

        return True

    async def link_discovery(
        self,
        result: CrawlResult,
        source_url: str,
        current_depth: int,
        visited: Set[str],
        next_links: List[Tuple[str, Optional[str]]],
        depths: Dict[str, int],
    ) -> None:
        """
        Extract links from the crawl result, validate them, and append new URLs
        (with their parent references) to next_links.
        Also updates the depths dictionary.
        """
        new_depth = current_depth + 1
        if new_depth > self.max_depth:
            return
            
        # If we've reached the max pages limit, don't discover new links
        remaining_capacity = self.max_pages - self._pages_crawled
        if remaining_capacity <= 0:
            self.logger.info(f"Max pages limit ({self.max_pages}) reached, stopping link discovery")
            return

        # Retrieve internal links; include external links if enabled.
        links = result.links.get("internal", [])
        if self.include_external:
            links += result.links.get("external", [])

        # If we have more links than remaining capacity, limit how many we'll process
        valid_links = []
        for link in links:
            url = link.get("href")
            base_url = normalize_url_for_deep_crawl(url, source_url)
            if base_url in visited:
                continue
            # Consult cached filter decision first to avoid repeated filter checks/logs
            allowed = self._filter_cache.get(base_url)
            if allowed is None:
                allowed = await self.can_process_url(url, new_depth)
                self._filter_cache[base_url] = allowed
            if not allowed:
                self.stats.urls_skipped += 1
                continue
            
            valid_links.append(base_url)
            
        # Record the new depths and add to next_links (do not limit here; we will
        # score and limit later in the main loop to keep the highest-scoring URLs)
        for url in valid_links:
            depths[url] = new_depth
            next_links.append((url, source_url))


    async def link_discovery_with_depth(
        self,
        result: CrawlResult,
        source_url: str,
        current_depth: int,
        visited: Set[str],
        next_links: List[Tuple[str, Optional[str]]],
        depths: Dict[str, int],
    ) -> None:
        """
        Extract links from the crawl result, validate them, and append new URLs
        (with their parent references) to next_links.
        Also updates the depths dictionary.
        """ 
        # If we've reached the max pages limit, don't discover new links
        remaining_capacity = self.max_pages - self._pages_crawled
        if remaining_capacity <= 0:
            self.logger.info(f"Max pages limit ({self.max_pages}) reached, stopping link discovery")
            return

        # Retrieve internal links; include external links if enabled.
        links = result.links.get("internal", [])
        if self.include_external:
            links += result.links.get("external", [])

        # If we have more links than remaining capacity, limit how many we'll process
        valid_links = []
        for link in links:
            url = link.get("href")
            base_url = normalize_url_for_deep_crawl(url, source_url)
            if base_url in visited:
                continue

            link_depth = self._path_depth(url)
            """ if self.enforce_depth and link_depth > self.max_depth:
                self.logger.debug(f"Skipping {url} (depth {link_depth} > max_depth {self.max_depth})")
                self.stats.urls_skipped += 1
                continue """

            # Consult cached filter decision first to avoid repeated filter checks/logs
            allowed = self._filter_cache.get(base_url)
            if allowed is None:
                allowed = await self.can_process_url(url, link_depth)
                self._filter_cache[base_url] = allowed
            if not allowed:
                self.stats.urls_skipped += 1
                continue
                
            valid_links.append(base_url)
            depths[base_url] = link_depth
            
        # Record the new depths and add to next_links (do not limit here; we will
        # score and limit later in the main loop to keep the highest-scoring URLs)
        for url in valid_links:
            next_links.append((url, source_url))


    async def _arun_batch(
        self,
        start_url: str,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> List[CrawlResult]:
        """
        Best-first crawl in batch mode.
        
        Aggregates all CrawlResults into a list.
        """
        results: List[CrawlResult] = []
        async for result in self._arun_highest_score_first(start_url, crawler, config):
            results.append(result)
        return results

    async def _arun_stream(
        self,
        start_url: str,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> AsyncGenerator[CrawlResult, None]:
        """
        Best-first crawl in streaming mode.
        
        Yields CrawlResults as they become available.
        """
        async for result in self._arun_highest_score_first(start_url, crawler, config):
            yield result

    async def arun(
        self,
        start_url: str,
        crawler: AsyncWebCrawler,
        config: Optional[CrawlerRunConfig] = None,
    ) -> "RunManyReturn":
        """
        Main entry point for best-first crawling.
        
        Returns either a list (batch mode) or an async generator (stream mode)
        of CrawlResults.
        """
        if config is None:
            raise ValueError("CrawlerRunConfig must be provided")

        # TODO TEST
        # Reset per-run state so that max_pages applies per start_url
        self._pages_crawled = 0
        self._cancel_event = asyncio.Event()
        self.stats = TraversalStats(start_time=datetime.now())
        # Reset scoring cache for this run
        self._score_cache.clear()
        # Reset filter decision cache for this run
        self._filter_cache.clear()
        # Reset crawl order tracking
        self._crawl_index_counter = 0
        self.crawl_order.clear()

    

        if config.stream:
            return self._arun_stream(start_url, crawler, config)
        else:
            return await self._arun_batch(start_url, crawler, config)

    async def shutdown(self) -> None:
        """
        Signal cancellation and clean up resources.
        """
        self._cancel_event.set()
        self.stats.end_time = datetime.now()


    async def _arun_highest_score_first(
        self,
        start_url: str,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> AsyncGenerator[CrawlResult, None]:
        """
        Highest-score-first crawl strategy using a priority queue.

        URLs are processed in batches, prioritizing those with the highest score.
        The priority queue stores (-score, depth, url, parent_url), so higher scores
        are popped first (since PriorityQueue is min-heap based).
        """
        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        await queue.put((-1, 0, start_url, None))  # initial URL (score=1, stored as -1 in queue)
        visited: Set[str] = set()
        depths: Dict[str, int] = {start_url: 0}
        # Track best (highest) known score for each URL so we can update priorities
        best_scores: Dict[str, float] = {start_url: 1.0}

        while not queue.empty() and not self._cancel_event.is_set():
            if self._pages_crawled >= self.max_pages:
                self.logger.info(f"Max pages limit ({self.max_pages}) reached, stopping crawl")
                # Clean up scoring cache when stopping due to page limit
                self._score_cache.clear()
                # Clean up filter cache when stopping due to page limit
                self._filter_cache.clear()
                break

            """ # Log the current queue contents before popping
            queue_items = list(queue._queue)  # _queue is a deque storing internal items
            self.logger.info("Queue contents before batch:")
            for qscore, qdepth, qurl, qparent in queue_items:
                self.logger.info(f"  URL: {qurl}, Score: {-qscore}, Depth: {qdepth}, Parent: {qparent}") """

            remaining = self.max_pages - self._pages_crawled
            batch_size = min(self.batch_size, remaining)
            if batch_size <= 0:
                self.logger.info(f"Max pages limit ({self.max_pages}) reached, stopping crawl")
                break

            batch: List[Tuple[float, int, str, Optional[str]]] = []
            for _ in range(batch_size):
                if queue.empty():
                    break
                score, depth, url, parent_url = await queue.get()
                # Discard stale queue entries whose score is not the current best
                # (queue holds negative scores; convert to positive for comparison)
                current_best = best_scores.get(url)
                if current_best is not None and -score != current_best:
                    continue
                # skip URLs that are too deep

                if self.enforce_depth and depth > self.max_depth:
                    self.logger.debug(f"Skipping {url} (depth {depth} > max_depth {self.max_depth})")
                    continue
                # skip duplicates already processed
                if url in visited:
                    continue
                visited.add(url)
                batch.append((score, depth, url, parent_url))

            if not batch:
                continue

            urls = [item[2] for item in batch]
            batch_config = config.clone(deep_crawl_strategy=None, stream=True)
            stream_gen = await crawler.arun_many(urls=urls, config=batch_config)

            async for result in stream_gen:
                result_url = result.url
                corresponding = next((item for item in batch if item[2] == result_url), None)
                if not corresponding:
                    continue
                score, depth, url, parent_url = corresponding

                result.metadata = result.metadata or {}
                result.metadata["depth"] = depth
                result.metadata["parent_url"] = parent_url
                result.metadata["score"] = -score  # store actual positive score
                # Attach company_id from run config shared_data when provided
                try:
                    company_id = getattr(config, "shared_data", {}).get("company_id") if hasattr(config, "shared_data") else None
                    if company_id is not None:
                        result.metadata["company_id"] = company_id
                    start_url_sd = getattr(config, "shared_data", {}).get("start_url") if hasattr(config, "shared_data") else None
                    if start_url_sd is not None:
                        result.metadata["start_url"] = start_url_sd
                except Exception:
                    # Be resilient if shared_data is not a dict-like
                    pass

                if result.success:
                    # Check if this is a sitemap URL - if so, increase max_pages by 1 and don't count it
                    if self._is_sitemap_url(result_url):
                        self.max_pages += 1
                        self.logger.info(f"Sitemap URL detected: {result_url}. Increased max_pages to {self.max_pages}")
                    else:
                        # Assign crawl order index and record total score for this crawled URL
                        result.metadata["crawl_index"] = self._crawl_index_counter
                        self.crawl_order[result_url] = self._crawl_index_counter
                        self._crawl_index_counter += 1
                        self._pages_crawled += 1

                # Always yield the result we just processed
                yield result
                self.logger.debug(f"After processing {url}, queue size: {queue.qsize()}")

                if result.success:
                    # Log only URLs that actually got processed successfully
                    if self.save_discovered_urls:
                        try:
                            self._append_discovered_url(result_url, depth)
                        except Exception as e:
                            self.logger.warning(f"Failed to log processed URL {result_url}: {e}")

                    # If we hit the page limit after yielding/logging, stop further work
                    if self._pages_crawled >= self.max_pages:
                        self.logger.info(
                            f"Max pages limit ({self.max_pages}) reached during batch, stopping crawl"
                        )
                        # Clean up scoring cache when stopping due to page limit
                        self._score_cache.clear()
                        # Clean up filter cache when stopping due to page limit
                        self._filter_cache.clear()
                        break

                if result.success:
                    new_links: List[Tuple[str, Optional[str]]] = []
                    await self.link_discovery_with_depth(result, result_url, depth, visited, new_links, depths)

                    if new_links:
                        # Score links first, then limit by remaining capacity keeping highest scores
                        urls_to_score = [u for (u, _) in new_links if u not in visited]
                        # Compute scores concurrently
                        if self.url_scorer:
                            scores: List[float] = []
                            tasks: List[asyncio.Task] = []
                            task_indices: List[int] = []
                            # Prepare scores list with placeholders and gather tasks for uncached URLs
                            for idx, u in enumerate(urls_to_score):
                                if u in self._score_cache:
                                    scores.append(self._score_cache[u])
                                else:
                                    scores.append(0.0)  # placeholder
                                    tasks.append(asyncio.create_task(self.url_scorer.ascore(u)))
                                    task_indices.append(idx)
                            if tasks:
                                results = await asyncio.gather(*tasks)
                                for j, res in enumerate(results):
                                    i = task_indices[j]
                                    scores[i] = res
                                    # Memoize the computed score for this run
                                    self._score_cache[urls_to_score[i]] = res
                        else:
                            scores = [0.0 for _ in urls_to_score]

                        url_parent_map: Dict[str, Optional[str]] = {u: p for (u, p) in new_links}
                        new_depths: Dict[str, int] = {u: depths.get(u, depth + 1) for u in urls_to_score}

                        scored_candidates: List[Tuple[float, str]] = list(zip(scores, urls_to_score))
                        scored_candidates.sort(key=lambda t: t[0], reverse=True)

                        remaining_capacity = self.max_pages - self._pages_crawled
                        if remaining_capacity <= 0:
                            continue
                        top_k = scored_candidates[:remaining_capacity]

                        for candidate_score, candidate_url in top_k:
                            prev_best = best_scores.get(candidate_url)
                            if prev_best is None or candidate_score > prev_best:
                                best_scores[candidate_url] = candidate_score
                                candidate_depth = new_depths.get(candidate_url, depth + 1)
                                candidate_parent = url_parent_map.get(candidate_url)
                                await queue.put((-candidate_score, candidate_depth, candidate_url, candidate_parent))
                                self.logger.info(f'Enqueued/updated link: {candidate_url} with score {candidate_score}.')
        # Ensure cache does not linger after run ends
        self._score_cache.clear()