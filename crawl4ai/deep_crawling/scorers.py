from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse, unquote
from ..utils import HeadPeekr
from ..utils import get_text_embeddings, cosine_similarity
import numpy as np
import re
import logging
from functools import lru_cache
import inspect
from array import array
from rapidfuzz import fuzz
import ctypes
import platform
PLATFORM = platform.system()

# Pre-computed scores for common year differences
_SCORE_LOOKUP = [1.0, 0.5, 0.3333333333333333, 0.25]

# Pre-computed scores for common year differences
_FRESHNESS_SCORES = [
   1.0,    # Current year
   0.9,    # Last year
   0.8,    # 2 years ago
   0.7,    # 3 years ago
   0.6,    # 4 years ago
   0.5,    # 5 years ago
]

class ScoringStats:
    __slots__ = ('_urls_scored', '_total_score', '_min_score', '_max_score')
    
    def __init__(self):
        self._urls_scored = 0
        self._total_score = 0.0
        self._min_score = None  # Lazy initialization
        self._max_score = None
    
    def update(self, score: float) -> None:
        """Optimized update with minimal operations"""
        self._urls_scored += 1
        self._total_score += score
        
        # Lazy min/max tracking - only if actually accessed
        if self._min_score is not None:
            if score < self._min_score:
                self._min_score = score
        if self._max_score is not None:
            if score > self._max_score:
                self._max_score = score
                
    def get_average(self) -> float:
        """Direct calculation instead of property"""
        return self._total_score / self._urls_scored if self._urls_scored else 0.0
    
    def get_min(self) -> float:
        """Lazy min calculation"""
        if self._min_score is None:
            self._min_score = self._total_score / self._urls_scored if self._urls_scored else 0.0
        return self._min_score
        
    def get_max(self) -> float:
        """Lazy max calculation"""
        if self._max_score is None:
            self._max_score = self._total_score / self._urls_scored if self._urls_scored else 0.0
        return self._max_score
class URLScorer(ABC):
    __slots__ = ('_weight', '_stats', '_logger')
    
    def __init__(self, weight: float = 1.0, logger: Optional[logging.Logger] = None):
        # Store weight directly as float32 for memory efficiency
        self._weight = ctypes.c_float(weight).value
        self._stats = ScoringStats()
        # Dedicated logger for scorers; caller can inject a specific logger
        self._logger = logger or logging.getLogger("deep_crawl.scorer")
    
    @abstractmethod
    def _calculate_score(self, url: str) -> float:
        """Calculate raw score for URL."""
        pass
    
    def score(self, url: str) -> float:
        """Calculate weighted score with minimal overhead."""
        result = self._calculate_score(url)
        if inspect.isawaitable(result):
            raise RuntimeError("Async scorer used in sync path; call ascore(url) instead")
        score = result * self._weight
        self._stats.update(score)
        # Light-weight DEBUG log for visibility when enabled by caller's logger level
        try:
            self._logger.debug(f"scorer={self.__class__.__name__} url={url} score={score:.4f}")
        except Exception:
            pass
        return score

    async def ascore(self, url: str) -> float:
        """Async-friendly scoring: awaits if implementation returns awaitable."""
        result = self._calculate_score(url)
        if inspect.isawaitable(result):
            result = await result
        score = result * self._weight
        self._stats.update(score)
        try:
            self._logger.debug(f"scorer={self.__class__.__name__} url={url} score={score:.4f}")
        except Exception:
            pass
        return score
    
    @property
    def stats(self):
        """Access to scoring statistics."""
        return self._stats
    
    @property
    def weight(self):
        return self._weight

class CompositeScorer(URLScorer):
    __slots__ = ('_scorers', '_normalize', '_weights_array', '_score_array')
    
    def __init__(self, scorers: List[URLScorer], normalize: bool = True, logger: Optional[logging.Logger] = None):
        """Initialize composite scorer combining multiple scoring strategies.
        
        Optimized for:
        - Fast parallel scoring
        - Memory efficient score aggregation
        - Quick short-circuit conditions
        - Pre-allocated arrays
        
        Args:
            scorers: List of scoring strategies to combine
            normalize: Whether to normalize final score by scorer count
        """
        super().__init__(weight=1.0, logger=logger)
        self._scorers = scorers
        self._normalize = normalize
        
        # Pre-allocate arrays for scores and weights
        self._weights_array = array('f', [s.weight for s in scorers])
        self._score_array = array('f', [0.0] * len(scorers))

    @lru_cache(maxsize=10000)
    def _calculate_score(self, url: str) -> float:
        """Calculate combined score from all scoring strategies.
        
        Uses:
        1. Pre-allocated arrays for scores
        2. Short-circuit on zero scores
        3. Optimized normalization
        4. Vectorized operations where possible
        
        Args:
            url: URL to score
            
        Returns:
            Combined and optionally normalized score
        """
        total_score = 0.0
        scores = self._score_array
        
        # Get scores from all scorers
        for i, scorer in enumerate(self._scorers):
            # Use public score() method which applies weight
            scores[i] = scorer.score(url)
            total_score += scores[i]
            
        # Normalize if requested
        if self._normalize and self._scorers:
            count = len(self._scorers)
            return total_score / count
            
        return total_score

    def score(self, url: str) -> float:
        """Public scoring interface with stats tracking.
        
        Args:
            url: URL to score
            
        Returns:
            Final combined score
        """
        score = self._calculate_score(url)
        self.stats.update(score)
        return score

    async def ascore(self, url: str) -> float:
        """Async-friendly combined scoring using concurrent awaits."""
        # Import locally to avoid global dependency when unused
        import asyncio
        if not self._scorers:
            self.stats.update(0.0)
            return 0.0
        # Kick off async scoring for all scorers (all have ascore on base)
        tasks = [s.ascore(url) for s in self._scorers]
        results = await asyncio.gather(*tasks)
        total_score = sum(results)
        if self._normalize:
            total_score = total_score / len(self._scorers)
        self.stats.update(total_score)
        return total_score

class KeywordRelevanceScorer(URLScorer):
    __slots__ = ('_weight', '_stats', '_keywords', '_case_sensitive')
    
    def __init__(self, keywords: List[str], weight: float = 1.0, case_sensitive: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(weight=weight, logger=logger)
        self._case_sensitive = case_sensitive
        # Pre-process keywords once
        self._keywords = [k if case_sensitive else k.lower() for k in keywords]
    
    @lru_cache(maxsize=10000)
    def _url_bytes(self, url: str) -> bytes:
        """Cache decoded URL bytes"""
        return url.encode('utf-8') if self._case_sensitive else url.lower().encode('utf-8')
    
    
    def _calculate_score(self, url: str) -> float:
        """Fast string matching without regex or byte conversion"""
        if not self._case_sensitive:
            url = url.lower()
            
        matches = sum(1 for k in self._keywords if k in url)
        
        # Fast return paths
        if not matches:
            return 0.0
        if matches == len(self._keywords):
            return 1.0
            
        return matches / len(self._keywords)

class FuzzyKeywordRelevanceScorer(URLScorer):
    __slots__ = ('_weight', '_stats', '_keywords', '_case_sensitive')
    
    def __init__(self, keywords: List[str], weight: float = 1.0, case_sensitive: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(weight=weight, logger=logger)
        self._case_sensitive = case_sensitive
        self._keywords = [k if case_sensitive else k.lower() for k in keywords]

    @lru_cache(maxsize=10000)
    def _url_bytes(self, url: str) -> bytes:
        """Cache decoded URL bytes"""
        return url.encode('utf-8') if self._case_sensitive else url.lower().encode('utf-8')
    
    def _tokenize_url(self, url: str) -> List[str]:
        url = url if self._case_sensitive else url.lower()
        return re.findall(r"[a-z0-9äöüß]+", url)

    def _calculate_score(self, url: str) -> float:
        """Return 1.0 if any keyword matches any token with high similarity."""
        if not self._case_sensitive:
            url = url.lower()
        
        # Quick exact-like fuzzy match on whole URL
        if any(fuzz.ratio(k, url) > 80 for k in self._keywords):
            return 1.0
        
        # Tokenize
        url_tokens = self._tokenize_url(url)
        if not url_tokens:
            return 0.0
        
        # Check if any keyword is similar to any token
        for k in self._keywords:
            for token in url_tokens:
                if fuzz.token_set_ratio(k, token) > 85:
                    return 1.0  # High match → success
        
        # If no strong match found
        return 0.0

class WeightedFuzzyKeywordRelevanceScorer(URLScorer):
    __slots__ = (
        '_weight', '_stats', '_keyword_weights',
        '_case_sensitive', '_partial_penalty', '_fuzzy_penalty'
    )

    def __init__(
        self,
        keyword_weights: Dict[str, float],
        weight: float = 1.0,
        case_sensitive: bool = False,
        partial_penalty: float = 0.1,
        fuzzy_penalty: float = 0.15,
        logger: Optional[logging.Logger] = None
    ):
        """Weighted fuzzy keyword relevance scorer.

        Behaves like FuzzyKeywordRelevanceScorer but supports per-keyword weights and
        applies a small penalty when the match is not an exact token match, e.g.,
        "/produkte-alt" vs the exact keyword "produkte".
        """
        super().__init__(weight=weight, logger=logger)
        self._case_sensitive = case_sensitive
        # Normalize keywords according to case sensitivity
        self._keyword_weights = {
            (k if case_sensitive else k.lower()): float(w)
            for k, w in (keyword_weights or {}).items()
            if w is not None and float(w) > 0.0
        }
        self._partial_penalty = max(0.0, min(1.0, partial_penalty))
        self._fuzzy_penalty = max(0.0, min(1.0, fuzzy_penalty))

    @lru_cache(maxsize=10000)
    def _tokenize_url(self, url: str) -> List[str]:
        text = url if self._case_sensitive else url.lower()
        # Keep hyphens as part of tokens so "produkte-alt" remains a single token
        # This allows us to penalize partial matches like keyword "produkte" inside "produkte-alt".
        return re.findall(r"[a-z0-9äöüß-]+", text)

    @staticmethod
    def _is_exact_token_match(keyword: str, token: str) -> bool:
        return token == keyword

    @staticmethod
    def _is_subtoken_partial_match(keyword: str, token: str) -> bool:
        """True if keyword appears as a sub-token inside token (e.g., "produkte" in "produkte-alt")."""
        if token == keyword:
            return False
        # Consider hyphen/underscore separated subtokens as partial when containing keyword
        # Also accept generic substring as partial (e.g., …/produktealt)
        if keyword in token:
            return True
        return False

    def _best_match_score_for_keyword(self, keyword: str, url_tokens: List[str], full_url: str) -> float:
        # Quick whole-URL fuzzy boost similar to original logic
        url_text = full_url if self._case_sensitive else full_url.lower()
        if fuzz.ratio(keyword, url_text) > 80:
            return 1.0

        best = 0.0
        for token in url_tokens:
            if self._is_exact_token_match(keyword, token):
                best = max(best, 1.0)
                if best == 1.0:
                    break
            elif self._is_subtoken_partial_match(keyword, token):
                # Partial token match gets a small penalty
                best = max(best, max(0.0, 1.0 - self._partial_penalty))
            else:
                # Fuzzy similarity on token level
                if fuzz.token_set_ratio(keyword, token) > 85:
                    best = max(best, max(0.0, 1.0 - self._fuzzy_penalty))
        return best

    @lru_cache(maxsize=10000)
    def _calculate_score(self, url: str) -> float:
        if not self._keyword_weights:
            return 0.0

        url_tokens = self._tokenize_url(url)
        if not url_tokens:
            return 0.0

        # Find the keyword with the best match score
        best_weighted_score = 0.0
        for keyword, kw_weight in self._keyword_weights.items():
            match_score = self._best_match_score_for_keyword(keyword, url_tokens, url)
            # Use the best match score multiplied by its weight
            weighted_score = match_score * kw_weight
            best_weighted_score = max(best_weighted_score, weighted_score)

        return best_weighted_score

class PathDepthScorer(URLScorer):
    __slots__ = ('_weight', '_stats', '_optimal_depth')  # Remove _url_cache
    
    def __init__(self, optimal_depth: int = 3, weight: float = 1.0, logger: Optional[logging.Logger] = None):
        super().__init__(weight=weight, logger=logger)
        self._optimal_depth = optimal_depth

    @staticmethod
    @lru_cache(maxsize=10000)
    def _quick_depth(path: str) -> int:
        """Ultra fast path depth calculation.
        
        Examples:
            - "http://example.com" -> 0  # No path segments
            - "http://example.com/" -> 0  # Empty path
            - "http://example.com/a" -> 1
            - "http://example.com/a/b" -> 2
        """
        if not path or path == '/':
            return 0
            
        if '/' not in path:
            return 0
            
        depth = 0
        last_was_slash = True
        
        for c in path:
            if c == '/':
                if not last_was_slash:
                    depth += 1
                last_was_slash = True
            else:
                last_was_slash = False
                
        if not last_was_slash:
            depth += 1
            
        return depth

    @lru_cache(maxsize=10000)  # Cache the whole calculation
    def _calculate_score(self, url: str) -> float:
        pos = url.find('/', url.find('://') + 3)
        if pos == -1:
            depth = 0
        else:
            depth = self._quick_depth(url[pos:])
            
        # Use lookup table for common distances
        distance = depth - self._optimal_depth
        distance = distance if distance >= 0 else -distance  # Faster than abs()
        
        if distance < 4:
            return _SCORE_LOOKUP[distance]
            
        return 1.0 / (1.0 + distance)                                             

class ContentTypeScorer(URLScorer):
    __slots__ = ('_weight', '_exact_types', '_regex_types')

    def __init__(self, type_weights: Dict[str, float], weight: float = 1.0, logger: Optional[logging.Logger] = None):
        """Initialize scorer with type weights map.
        
        Args:
            type_weights: Dict mapping file extensions/patterns to scores (e.g. {'.html$': 1.0})
            weight: Overall weight multiplier for this scorer
        """
        super().__init__(weight=weight, logger=logger)
        self._exact_types = {}  # Fast lookup for simple extensions
        self._regex_types = []  # Fallback for complex patterns
        
        # Split into exact vs regex matchers for performance
        for pattern, score in type_weights.items():
            if pattern.startswith('.') and pattern.endswith('$'):
                ext = pattern[1:-1]
                self._exact_types[ext] = score
            else:
                self._regex_types.append((re.compile(pattern), score))
                
        # Sort complex patterns by score for early exit
        self._regex_types.sort(key=lambda x: -x[1])

    @staticmethod
    @lru_cache(maxsize=10000)
    def _quick_extension(url: str) -> str:
        """Extract file extension ultra-fast without regex/splits.
        
        Handles:
        - Basic extensions: "example.html" -> "html"
        - Query strings: "page.php?id=1" -> "php" 
        - Fragments: "doc.pdf#page=1" -> "pdf"
        - Path params: "file.jpg;width=100" -> "jpg"
        
        Args:
            url: URL to extract extension from
            
        Returns:
            Extension without dot, or empty string if none found
        """
        pos = url.rfind('.')
        if pos == -1:
            return ''
        
        # Find first non-alphanumeric char after extension
        end = len(url)
        for i in range(pos + 1, len(url)):
            c = url[i]
            # Stop at query string, fragment, path param or any non-alphanumeric
            if c in '?#;' or not c.isalnum():
                end = i
                break
                
        return url[pos + 1:end].lower()

    @lru_cache(maxsize=10000)
    def _calculate_score(self, url: str) -> float:
        """Calculate content type score for URL.
        
        Uses staged approach:
        1. Try exact extension match (fast path)
        2. Fall back to regex patterns if needed
        
        Args:
            url: URL to score
            
        Returns:
            Score between 0.0 and 1.0 * weight
        """
        # Fast path: direct extension lookup
        ext = self._quick_extension(url)
        if ext:
            score = self._exact_types.get(ext, None)
            if score is not None:
                return score
                
        # Slow path: regex patterns
        for pattern, score in self._regex_types:
            if pattern.search(url):
                return score

        return 0.0

class FreshnessScorer(URLScorer):
    __slots__ = ('_weight', '_date_pattern', '_current_year')

    def __init__(self, weight: float = 1.0, current_year: int = 2024, logger: Optional[logging.Logger] = None):
        """Initialize freshness scorer.
        
        Extracts and scores dates from URLs using format:
        - YYYY/MM/DD 
        - YYYY-MM-DD
        - YYYY_MM_DD
        - YYYY (year only)
        
        Args:
            weight: Score multiplier
            current_year: Year to calculate freshness against (default 2024)
        """
        super().__init__(weight=weight, logger=logger)
        self._current_year = current_year
        
        # Combined pattern for all date formats
        # Uses non-capturing groups (?:) and alternation
        self._date_pattern = re.compile(
            r'(?:/'  # Path separator
            r'|[-_])'  # or date separators
            r'((?:19|20)\d{2})'  # Year group (1900-2099)
            r'(?:'  # Optional month/day group
            r'(?:/|[-_])'  # Date separator  
            r'(?:\d{2})'  # Month
            r'(?:'  # Optional day
            r'(?:/|[-_])'  # Date separator
            r'(?:\d{2})'  # Day
            r')?'  # Day is optional
            r')?'  # Month/day group is optional
        )

    @lru_cache(maxsize=10000)
    def _extract_year(self, url: str) -> Optional[int]:
        """Extract the most recent year from URL.
        
        Args:
            url: URL to extract year from
            
        Returns:
            Year as int or None if no valid year found
        """
        matches = self._date_pattern.finditer(url)
        latest_year = None
        
        # Find most recent year
        for match in matches:
            year = int(match.group(1))
            if (year <= self._current_year and  # Sanity check
                (latest_year is None or year > latest_year)):
                latest_year = year
                
        return latest_year

    @lru_cache(maxsize=10000) 
    def _calculate_score(self, url: str) -> float:
        """Calculate freshness score based on URL date.
        
        More recent years score higher. Uses pre-computed scoring
        table for common year differences.
        
        Args:
            url: URL to score
            
        Returns:
            Score between 0.0 and 1.0 * weight
        """
        year = self._extract_year(url)
        if year is None:
            return 0.5  # Default score
            
        # Use lookup table for common year differences
        year_diff = self._current_year - year
        if year_diff < len(_FRESHNESS_SCORES):
            return _FRESHNESS_SCORES[year_diff]
            
        # Fallback calculation for older content
        return max(0.1, 1.0 - year_diff * 0.1)

class DomainAuthorityScorer(URLScorer):
    __slots__ = ('_weight', '_domain_weights', '_default_weight', '_top_domains')
    
    def __init__(
        self,
        domain_weights: Dict[str, float],
        default_weight: float = 0.5,
        weight: float = 1.0,
    ):
        """Initialize domain authority scorer.
        
        Args:
            domain_weights: Dict mapping domains to authority scores
            default_weight: Score for unknown domains
            weight: Overall scorer weight multiplier
            
        Example:
            {
                'python.org': 1.0,
                'github.com': 0.9,
                'medium.com': 0.7
            }
        """
        super().__init__(weight=weight)
        
        # Pre-process domains for faster lookup
        self._domain_weights = {
            domain.lower(): score 
            for domain, score in domain_weights.items()
        }
        self._default_weight = default_weight
        
        # Cache top domains for fast path
        self._top_domains = {
            domain: score
            for domain, score in sorted(
                domain_weights.items(), 
                key=lambda x: -x[1]
            )[:5]  # Keep top 5 highest scoring domains
        }

    @staticmethod
    @lru_cache(maxsize=10000)
    def _extract_domain(url: str) -> str:
        """Extract domain from URL ultra-fast.
        
        Handles:
        - Basic domains: "example.com"
        - Subdomains: "sub.example.com" 
        - Ports: "example.com:8080"
        - IPv4: "192.168.1.1"
        
        Args:
            url: Full URL to extract domain from
            
        Returns:
            Lowercase domain without port
        """
        # Find domain start
        start = url.find('://') 
        if start == -1:
            start = 0
        else:
            start += 3
            
        # Find domain end
        end = url.find('/', start)
        if end == -1:
            end = url.find('?', start)
            if end == -1:
                end = url.find('#', start)
                if end == -1:
                    end = len(url)
                    
        # Extract domain and remove port
        domain = url[start:end]
        port_idx = domain.rfind(':')
        if port_idx != -1:
            domain = domain[:port_idx]
            
        return domain.lower()

    @lru_cache(maxsize=10000)
    def _calculate_score(self, url: str) -> float:
        """Calculate domain authority score.
        
        Uses staged approach:
        1. Check top domains (fastest)
        2. Check full domain weights
        3. Return default weight
        
        Args:
            url: URL to score
            
        Returns:
            Authority score between 0.0 and 1.0 * weight
        """
        domain = self._extract_domain(url)
        
        # Fast path: check top domains first
        score = self._top_domains.get(domain)
        if score is not None:
            return score
            
        # Regular path: check all domains
        return self._domain_weights.get(domain, self._default_weight)

class EmbeddingScorer(URLScorer):
    __slots__ = (
        '_weight', '_stats',
        '_reference_embedding', '_reference_source',
        '_embedding_model', '_embedding_llm_config'
    )

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_llm_config: Optional[Dict] = None,
        reference: Optional[Any] = None,
        weight: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """Embedding-based URL scorer using page head metadata when available.

        Args:
            embedding_model: SentenceTransformers model name (when llm_config is None)
            embedding_llm_config: Config for API-based embeddings (see get_text_embeddings)
            reference: Reference text(s) or precomputed embedding vector to compare against
            weight: Weight multiplier for this scorer
        """
        super().__init__(weight=weight, logger=logger)
        self._embedding_model = embedding_model
        self._embedding_llm_config = embedding_llm_config
        self._reference_embedding = None 
        self._reference_source = reference

    async def _ensure_reference_embedding(self, reference: Any) -> Optional[np.ndarray]:
        """Ensure we have a single reference embedding vector, computing it if needed."""
        # If already a numpy vector
        if isinstance(reference, np.ndarray):
            return reference
        # If list of texts or single text
        if isinstance(reference, (str, list, tuple)):
            texts = [reference] if isinstance(reference, str) else list(reference)
            vectors = await get_text_embeddings(
                texts=texts,
                llm_config=self._embedding_llm_config,
                model_name=self._embedding_model,
                batch_size=32
            )
            if vectors.size == 0:
                return None
            # If multiple texts provided, average them
            if len(texts) > 1:
                return np.mean(vectors, axis=0)
            return vectors[0]
        # Unknown format
        return None

    # # TODO: remove static and add self
    # @staticmethod
    async def _assemble_page_signature(self, url: str) -> str:
        """Build a compact textual signature from head metadata and title.
        Falls back to the URL if metadata isn't available.
        """
        try:
            head_content = await HeadPeekr.peek_html(url)

            if not head_content:
                return url

            title = HeadPeekr.get_title(head_content) or ""
            meta = HeadPeekr.extract_meta_tags(head_content) or {}

            description = meta.get("description", "")
            keywords = meta.get("keywords", "")
            og_title = meta.get("og:title", "")
            og_description = meta.get("og:description", "")

            if keywords:
                return keywords
            if description:
                if title:
                    return " ".join([title, description])
                else:
                    return description
            if og_title and not title:
                if og_description:
                    return " ".join([og_title, og_description])
                else:
                    return og_title
            else:
                return " ".join(re.findall(r"[a-z0-9äöüß]+", urlparse(url).path.lower()))

        except Exception:
            return url

    async def _calculate_score(self, url: str) -> float:
        """Calculate similarity between page metadata embedding and reference embedding."""
        # Ensure reference embedding exists
        if self._reference_embedding is None and self._reference_source is not None:
            self._reference_embedding = await self._ensure_reference_embedding(self._reference_source)

        if self._reference_embedding is None:
            return 0.0

        text = await self._assemble_page_signature(url)
        vectors = await get_text_embeddings(
            texts=[text],
            llm_config=self._embedding_llm_config,
            model_name=self._embedding_model,
            batch_size=8
        )
        if vectors.size == 0:
            return 0.0
        url_vec = vectors[0]
        return cosine_similarity(url_vec, self._reference_embedding)

class EmbeddingKeywordScorer(URLScorer):
    __slots__ = (
        '_weight', '_stats',
        '_reference_embeddings', '_keywords',
        '_embedding_model', '_embedding_llm_config'
    )

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_llm_config: Optional[Dict] = None,
        keywords: Optional[List[str]] = None,
        weight: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """Embedding-based URL scorer using page head metadata when available.

        Args:
            embedding_model: SentenceTransformers model name (when llm_config is None)
            embedding_llm_config: Config for API-based embeddings (see get_text_embeddings)
            keywords: Keywords to compare against
            weight: Weight multiplier for this scorer
        """
        super().__init__(weight=weight, logger=logger)
        self._embedding_model = embedding_model
        self._embedding_llm_config = embedding_llm_config
        self._reference_embeddings = None 
        self._keywords = [k.lower() for k in keywords]

    async def _ensure_keywords_embeddings(self, keywords: List[str]) -> Optional[np.ndarray]:
        """Ensure we have a single reference embedding vector, computing it if needed."""
        vectors = await get_text_embeddings(
            texts=keywords,
            llm_config=self._embedding_llm_config,
            model_name=self._embedding_model,
            batch_size=32
        )
        return vectors

    async def _assemble_page_signature(self, url: str) -> str:
        """Build a compact textual signature from head metadata and title.
        Falls back to the URL if metadata isn't available.
        """
        try:
            head_content = await HeadPeekr.peek_html(url)

            if not head_content:
                return " ".join(re.findall(r"[a-z0-9äöüß]+", urlparse(url).path.lower()))

            title = HeadPeekr.get_title(head_content) or ""
            meta = HeadPeekr.extract_meta_tags(head_content) or {}

            description = meta.get("description", "")
            keywords = meta.get("keywords", "")
            og_title = meta.get("og:title", "")
            og_description = meta.get("og:description", "")

            if keywords:
                return keywords
            # if description:
            #     if title:
            #         return " ".join([title, description])
            #     else:
            #         return description
            # if og_title and not title:
            #     if og_description:
            #         return " ".join([og_title, og_description])
            #     else:
            #         return og_title
            else:
                return " ".join(re.findall(r"[a-z0-9äöüß]+", urlparse(url).path.lower()))

        except Exception:
            return " ".join(re.findall(r"[a-z0-9äöüß]+", urlparse(url).path.lower()))

    async def _calculate_score(self, url: str) -> float:
        """Calculate maximum similarity between page metadata embedding and reference embeddings."""
        # Ensure reference embedding exists
        if self._reference_embeddings is None and self._keywords is not None:
            self._reference_embeddings = await self._ensure_keywords_embeddings(self._keywords)

        if self._reference_embeddings is None:
            return 0.0

        text = await self._assemble_page_signature(url)
        vectors = await get_text_embeddings(
            texts=[text],
            llm_config=self._embedding_llm_config,
            model_name=self._embedding_model,
            batch_size=8
        )
        if vectors.size == 0:
            return 0.0
        url_vec = vectors[0]
        similarities = []
        print("URL: ", url, " with text: ", text)
        for k_vec in self._reference_embeddings:
            similarities.append(cosine_similarity(url_vec, k_vec))
        print("max similarity:", max(similarities))
        return max(cosine_similarity(url_vec, k_vec) for k_vec in self._reference_embeddings)