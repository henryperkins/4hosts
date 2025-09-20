"""
Source Credibility Scoring System for Four Hosts Research Application
Implements domain authority checking, bias detection, and source reputation scoring.
"""

import asyncio
import aiohttp
# json unused
import logging
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from utils.date_utils import calculate_age_days, get_current_utc, iso_or_none
# urlparse unused
# re unused
import os
import math
from collections import defaultdict

from .cache import cache_manager
from .brave_grounding import brave_client
from utils.retry import calculate_exponential_backoff, handle_rate_limit

logger = structlog.get_logger(__name__)


@dataclass
class CredibilityScore:
    """Represents the credibility assessment of a source"""

    domain: str
    overall_score: float  # 0.0 to 1.0
    domain_authority: Optional[float] = None  # 0-100 scale
    bias_rating: Optional[str] = None  # "left", "center", "right", "mixed"
    bias_score: Optional[float] = None  # 0.0 (extreme bias) to 1.0 (neutral)
    fact_check_rating: Optional[str] = None  # "high", "medium", "low", "mixed"
    paradigm_alignment: Dict[str, float] = field(default_factory=dict)
    reputation_factors: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=get_current_utc)
    
    # New comprehensive features
    recency_score: float = 1.0  # Temporal relevance (0.0 to 1.0)
    cross_source_agreement: Optional[float] = None
    controversy_score: float = 0.0
    update_frequency: Optional[str] = None
    social_proof_score: Optional[float] = None
    source_category: Optional[str] = None
    topic_credibility: Dict[str, float] = field(default_factory=dict)  # Topic-specific scores
    controversy_indicators: List[str] = field(default_factory=list)  # Specific controversy markers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "domain": self.domain,
            "overall_score": self.overall_score,
            "domain_authority": self.domain_authority,
            "bias_rating": self.bias_rating,
            "bias_score": self.bias_score,
            "fact_check_rating": self.fact_check_rating,
            "paradigm_alignment": self.paradigm_alignment,
            "reputation_factors": self.reputation_factors,
            "last_updated": iso_or_none(self.last_updated),
            "recency_score": self.recency_score,
            "cross_source_agreement": self.cross_source_agreement,
            "controversy_score": self.controversy_score,
            "update_frequency": self.update_frequency,
            "social_proof_score": self.social_proof_score,
            "source_category": self.source_category,
            "topic_credibility": self.topic_credibility,
            "controversy_indicators": self.controversy_indicators,
        }
    
    def generate_credibility_card(self) -> Dict[str, Any]:
        """Generate a structured credibility assessment card"""
        # Calculate trust level
        if self.overall_score >= 0.8:
            trust_level = "Very High"
            trust_color = "green"
        elif self.overall_score >= 0.6:
            trust_level = "High"
            trust_color = "light-green"
        elif self.overall_score >= 0.4:
            trust_level = "Moderate"
            trust_color = "yellow"
        elif self.overall_score >= 0.2:
            trust_level = "Low"
            trust_color = "orange"
        else:
            trust_level = "Very Low"
            trust_color = "red"
        
        # Generate recommendations
        recommendations = []
        if self.controversy_score > 0.7:
            recommendations.append("Cross-reference with multiple sources due to high controversy")
        if self.bias_score and self.bias_score < 0.5:
            recommendations.append("Be aware of potential political bias")
        if self.recency_score < 0.5:
            recommendations.append("Information may be outdated, seek recent sources")
        if self.fact_check_rating == "low":
            recommendations.append("Verify facts independently")
        
        return {
            "domain": self.domain,
            "trust_level": trust_level,
            "trust_color": trust_color,
            "overall_score": round(self.overall_score, 2),
            "key_factors": {
                "Authority": f"{self.domain_authority or 'Unknown'}/100" if self.domain_authority else "Unknown",
                "Bias": self.bias_rating or "Unknown",
                "Factual Accuracy": self.fact_check_rating or "Unknown",
                "Controversy": "High" if self.controversy_score > 0.7 else "Low" if self.controversy_score < 0.3 else "Moderate",
                "Category": self.source_category or "Unknown",
            },
            "strengths": [f for f in self.reputation_factors if "High" in f or "Neutral" in f],
            "concerns": self.controversy_indicators + [f for f in self.reputation_factors if "Low" in f],
            "recommendations": recommendations,
            "paradigm_scores": {k: round(v, 2) for k, v in self.paradigm_alignment.items()},
            "last_updated": self.last_updated.strftime("%Y-%m-%d %H:%M:%S"),
        }


class DomainAuthorityChecker:
    """Checks domain authority using various APIs"""

    def __init__(self):
        self.moz_api_key = os.getenv("MOZ_API_KEY")
        self.moz_secret_key = os.getenv("MOZ_SECRET_KEY")
        self.session: Optional[aiohttp.ClientSession] = None
        self.moz_failures = 0
        self.moz_backoff_until = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_domain_authority(self, domain: str) -> Optional[float]:
        """Get domain authority score (0-100)"""

        # Try cached DA first (separate key namespace)
        try:
            # Prefer new KV API, but fall back gracefully if unavailable
            get_kv = getattr(cache_manager, "get_kv", None)
            if callable(get_kv):
                cached_da = await get_kv(f"cred:da:{domain}")
                if cached_da is not None:
                    return float(cached_da)
            else:
                # Polyfill path using existing credibility namespace
                cached_card = await cache_manager.get_source_credibility(f"cred:da:{domain}")
                if cached_card is not None:
                    try:
                        return float(cached_card)
                    except Exception:
                        pass
        except Exception:
            pass

        # Try Moz API if available
        if self.moz_api_key and self.moz_secret_key:
            moz_da = await self._get_moz_domain_authority(domain)
            if moz_da is not None:
                return moz_da

        # Fallback to heuristic scoring
        return await self._heuristic_domain_authority(domain)

    async def _get_moz_domain_authority(self, domain: str) -> Optional[float]:
        """Get domain authority from Moz API with exponential backoff on repeated failures"""

        # Check if we're in backoff period
        if self.moz_backoff_until and get_current_utc() < self.moz_backoff_until:
            logger.debug(f"Moz API in backoff period, skipping DA check for {domain}")
            return None
        
        try:
            import hmac
            import hashlib
            import base64
            from urllib.parse import quote

            # Moz API authentication
            expires = int((get_current_utc() + timedelta(minutes=5)).timestamp())
            string_to_sign = f"{self.moz_api_key}\n{expires}"
            signature = base64.b64encode(
                hmac.new(
                    self.moz_secret_key.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    hashlib.sha1,
                ).digest()
            ).decode("utf-8")

            url = f"https://lsapi.seomoz.com/v2/url_metrics"

            # Correct payload for POST request
            payload = {"targets": [domain]}

            # Correct authentication
            auth = (self.moz_api_key, self.moz_secret_key)

            async with self.session.post(
                url, auth=aiohttp.BasicAuth(*auth), json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract Domain Authority or closest metric from the response
                    if "results" in data and data["results"]:
                        # Reset failure count on success
                        self.moz_failures = 0
                        self.moz_backoff_until = None
                        # Prefer domain_authority if available; fallback to page_authority
                        metrics = data["results"][0]
                        da_val = metrics.get("domain_authority") or metrics.get("page_authority")
                        try:
                            if da_val is not None:
                                # Cache DA with KV if available, else fallback
                                setter = getattr(cache_manager, "set_kv", None)
                                if callable(setter):
                                    await setter(f"cred:da:{domain}", float(da_val), ttl=86400)
                                else:
                                    await cache_manager.set_source_credibility(f"cred:da:{domain}", float(da_val))
                                return float(da_val)
                        except Exception:
                            return da_val
                    return None
                else:
                    error_text = await response.text()
                    logger.warning("Moz API error for %s: %s - %s", domain, response.status, error_text)
                    
                    # Handle auth and rate/5xx errors with exponential backoff
                    if response.status in (401, 429, 500, 502, 503, 504):
                        self.moz_failures += 1
                        if self.moz_failures >= 3:
                            # Use exponential backoff calculation for longer failure periods
                            backoff_seconds = calculate_exponential_backoff(
                                attempt=self.moz_failures,
                                base_delay=60,  # Start with 1 minute
                                factor=2.0,
                                max_delay=3600,  # Cap at 1 hour
                                jitter_mode="full"
                            )
                            self.moz_backoff_until = get_current_utc() + timedelta(seconds=backoff_seconds)
                            logger.warning("Moz API failing repeatedly, backing off for %d seconds", backoff_seconds)
                        else:
                            logger.debug("Moz API recoverable error (%d/3), will retry", self.moz_failures)

                        # Handle rate limiting headers if present
                        if response.status == 429:
                            await handle_rate_limit(
                                url=url,
                                response_headers=response.headers,
                                attempt=self.moz_failures,
                                prefer_server=True,
                                base_delay=60,
                                max_delay=3600
                            )

        except Exception as e:
            # Prefer structured status where possible
            status = None
            try:
                if hasattr(e, "status"):
                    status = int(getattr(e, "status"))
            except Exception:
                status = None

            msg = str(e)
            if status == 401 or "401" in msg:
                self.moz_failures += 1
                if self.moz_failures >= 3:
                    # Use exponential backoff for authentication failures
                    backoff_seconds = calculate_exponential_backoff(
                        attempt=self.moz_failures,
                        base_delay=60,
                        factor=2.0,
                        max_delay=3600,
                        jitter_mode="full"
                    )
                    self.moz_backoff_until = get_current_utc() + timedelta(seconds=backoff_seconds)
                    logger.warning("Moz API failing repeatedly (401), backing off for %d seconds", backoff_seconds)
                else:
                    logger.debug("Moz API 401 error (%d/3), will retry", self.moz_failures)
            else:
                logger.error("Moz API error: %s", msg)

        return None

    async def _heuristic_domain_authority(self, domain: str) -> float:
        """Heuristic domain authority based on known high-authority domains"""
        from utils.domain_categorizer import categorize

        # High authority domains (score 80-100)
        high_authority = {
            "wikipedia.org": 95,
            "google.com": 100,
            "youtube.com": 95,
            "facebook.com": 95,
            "twitter.com": 92,
            "linkedin.com": 90,
            "github.com": 88,
            "stackoverflow.com": 88,
            "reddit.com": 85,
            "medium.com": 82,
            # News sources
            "nytimes.com": 90,
            "washingtonpost.com": 88,
            "bbc.com": 92,
            "cnn.com": 85,
            "reuters.com": 88,
            "apnews.com": 87,
            # Academic
            "nature.com": 90,
            "science.org": 88,
            "arxiv.org": 85,
            "pubmed.ncbi.nlm.nih.gov": 90,
            # Government
            "gov": 85,  # Any .gov domain
            "edu": 80,  # Any .edu domain
        }

        # Check exact matches first
        if domain in high_authority:
            return high_authority[domain]

        # Use domain categorizer for consistent scoring
        category = categorize(domain)
        category_scores = {
            "government": 85,
            "academic": 80,
            "news": 70,
            "reference": 75,
            "blog": 60,
            "social": 50,
            "tech": 55,
            "video": 50,
            "pdf": 65,
            "other": 40
        }

        return category_scores.get(category, 40)


class BiasDetector:
    """Detects political bias and factual accuracy of sources"""

    def __init__(self):
        # Load bias ratings from AllSides, Media Bias/Fact Check, etc.
        self.bias_database = self._load_bias_database()

    def _load_bias_database(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive bias ratings database with expanded sources"""
        return {
            # News Sources - Left-leaning
            "cnn.com": {"bias": "left", "bias_score": 0.3, "factual": "mixed", "category": "news", "update_freq": "real-time"},
            "msnbc.com": {"bias": "left", "bias_score": 0.2, "factual": "mixed", "category": "news", "update_freq": "real-time"},
            "huffpost.com": {"bias": "left", "bias_score": 0.1, "factual": "mixed", "category": "news", "update_freq": "daily"},
            "theguardian.com": {"bias": "left", "bias_score": 0.3, "factual": "high", "category": "news", "update_freq": "real-time"},
            "washingtonpost.com": {"bias": "left", "bias_score": 0.4, "factual": "high", "category": "news", "update_freq": "real-time"},
            "nytimes.com": {"bias": "left", "bias_score": 0.4, "factual": "high", "category": "news", "update_freq": "real-time"},
            "vox.com": {"bias": "left", "bias_score": 0.3, "factual": "high", "category": "news", "update_freq": "daily"},
            "slate.com": {"bias": "left", "bias_score": 0.3, "factual": "mixed", "category": "news", "update_freq": "daily"},
            "motherjones.com": {"bias": "left", "bias_score": 0.2, "factual": "high", "category": "news", "update_freq": "daily"},
            "thedailybeast.com": {"bias": "left", "bias_score": 0.3, "factual": "mixed", "category": "news", "update_freq": "daily"},
            
            # News Sources - Right-leaning
            "foxnews.com": {"bias": "right", "bias_score": 0.2, "factual": "mixed", "category": "news", "update_freq": "real-time"},
            "breitbart.com": {"bias": "right", "bias_score": 0.1, "factual": "low", "category": "news", "update_freq": "daily"},
            "dailywire.com": {"bias": "right", "bias_score": 0.2, "factual": "mixed", "category": "news", "update_freq": "daily"},
            "wsj.com": {"bias": "right", "bias_score": 0.4, "factual": "high", "category": "news", "update_freq": "real-time"},
            "theblaze.com": {"bias": "right", "bias_score": 0.2, "factual": "mixed", "category": "news", "update_freq": "daily"},
            "newsmax.com": {"bias": "right", "bias_score": 0.2, "factual": "low", "category": "news", "update_freq": "daily"},
            "oann.com": {"bias": "right", "bias_score": 0.1, "factual": "low", "category": "news", "update_freq": "daily"},
            "nationalreview.com": {"bias": "right", "bias_score": 0.3, "factual": "high", "category": "news", "update_freq": "daily"},
            
            # News Sources - Center/Neutral
            "reuters.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "news", "update_freq": "real-time"},
            "apnews.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "news", "update_freq": "real-time"},
            "bbc.com": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "news", "update_freq": "real-time"},
            "npr.org": {"bias": "center", "bias_score": 0.7, "factual": "high", "category": "news", "update_freq": "real-time"},
            "politico.com": {"bias": "center", "bias_score": 0.7, "factual": "high", "category": "news", "update_freq": "real-time"},
            "thehill.com": {"bias": "center", "bias_score": 0.7, "factual": "high", "category": "news", "update_freq": "real-time"},
            "axios.com": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "news", "update_freq": "daily"},
            "csmonitor.com": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "news", "update_freq": "daily"},
            
            # Academic/Scientific Sources
            "nature.com": {"bias": "center", "bias_score": 0.95, "factual": "high", "category": "academic", "update_freq": "weekly"},
            "science.org": {"bias": "center", "bias_score": 0.95, "factual": "high", "category": "academic", "update_freq": "weekly"},
            "arxiv.org": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "academic", "update_freq": "daily"},
            "pubmed.ncbi.nlm.nih.gov": {"bias": "center", "bias_score": 0.95, "factual": "high", "category": "academic", "update_freq": "daily"},
            "scholar.google.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "academic", "update_freq": "real-time"},
            "jstor.org": {"bias": "center", "bias_score": 0.95, "factual": "high", "category": "academic", "update_freq": "monthly"},
            "researchgate.net": {"bias": "center", "bias_score": 0.85, "factual": "high", "category": "academic", "update_freq": "daily"},
            "plos.org": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "academic", "update_freq": "weekly"},
            "springer.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "academic", "update_freq": "monthly"},
            "wiley.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "academic", "update_freq": "monthly"},
            
            # Reference Sources
            "wikipedia.org": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "reference", "update_freq": "real-time"},
            "britannica.com": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "reference", "update_freq": "monthly"},
            "snopes.com": {"bias": "center", "bias_score": 0.85, "factual": "high", "category": "fact-check", "update_freq": "daily"},
            "factcheck.org": {"bias": "center", "bias_score": 0.9, "factual": "high", "category": "fact-check", "update_freq": "daily"},
            "politifact.com": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "fact-check", "update_freq": "daily"},
            
            # Tech/Blog Sources
            "techcrunch.com": {"bias": "center", "bias_score": 0.7, "factual": "high", "category": "tech", "update_freq": "daily"},
            "wired.com": {"bias": "center", "bias_score": 0.6, "factual": "high", "category": "tech", "update_freq": "daily"},
            "arstechnica.com": {"bias": "center", "bias_score": 0.8, "factual": "high", "category": "tech", "update_freq": "daily"},
            "medium.com": {"bias": "mixed", "bias_score": 0.5, "factual": "mixed", "category": "blog", "update_freq": "real-time"},
            "substack.com": {"bias": "mixed", "bias_score": 0.5, "factual": "mixed", "category": "blog", "update_freq": "daily"},
            
            # Social Media
            "twitter.com": {"bias": "mixed", "bias_score": 0.3, "factual": "low", "category": "social", "update_freq": "real-time"},
            "reddit.com": {"bias": "mixed", "bias_score": 0.4, "factual": "mixed", "category": "social", "update_freq": "real-time"},
            "facebook.com": {"bias": "mixed", "bias_score": 0.3, "factual": "low", "category": "social", "update_freq": "real-time"},
            "youtube.com": {"bias": "mixed", "bias_score": 0.4, "factual": "mixed", "category": "social", "update_freq": "real-time"},
        }

    async def get_bias_score(
        self, domain: str
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Get bias rating, bias score, and factual rating for domain"""

        # Check cache first
        cached_result = await cache_manager.get_source_credibility(domain)
        if cached_result:
            return (
                cached_result.get("bias_rating"),
                cached_result.get("bias_score"),
                cached_result.get("fact_check_rating"),
            )

        # Check our database
        if domain in self.bias_database:
            data = self.bias_database[domain]
            return data["bias"], data["bias_score"], data["factual"]

        # Heuristic scoring for unknown domains
        return await self._heuristic_bias_score(domain)

    async def _heuristic_bias_score(self, domain: str) -> Tuple[str, float, str]:
        """Heuristic bias scoring for unknown domains"""
        from utils.domain_categorizer import categorize

        category = categorize(domain)

        # Category-based bias and factual scoring
        category_scores = {
            "government": ("center", 0.8, "high"),
            "academic": ("center", 0.8, "high"),
            "news": ("center", 0.7, "high"),
            "reference": ("center", 0.8, "high"),
            "blog": ("center", 0.7, "medium"),  # NGOs/blogs can have bias but often factual
            "social": ("center", 0.4, "low"),
            "tech": ("center", 0.6, "medium"),
            "video": ("center", 0.5, "medium"),
            "pdf": ("center", 0.7, "medium"),
            "other": ("center", 0.5, "medium")
        }

        return category_scores.get(category, ("center", 0.5, "medium"))


class ControversyDetector:
    """Detects and scores controversial topics and content"""
    
    def __init__(self):
        # Controversial topics and keywords
        self.controversial_topics = {
            "politics": {
                "keywords": ["election", "democrat", "republican", "trump", "biden", "congress", "senate", "impeach"],
                "weight": 0.8
            },
            "social_issues": {
                "keywords": ["abortion", "gun control", "immigration", "racism", "lgbtq", "gender", "trans", "blm"],
                "weight": 0.9
            },
            "health": {
                "keywords": ["vaccine", "covid", "mask mandate", "lockdown", "ivermectin", "hydroxychloroquine"],
                "weight": 0.7
            },
            "climate": {
                "keywords": ["climate change", "global warming", "fossil fuel", "renewable energy", "carbon tax"],
                "weight": 0.6
            },
            "religion": {
                "keywords": ["christian", "muslim", "atheist", "evolution", "creationism", "religious freedom"],
                "weight": 0.7
            },
            "economics": {
                "keywords": ["capitalism", "socialism", "wealth inequality", "minimum wage", "universal basic income"],
                "weight": 0.6
            },
        }
        
        # Polarizing sources that often cover controversial topics
        self.polarizing_sources = {
            "breitbart.com": 0.9,
            "infowars.com": 1.0,
            "dailykos.com": 0.8,
            "commondreams.org": 0.7,
            "thegatewaypundit.com": 0.9,
            "occupy.com": 0.8,
        }
    
    def calculate_controversy_score(self, domain: str, content: Optional[str] = None, 
                                   search_terms: Optional[List[str]] = None) -> Tuple[float, List[str]]:
        """Calculate controversy score and identify indicators"""
        score = 0.0
        indicators = []
        
        # Check if domain is known to be polarizing
        if domain in self.polarizing_sources:
            score += self.polarizing_sources[domain] * 0.4
            indicators.append(f"Polarizing source: {domain}")
        
        # Analyze search terms and content for controversial topics
        if search_terms or content:
            text_to_analyze = " ".join(search_terms or []) + " " + (content or "")
            text_lower = text_to_analyze.lower()
            
            for topic, info in self.controversial_topics.items():
                matches = sum(1 for keyword in info["keywords"] if keyword in text_lower)
                if matches > 0:
                    topic_score = min(matches * 0.1, 0.5) * info["weight"]
                    score += topic_score
                    if matches >= 2:
                        indicators.append(f"Controversial topic: {topic.replace('_', ' ').title()}")
        
        # Check for conflicting viewpoints indicator words
        conflict_words = ["debate", "controversy", "disputed", "conflicting", "polarizing", 
                         "divisive", "contentious", "opposing views", "critics argue"]
        if content:
            content_lower = content.lower()
            conflict_matches = sum(1 for word in conflict_words if word in content_lower)
            if conflict_matches > 0:
                score += min(conflict_matches * 0.05, 0.2)
                if conflict_matches >= 3:
                    indicators.append("Contains conflicting viewpoints")
        
        # Normalize score to 0-1 range
        final_score = min(score, 1.0)
        
        return final_score, indicators
    
    def detect_conflicting_sources(self, sources: List[str]) -> float:
        """Detect if sources have conflicting biases"""
        if len(sources) < 2:
            return 0.0
        
        # Count sources from different bias categories
        bias_counts = defaultdict(int)
        for source in sources:
            if "foxnews" in source or "breitbart" in source or "dailywire" in source:
                bias_counts["right"] += 1
            elif "cnn" in source or "msnbc" in source or "huffpost" in source:
                bias_counts["left"] += 1
            else:
                bias_counts["center"] += 1
        
        # High conflict if both left and right sources are present
        if bias_counts["left"] > 0 and bias_counts["right"] > 0:
            total = sum(bias_counts.values())
            conflict_ratio = (bias_counts["left"] + bias_counts["right"]) / total
            return min(conflict_ratio * 0.8, 1.0)
        
        return 0.0


class RecencyModeler:
    """Models temporal credibility factors and recency decay"""
    
    def __init__(self):
        # Domain-specific decay rates (days to 50% credibility)
        self.decay_rates = {
            "news": 7,  # News loses relevance quickly
            "academic": 365,  # Academic papers stay relevant longer
            "government": 180,  # Government info moderately stable
            "reference": 730,  # Reference materials very stable
            "tech": 30,  # Tech news becomes outdated quickly
            "blog": 14,  # Blog posts lose relevance moderately fast
            "social": 1,  # Social media very ephemeral
            "fact-check": 90,  # Fact checks remain relevant for months
        }
        
        # Breaking news boost parameters
        self.breaking_news_window = timedelta(hours=24)
        self.breaking_news_boost = 0.2  # 20% boost for breaking news
    
    def calculate_recency_score(self, published_date: Optional[datetime], 
                               category: Optional[str] = None,
                               is_breaking: bool = False) -> float:
        """Calculate recency score with exponential decay"""
        if not published_date:
            return 0.5  # Unknown date gets neutral score

        now = get_current_utc()
        age = now - published_date

        # Apply breaking news boost if applicable
        if is_breaking and age < self.breaking_news_window:
            boost = self.breaking_news_boost * (1 - age.total_seconds() / self.breaking_news_window.total_seconds())
        else:
            boost = 0

        # Get decay rate for category
        decay_days = self.decay_rates.get(category, 30)  # Default 30 days

        # Calculate exponential decay
        # Score = e^(-ln(2) * days_old / half_life)
        days_old = calculate_age_days(published_date) or 0
        decay_score = math.exp(-0.693 * days_old / decay_days)
        
        # Combine decay score with any boost
        final_score = min(decay_score + boost, 1.0)
        
        return final_score
    
    def get_update_frequency(self, domain: str, recent_dates: List[datetime]) -> str:
        """Determine update frequency based on recent publication dates"""
        if not recent_dates:
            return "rarely"
        
        # Sort dates and calculate intervals
        sorted_dates = sorted(recent_dates, reverse=True)
        if len(sorted_dates) < 2:
            return "rarely"
        
        # Calculate average interval between updates
        intervals = []
        for i in range(min(len(sorted_dates) - 1, 5)):  # Look at last 5 updates
            interval = sorted_dates[i] - sorted_dates[i + 1]
            intervals.append(interval.total_seconds() / 86400)  # Convert to days
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Categorize update frequency
        if avg_interval < 0.5:  # Less than 12 hours
            return "real-time"
        elif avg_interval < 1.5:  # Daily
            return "daily"
        elif avg_interval < 8:  # Weekly
            return "weekly"
        elif avg_interval < 35:  # Monthly
            return "monthly"
        else:
            return "rarely"


class CrossSourceAgreementCalculator:
    """Calculates agreement between multiple sources"""
    
    @staticmethod
    def calculate_agreement_score(sources: List[Dict[str, Any]]) -> float:
        """Calculate how well sources agree on facts and perspectives"""
        if len(sources) < 2:
            return 0.5  # Neutral score for single source
        
        # Extract key metrics from sources
        bias_scores = []
        factual_ratings = []
        categories = []
        
        for source in sources:
            if "bias_score" in source and source["bias_score"] is not None:
                bias_scores.append(source["bias_score"])
            if "factual" in source:
                # Convert factual ratings to numeric
                fact_map = {"high": 1.0, "medium": 0.5, "low": 0.0, "mixed": 0.5}
                factual_ratings.append(fact_map.get(source["factual"], 0.5))
            if "category" in source:
                categories.append(source["category"])
        
        agreement_score = 0.0
        components = 0
        
        # Calculate bias agreement (lower variance = higher agreement)
        if len(bias_scores) >= 2:
            bias_variance = sum((x - sum(bias_scores)/len(bias_scores))**2 for x in bias_scores) / len(bias_scores)
            bias_agreement = 1.0 - min(bias_variance * 2, 1.0)  # Scale variance to 0-1
            agreement_score += bias_agreement
            components += 1
        
        # Calculate factual rating agreement
        if len(factual_ratings) >= 2:
            fact_variance = sum((x - sum(factual_ratings)/len(factual_ratings))**2 for x in factual_ratings) / len(factual_ratings)
            fact_agreement = 1.0 - min(fact_variance * 2, 1.0)
            agreement_score += fact_agreement
            components += 1
        
        # Bonus for same category sources agreeing
        if len(categories) >= 2:
            category_counts = defaultdict(int)
            for cat in categories:
                category_counts[cat] += 1
            
            # If majority are from same category, add bonus
            max_category_count = max(category_counts.values())
            if max_category_count / len(categories) > 0.6:
                agreement_score += 0.2
                components += 0.2
        
        return agreement_score / components if components > 0 else 0.5


class SourceReputationDatabase:
    """Manages source reputation scores and paradigm alignment"""

    def __init__(self):
        self.domain_checker = DomainAuthorityChecker()
        self.bias_detector = BiasDetector()
        self.controversy_detector = ControversyDetector()
        self.recency_modeler = RecencyModeler()
        self.agreement_calculator = CrossSourceAgreementCalculator()

        # Paradigm-specific source preferences (centralized registry)
        from models.paradigms_sources import PREFERRED_SOURCES  # local import to avoid early import cycles

        self.paradigm_preferences = {
            "dolores": {
                "preferred_sources": PREFERRED_SOURCES["dolores"],
                "bias_preference": "left",
                "factual_weight": 0.7,
                "authority_weight": 0.5,
            },
            "teddy": {
                "preferred_sources": PREFERRED_SOURCES["teddy"],
                "bias_preference": "center",
                "factual_weight": 0.8,
                "authority_weight": 0.6,
            },
            "bernard": {
                "preferred_sources": PREFERRED_SOURCES["bernard"],
                "bias_preference": "center",
                "factual_weight": 0.95,
                "authority_weight": 0.9,
            },
            "maeve": {
                "preferred_sources": PREFERRED_SOURCES["maeve"],
                "bias_preference": "center",
                "factual_weight": 0.8,
                "authority_weight": 0.8,
            },
        }

    async def calculate_credibility_score(
        self, 
        domain: str, 
        paradigm: str = "bernard",
        content: Optional[str] = None,
        search_terms: Optional[List[str]] = None,
        published_date: Optional[datetime] = None,
        other_sources: Optional[List[Dict[str, Any]]] = None
    ) -> CredibilityScore:
        """Calculate comprehensive credibility score for a domain"""
        logger.info("Calculating credibility", stage="credibility_start", domain=domain, paradigm=paradigm)

        # Check cache first (only for basic domain info)
        cache_key = f"cred:card:{domain}:{paradigm}"
        # Unified single-path cache lookup with KV then fallback
        kv_get = getattr(cache_manager, "get_kv", None)
        cached_data = None
        try:
            if callable(kv_get):
                cached_data = await kv_get(cache_key)
            if not cached_data:
                cached_data = await cache_manager.get_source_credibility(cache_key)
        except Exception:
            cached_data = None
        if cached_data and not content and not search_terms:  # Use cache only for basic lookups
            # Convert back to CredibilityScore object
            from utils.date_utils import ensure_datetime
            cached_data["last_updated"] = ensure_datetime(cached_data["last_updated"])
            return CredibilityScore(**cached_data)

        async with self.domain_checker:
            # Get domain authority
            domain_authority = await self.domain_checker.get_domain_authority(domain)

            # Get bias information with category
            bias_data = self.bias_detector.bias_database.get(domain, {})
            bias_rating = bias_data.get("bias", "center")
            bias_score = bias_data.get("bias_score", 0.5)
            fact_check_rating = bias_data.get("factual", "medium")
            source_category = bias_data.get("category", self._infer_category(domain))
            update_frequency = bias_data.get("update_freq", "unknown")

            # Calculate controversy score
            controversy_score, controversy_indicators = self.controversy_detector.calculate_controversy_score(
                domain, content, search_terms
            )
            
            # Add source conflict detection if multiple sources
            if other_sources and len(other_sources) > 1:
                source_domains = [s.get("domain", "") for s in other_sources if s.get("domain")]
                conflict_score = self.controversy_detector.detect_conflicting_sources(source_domains)
                controversy_score = min(controversy_score + conflict_score * 0.3, 1.0)
                if conflict_score > 0.5:
                    controversy_indicators.append("Conflicting source perspectives detected")
            
            # Calculate recency score
            recency_score = self.recency_modeler.calculate_recency_score(
                published_date, source_category, is_breaking=False
            )
            
            # Calculate cross-source agreement if other sources provided
            cross_source_agreement = None
            if other_sources and len(other_sources) > 1:
                # Add current source to the list for comparison
                all_sources = other_sources + [{
                    "domain": domain,
                    "bias_score": bias_score,
                    "factual": fact_check_rating,
                    "category": source_category
                }]
                cross_source_agreement = self.agreement_calculator.calculate_agreement_score(all_sources)

            # Optional: augment with Brave AI Grounding citations
            brave_enabled = os.getenv("ENABLE_BRAVE_GROUNDING", "0").lower() in ("1", "true", "yes")
            brave_deep = os.getenv("BRAVE_ENABLE_RESEARCH", "0").lower() in ("1", "true", "yes")
            brave_agreement = None
            brave_citation_count = 0
            if brave_enabled and brave_client().is_configured():
                # Build a verification query
                if content and len(content) > 10:
                    verify_query = content[:1500]
                elif search_terms:
                    verify_query = " ".join(search_terms)[:1500]
                else:
                    verify_query = f"What is the credibility and reputation of {domain}? Provide recent sources."

                try:
                    citations = await brave_client().fetch_citations(
                        verify_query, enable_research=brave_deep
                    )
                    brave_citation_count = len(citations)
                    if brave_citation_count:
                        # Evaluate cited domains for agreement and authority
                        cited_domains = []
                        seen = set()
                        for c in citations:
                            host = (c.hostname or "").lower()
                            # normalize bare domains
                            if host.startswith("www."):
                                host = host[4:]
                            if host and host not in seen:
                                seen.add(host)
                                cited_domains.append(host)

                        # Limit work to first 10 unique domains
                        cited_domains = cited_domains[:10]

                        # Gather bias/factual/DA concurrently
                        bias_tasks = [self.bias_detector.get_bias_score(h) for h in cited_domains]
                        da_tasks = [self.domain_checker.get_domain_authority(h) for h in cited_domains]
                        bias_results = await asyncio.gather(*bias_tasks, return_exceptions=True)
                        da_results = await asyncio.gather(*da_tasks, return_exceptions=True)

                        sources_for_agreement: List[Dict[str, Any]] = []
                        da_values: List[float] = []
                        for i, host in enumerate(cited_domains):
                            bias_tuple = bias_results[i]
                            if isinstance(bias_tuple, Exception):
                                bias_rating2, bias_score2, factual2 = "center", 0.5, "medium"
                            else:
                                bias_rating2, bias_score2, factual2 = bias_tuple
                            da_val = da_results[i]
                            if isinstance(da_val, Exception) or da_val is None:
                                da = 40.0
                            else:
                                da = float(da_val)
                            da_values.append(da)
                            sources_for_agreement.append({
                                "domain": host,
                                "bias_score": bias_score2,
                                "factual": factual2,
                                "category": self._infer_category(host),
                            })

                        # Use existing agreement calculator + DA quality
                        agree = self.agreement_calculator.calculate_agreement_score(sources_for_agreement)
                        avg_da = (sum(da_values) / len(da_values) / 100.0) if da_values else 0.4
                        brave_agreement = max(0.0, min(1.0, 0.7 * agree + 0.3 * avg_da))

                        # Blend with any precomputed agreement
                        if cross_source_agreement is None:
                            cross_source_agreement = brave_agreement
                        else:
                            cross_source_agreement = (cross_source_agreement + brave_agreement) / 2.0
                except Exception as e:
                    logger.debug(f"Brave grounding verification skipped due to error: {e}")

            # Calculate paradigm alignment
            paradigm_alignment = self._calculate_paradigm_alignment(
                domain, bias_rating, fact_check_rating, paradigm
            )

            # Calculate overall score with new factors
            overall_score = self._calculate_enhanced_overall_score(
                domain_authority, bias_score, fact_check_rating, paradigm, domain,
                recency_score, controversy_score, cross_source_agreement
            )

            # Identify reputation factors
            reputation_factors = self._identify_reputation_factors(
                domain, domain_authority, bias_rating, fact_check_rating
            )
            
            # Add new reputation factors
            if recency_score > 0.8:
                reputation_factors.append("Recently Updated")
            elif recency_score < 0.3:
                reputation_factors.append("Potentially Outdated")
            
            if controversy_score > 0.7:
                reputation_factors.append("Highly Controversial")
            
            if cross_source_agreement and cross_source_agreement > 0.8:
                reputation_factors.append("High Source Agreement")
            elif cross_source_agreement and cross_source_agreement < 0.3:
                reputation_factors.append("Low Source Agreement")
            if brave_citation_count:
                reputation_factors.append(f"Brave citations: {brave_citation_count}")

            # Create credibility score object
            credibility = CredibilityScore(
                domain=domain,
                overall_score=overall_score,
                domain_authority=domain_authority,
                bias_rating=bias_rating,
                bias_score=bias_score,
                fact_check_rating=fact_check_rating,
                paradigm_alignment=paradigm_alignment,
                reputation_factors=reputation_factors,
                recency_score=recency_score,
                cross_source_agreement=cross_source_agreement,
                controversy_score=controversy_score,
                update_frequency=update_frequency,
                source_category=source_category,
                controversy_indicators=controversy_indicators,
            )

            # Cache the result under card namespace
            kv_set = getattr(cache_manager, "set_kv", None)
            payload = credibility.to_dict()
            if callable(kv_set):
                await kv_set(cache_key, payload, ttl=cache_manager.ttl_config.get("source_credibility", 30*24*3600))
            else:
                await cache_manager.set_source_credibility(cache_key, payload)

            logger.info("Finished credibility calculation", stage="credibility_end", domain=domain, paradigm=paradigm, overall_score=overall_score, domain_authority=domain_authority, bias_score=bias_score, fact_check_rating=fact_check_rating, recency_score=recency_score, controversy_score=controversy_score, cross_source_agreement=cross_source_agreement)

            return credibility
    
    def _infer_category(self, domain: str) -> str:
        """Infer source category from domain via shared categorizer."""
        try:
            from utils.domain_categorizer import categorize as _categorize
            return _categorize(domain)
        except Exception:
            return "general"

    def _calculate_paradigm_alignment(
        self,
        domain: str,
        bias_rating: Optional[str],
        fact_check_rating: Optional[str],
        paradigm: str,
    ) -> Dict[str, float]:
        """Calculate how well a source aligns with each paradigm"""
        alignment = {}

        for para_name, preferences in self.paradigm_preferences.items():
            score = 0.5  # Base score

            # Bonus for preferred sources
            if domain in preferences["preferred_sources"]:
                score += 0.3

            # Bias alignment
            if bias_rating:
                if preferences["bias_preference"] == bias_rating:
                    score += 0.2
                elif (
                    preferences["bias_preference"] == "center"
                    and bias_rating == "center"
                ):
                    score += 0.2
                elif (
                    preferences["bias_preference"] != "center"
                    and bias_rating != "center"
                ):
                    score += 0.1  # Some bias tolerance

            # Factual accuracy bonus
            if fact_check_rating == "high":
                score += 0.2 * preferences["factual_weight"]
            elif fact_check_rating == "medium":
                score += 0.1 * preferences["factual_weight"]

            alignment[para_name] = min(1.0, score)

        return alignment

    def _calculate_overall_score(
        self,
        domain_authority: Optional[float],
        bias_score: Optional[float],
        fact_check_rating: Optional[str],
        paradigm: str,
        domain: str,
    ) -> float:
        """Calculate overall credibility score (0.0 to 1.0) - legacy method"""

        weights = self.paradigm_preferences.get(
            paradigm, self.paradigm_preferences["bernard"]
        )

        score = 0.0

        # Domain authority component (0-1 scale)
        if domain_authority is not None:
            da_score = domain_authority / 100.0
            score += da_score * weights["authority_weight"] * 0.4

        # Bias score component
        if bias_score is not None:
            score += bias_score * 0.3

        # Factual accuracy component
        fact_scores = {"high": 1.0, "medium": 0.7, "low": 0.3}
        if fact_check_rating in fact_scores:
            score += fact_scores[fact_check_rating] * weights["factual_weight"] * 0.3

        return min(1.0, score)
    
    def _calculate_enhanced_overall_score(
        self,
        domain_authority: Optional[float],
        bias_score: Optional[float],
        fact_check_rating: Optional[str],
        paradigm: str,
        domain: str,
        recency_score: float,
        controversy_score: float,
        cross_source_agreement: Optional[float]
    ) -> float:
        """Calculate enhanced overall credibility score with new factors"""
        
        weights = self.paradigm_preferences.get(
            paradigm, self.paradigm_preferences["bernard"]
        )
        
        # Start with base score components (60% of total)
        base_score = 0.0
        
        # Domain authority (20% of total)
        if domain_authority is not None:
            da_score = domain_authority / 100.0
            base_score += da_score * weights["authority_weight"] * 0.2
        
        # Bias score (15% of total)
        if bias_score is not None:
            base_score += bias_score * 0.15
        
        # Factual accuracy (25% of total)
        fact_scores = {"high": 1.0, "medium": 0.7, "low": 0.3, "mixed": 0.5}
        if fact_check_rating in fact_scores:
            base_score += fact_scores[fact_check_rating] * weights["factual_weight"] * 0.25
        
        # New factors (40% of total)
        
        # Recency (15% of total)
        base_score += recency_score * 0.15
        
        # Controversy penalty (15% of total) - inverted
        controversy_penalty = (1.0 - controversy_score) * 0.15
        base_score += controversy_penalty
        
        # Cross-source agreement (10% of total)
        if cross_source_agreement is not None:
            base_score += cross_source_agreement * 0.10
        else:
            # If no cross-source data, redistribute weight
            base_score += 0.05  # Neutral contribution
        
        # Apply paradigm-specific adjustments
        if paradigm == "dolores" and controversy_score > 0.5:
            # Revolutionary paradigm values controversial topics
            base_score += 0.05
        elif paradigm == "bernard" and controversy_score > 0.7:
            # Analytical paradigm penalizes high controversy
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))

    def _identify_reputation_factors(
        self,
        domain: str,
        domain_authority: Optional[float],
        bias_rating: Optional[str],
        fact_check_rating: Optional[str],
    ) -> List[str]:
        """Identify factors contributing to reputation"""
        factors = []

        if domain_authority and domain_authority > 80:
            factors.append("High Domain Authority")
        elif domain_authority and domain_authority > 60:
            factors.append("Medium Domain Authority")

        if bias_rating == "center":
            factors.append("Neutral Bias")
        elif bias_rating in ["left", "right"]:
            factors.append(f"Political Bias: {bias_rating.title()}")

        if fact_check_rating == "high":
            factors.append("High Factual Accuracy")
        elif fact_check_rating == "low":
            factors.append("Low Factual Accuracy")

        # Use domain categorizer for consistent categorization
        from utils.domain_categorizer import categorize
        category = categorize(domain)

        category_factors = {
            "government": "Government Source",
            "academic": "Academic Institution",
            "news": "News Organization",
            "reference": "Reference Source",
            "blog": "Non-profit Organization",
            "social": "Social Media Platform",
            "tech": "Technology Source",
            "video": "Video Platform",
            "pdf": "Document Source"
        }

        if category in category_factors:
            factors.append(category_factors[category])

        return factors


# Global instance
credibility_system = SourceReputationDatabase()


async def get_source_credibility(
    domain: str,
    paradigm: str = "bernard",
    content: Optional[str] = None,
    search_terms: Optional[List[str]] = None,
    published_date: Optional[datetime] = None,
    other_sources: Optional[List[Dict[str, Any]]] = None
) -> CredibilityScore:
    """Convenience function to get source credibility with enhanced features"""
    return await credibility_system.calculate_credibility_score(
        domain=domain,
        paradigm=paradigm,
        content=content,
        search_terms=search_terms,
        published_date=published_date,
        other_sources=other_sources
    )


async def get_source_credibility_safe(
    domain: str,
    paradigm: str,
    credibility_enabled: bool = True,
    log_failures: bool = True
) -> Tuple[float, str, str]:
    """Get source credibility with non-blocking error handling

    Returns: (overall_score, explanation, status)
    where status is one of: 'success', 'failed', 'skipped'
    """
    try:
        if not credibility_enabled:
            return 0.5, "Credibility checking disabled", "skipped"

        credibility = await get_source_credibility(domain, paradigm)
        # Prefer structured to_dict if available; fall back to attribute
        explanation = ""
        if hasattr(credibility, "to_dict"):
            card = credibility.to_dict()
            explanation = (
                f"bias={card.get('bias_rating')}, "
                f"fact={card.get('fact_check_rating')}, "
                f"cat={card.get('source_category')}"
            )
        elif hasattr(credibility, "explanation"):
            explanation = getattr(credibility, "explanation") or ""
        return getattr(credibility, "overall_score", 0.5), explanation, "success"
    except Exception as e:
        if log_failures:
            logger.warning("Credibility check failed for %s: %s", domain, str(e))
        return 0.5, f"Credibility check failed: {str(e)}", "failed"


# Example usage and testing
async def test_credibility_system():
    """Test the enhanced credibility system"""
    print("Testing Enhanced Source Credibility System...")
    print("=" * 50)

    # Test basic credibility scoring
    test_domains = [
        "nytimes.com",
        "foxnews.com",
        "reuters.com",
        "nature.com",
        "breitbart.com",
        "wikipedia.org",
        "twitter.com",
        "arxiv.org",
        "bbc.com",
    ]

    print("\n1. BASIC CREDIBILITY ASSESSMENT")
    print("-" * 40)
    
    for domain in test_domains:
        print(f"\n{domain}:")
        credibility = await get_source_credibility(domain, "bernard")
        card = credibility.generate_credibility_card()
        
        print(f"  Trust Level: {card['trust_level']} ({card['overall_score']})")
        print(f"  Category: {credibility.source_category}")
        print(f"  Key Factors: {', '.join(card['key_factors'].values())}")
        if card['concerns']:
            print(f"  Concerns: {', '.join(card['concerns'])}")
    
    print("\n\n2. CONTROVERSY DETECTION TEST")
    print("-" * 40)
    
    controversial_test = [
        ("cnn.com", ["trump", "election", "fraud"], "Political controversy test"),
        ("nature.com", ["climate", "change", "debate"], "Climate science test"),
        ("foxnews.com", ["vaccine", "mandate", "freedom"], "Health policy test"),
    ]
    
    for domain, terms, description in controversial_test:
        print(f"\n{description} - {domain}:")
        credibility = await get_source_credibility(
            domain=domain,
            search_terms=terms,
            paradigm="bernard"
        )
        print(f"  Controversy Score: {credibility.controversy_score:.2f}")
        if credibility.controversy_indicators:
            print(f"  Indicators: {', '.join(credibility.controversy_indicators)}")
    
    print("\n\n3. RECENCY AND TEMPORAL RELEVANCE TEST")
    print("-" * 40)
    
    # Test with different publication dates
    test_dates = [
        (get_current_utc() - timedelta(hours=12), "12 hours ago"),
        (get_current_utc() - timedelta(days=7), "1 week ago"),
        (get_current_utc() - timedelta(days=30), "1 month ago"),
        (get_current_utc() - timedelta(days=365), "1 year ago"),
    ]
    
    for date, description in test_dates:
        print(f"\nNews article from {description}:")
        credibility = await get_source_credibility(
            domain="cnn.com",
            published_date=date,
            paradigm="bernard"
        )
        print(f"  Recency Score: {credibility.recency_score:.2f}")
        print(f"  Overall Impact on Credibility: {credibility.overall_score:.2f}")
    
    print("\n\n4. CROSS-SOURCE AGREEMENT TEST")
    print("-" * 40)
    
    # Test agreement between sources
    print("\nTesting agreement between similar sources:")
    similar_sources = [
        {"domain": "reuters.com", "bias_score": 0.9, "factual": "high", "category": "news"},
        {"domain": "apnews.com", "bias_score": 0.9, "factual": "high", "category": "news"},
        {"domain": "bbc.com", "bias_score": 0.8, "factual": "high", "category": "news"},
    ]
    
    credibility = await get_source_credibility(
        domain="npr.org",
        other_sources=similar_sources,
        paradigm="bernard"
    )
    print(f"  Cross-Source Agreement: {credibility.cross_source_agreement:.2f}")
    print(f"  Overall Credibility: {credibility.overall_score:.2f}")
    
    print("\nTesting agreement between conflicting sources:")
    conflicting_sources = [
        {"domain": "foxnews.com", "bias_score": 0.2, "factual": "mixed", "category": "news"},
        {"domain": "cnn.com", "bias_score": 0.3, "factual": "mixed", "category": "news"},
        {"domain": "breitbart.com", "bias_score": 0.1, "factual": "low", "category": "news"},
    ]
    
    credibility = await get_source_credibility(
        domain="huffpost.com",
        other_sources=conflicting_sources,
        paradigm="bernard"
    )
    print(f"  Cross-Source Agreement: {credibility.cross_source_agreement:.2f}")
    print(f"  Controversy Score: {credibility.controversy_score:.2f}")
    print(f"  Overall Credibility: {credibility.overall_score:.2f}")
    
    print("\n\n5. PARADIGM-SPECIFIC CREDIBILITY")
    print("-" * 40)
    
    test_domain = "propublica.org"
    print(f"\nTesting {test_domain} across paradigms:")
    
    for paradigm in ["dolores", "teddy", "bernard", "maeve"]:
        credibility = await get_source_credibility(
            domain=test_domain,
            paradigm=paradigm,
            search_terms=["investigation", "corruption"]
        )
        alignment = credibility.paradigm_alignment.get(paradigm, 0.0)
        print(f"  {paradigm.upper()}: Score={credibility.overall_score:.2f}, Alignment={alignment:.2f}")
    
    print("\n\n6. CREDIBILITY CARD GENERATION")
    print("-" * 40)
    
    # Generate a detailed credibility card
    credibility = await get_source_credibility(
        domain="nytimes.com",
        search_terms=["politics", "election"],
        published_date=get_current_utc() - timedelta(days=2),
        paradigm="bernard"
    )
    
    card = credibility.generate_credibility_card()
    print(f"\nCredibility Card for {card['domain']}:")
    print(f"  Trust Level: {card['trust_level']} (Score: {card['overall_score']})")
    print("\n  Key Factors:")
    for factor, value in card['key_factors'].items():
        print(f"    - {factor}: {value}")
    
    if card['strengths']:
        print("\n  Strengths:")
        for strength in card['strengths']:
            print(f"    + {strength}")
    
    if card['concerns']:
        print("\n  Concerns:")
        for concern in card['concerns']:
            print(f"    - {concern}")
    
    if card['recommendations']:
        print("\n  Recommendations:")
        for rec in card['recommendations']:
            print(f"    → {rec}")
    
    print("\n  Paradigm Alignment Scores:")
    for paradigm, score in card['paradigm_scores'].items():
        print(f"    {paradigm}: {score}")


# Additional utility functions for credibility analysis
async def analyze_source_credibility_batch(
    sources: List[Dict[str, Any]], 
    paradigm: str = "bernard"
) -> Dict[str, Any]:
    """Analyze credibility for multiple sources and provide aggregate insights"""
    credibility_scores = []
    
    for source in sources:
        credibility = await get_source_credibility(
            domain=source.get("domain", ""),
            paradigm=paradigm,
            content=source.get("content"),
            search_terms=source.get("search_terms"),
            published_date=source.get("published_date"),
            other_sources=[s for s in sources if s != source]
        )
        credibility_scores.append(credibility)
    
    # Calculate aggregate metrics
    avg_credibility = sum(c.overall_score for c in credibility_scores) / len(credibility_scores)
    high_credibility_count = sum(1 for c in credibility_scores if c.overall_score >= 0.7)
    controversial_count = sum(1 for c in credibility_scores if c.controversy_score > 0.5)
    
    # Identify best and worst sources
    sorted_scores = sorted(credibility_scores, key=lambda x: x.overall_score, reverse=True)
    
    return {
        "total_sources": len(sources),
        "average_credibility": avg_credibility,
        "high_credibility_sources": high_credibility_count,
        "controversial_sources": controversial_count,
        "most_credible": sorted_scores[0].domain if sorted_scores else None,
        "least_credible": sorted_scores[-1].domain if sorted_scores else None,
        "credibility_distribution": {
            "very_high": sum(1 for c in credibility_scores if c.overall_score >= 0.8),
            "high": sum(1 for c in credibility_scores if 0.6 <= c.overall_score < 0.8),
            "moderate": sum(1 for c in credibility_scores if 0.4 <= c.overall_score < 0.6),
            "low": sum(1 for c in credibility_scores if 0.2 <= c.overall_score < 0.4),
            "very_low": sum(1 for c in credibility_scores if c.overall_score < 0.2),
        },
        "paradigm_recommendations": _get_paradigm_recommendations(credibility_scores, paradigm)
    }

def _get_paradigm_recommendations(scores: List[CredibilityScore], paradigm: str) -> List[str]:
    """Generate paradigm-specific recommendations based on credibility scores"""
    recommendations = []
    
    if paradigm == "bernard":  # Analytical
        academic_sources = [s for s in scores if s.source_category == "academic"]
        if len(academic_sources) < len(scores) * 0.3:
            recommendations.append("Consider adding more academic sources for analytical rigor")
    
    elif paradigm == "dolores":  # Revolutionary
        controversial_sources = [s for s in scores if s.controversy_score > 0.5]
        if not controversial_sources:
            recommendations.append("Include alternative viewpoints to challenge the status quo")
    
    elif paradigm == "teddy":  # Devotion
        biased_sources = [s for s in scores if s.bias_score and s.bias_score < 0.5]
        if len(biased_sources) > len(scores) * 0.5:
            recommendations.append("Balance with more neutral, community-focused sources")
    
    elif paradigm == "maeve":  # Strategic
        outdated_sources = [s for s in scores if s.recency_score < 0.5]
        if outdated_sources:
            recommendations.append("Update with more recent business intelligence for strategic insights")
    
    return recommendations


if __name__ == "__main__":
    asyncio.run(test_credibility_system())
