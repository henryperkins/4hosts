"""
Source Credibility Scoring System for Four Hosts Research Application
Implements domain authority checking, bias detection, and source reputation scoring
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import os

from .cache import cache_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    last_updated: datetime = field(default_factory=datetime.now)
    
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
            "last_updated": self.last_updated.isoformat()
        }

class DomainAuthorityChecker:
    """Checks domain authority using various APIs"""
    
    def __init__(self):
        self.moz_api_key = os.getenv("MOZ_API_KEY")
        self.moz_secret_key = os.getenv("MOZ_SECRET_KEY")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_domain_authority(self, domain: str) -> Optional[float]:
        """Get domain authority score (0-100)"""
        
        # Try cached result first
        cached_result = await cache_manager.get_source_credibility(domain)
        if cached_result and cached_result.get("domain_authority"):
            return cached_result["domain_authority"]
        
        # Try Moz API if available
        if self.moz_api_key and self.moz_secret_key:
            moz_da = await self._get_moz_domain_authority(domain)
            if moz_da is not None:
                return moz_da
        
        # Fallback to heuristic scoring
        return await self._heuristic_domain_authority(domain)
    
    async def _get_moz_domain_authority(self, domain: str) -> Optional[float]:
        """Get domain authority from Moz API"""
        try:
            import hmac
            import hashlib
            import base64
            from urllib.parse import quote
            
            # Moz API authentication
            expires = int((datetime.now() + timedelta(minutes=5)).timestamp())
            string_to_sign = f"{self.moz_api_key}\n{expires}"
            signature = base64.b64encode(
                hmac.new(
                    self.moz_secret_key.encode('utf-8'),
                    string_to_sign.encode('utf-8'),
                    hashlib.sha1
                ).digest()
            ).decode('utf-8')
            
            url = f"https://lsapi.seomoz.com/v2/url_metrics"
            
            # Correct payload for POST request
            payload = {
                "targets": [domain]
            }

            # Correct authentication
            auth = (self.moz_api_key, self.moz_secret_key)
            
            async with self.session.post(url, auth=aiohttp.BasicAuth(*auth), json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract Domain Authority from the new response structure
                    if "results" in data and data["results"]:
                        return data["results"][0].get("page_authority") # Moz v2 uses page_authority
                    return None
                else:
                    logger.warning(f"Moz API error for {domain}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Moz API error: {str(e)}")
        
        return None
    
    async def _heuristic_domain_authority(self, domain: str) -> float:
        """Heuristic domain authority based on known high-authority domains"""
        
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
        
        # Check TLD patterns
        if domain.endswith(".gov"):
            return 85
        elif domain.endswith(".edu"):
            return 80
        elif domain.endswith(".org"):
            return 60  # Non-profits generally trusted
        elif domain.endswith(".com"):
            return 45  # Commercial sites vary widely
        else:
            return 40  # Other TLDs
    
class BiasDetector:
    """Detects political bias and factual accuracy of sources"""
    
    def __init__(self):
        # Load bias ratings from AllSides, Media Bias/Fact Check, etc.
        self.bias_database = self._load_bias_database()
    
    def _load_bias_database(self) -> Dict[str, Dict[str, Any]]:
        """Load bias ratings database"""
        # This would typically be loaded from a file or API
        # For now, using a subset of well-known sources
        return {
            # Left-leaning sources
            "cnn.com": {"bias": "left", "bias_score": 0.3, "factual": "mixed"},
            "msnbc.com": {"bias": "left", "bias_score": 0.2, "factual": "mixed"},
            "huffpost.com": {"bias": "left", "bias_score": 0.1, "factual": "mixed"},
            "theguardian.com": {"bias": "left", "bias_score": 0.3, "factual": "high"},
            "washingtonpost.com": {"bias": "left", "bias_score": 0.4, "factual": "high"},
            "nytimes.com": {"bias": "left", "bias_score": 0.4, "factual": "high"},
            
            # Right-leaning sources  
            "foxnews.com": {"bias": "right", "bias_score": 0.2, "factual": "mixed"},
            "breitbart.com": {"bias": "right", "bias_score": 0.1, "factual": "low"},
            "dailywire.com": {"bias": "right", "bias_score": 0.2, "factual": "mixed"},
            "wsj.com": {"bias": "right", "bias_score": 0.4, "factual": "high"},
            
            # Center/Neutral sources
            "reuters.com": {"bias": "center", "bias_score": 0.9, "factual": "high"},
            "apnews.com": {"bias": "center", "bias_score": 0.9, "factual": "high"},
            "bbc.com": {"bias": "center", "bias_score": 0.8, "factual": "high"},
            "npr.org": {"bias": "center", "bias_score": 0.7, "factual": "high"},
            
            # Academic/Scientific
            "nature.com": {"bias": "center", "bias_score": 0.95, "factual": "high"},
            "science.org": {"bias": "center", "bias_score": 0.95, "factual": "high"},
            "arxiv.org": {"bias": "center", "bias_score": 0.9, "factual": "high"},
            "pubmed.ncbi.nlm.nih.gov": {"bias": "center", "bias_score": 0.95, "factual": "high"},
            
            # Wikipedia and reference
            "wikipedia.org": {"bias": "center", "bias_score": 0.8, "factual": "high"},
        }
    
    async def get_bias_score(self, domain: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Get bias rating, bias score, and factual rating for domain"""
        
        # Check cache first
        cached_result = await cache_manager.get_source_credibility(domain)
        if cached_result:
            return (
                cached_result.get("bias_rating"),
                cached_result.get("bias_score"),
                cached_result.get("fact_check_rating")
            )
        
        # Check our database
        if domain in self.bias_database:
            data = self.bias_database[domain]
            return data["bias"], data["bias_score"], data["factual"]
        
        # Heuristic scoring for unknown domains
        return await self._heuristic_bias_score(domain)
    
    async def _heuristic_bias_score(self, domain: str) -> Tuple[str, float, str]:
        """Heuristic bias scoring for unknown domains"""
        
        # Government and academic sources are generally neutral and factual
        if domain.endswith(".gov"):
            return "center", 0.8, "high"
        elif domain.endswith(".edu"):
            return "center", 0.8, "high"
        elif domain.endswith(".org"):
            return "center", 0.7, "medium"  # NGOs can have bias but often factual
        
        # Commercial and personal sites default to unknown
        return "center", 0.5, "medium"

class SourceReputationDatabase:
    """Manages source reputation scores and paradigm alignment"""
    
    def __init__(self):
        self.domain_checker = DomainAuthorityChecker()
        self.bias_detector = BiasDetector()
        
        # Paradigm-specific source preferences
        self.paradigm_preferences = {
            "dolores": {
                # Revolutionary paradigm prefers investigative, alternative sources
                "preferred_sources": [
                    "propublica.org", "theintercept.com", "democracynow.org",
                    "jacobinmag.com", "commondreams.org", "truthout.org"
                ],
                "bias_preference": "left",  # Slight preference for left-leaning sources
                "factual_weight": 0.7,  # Facts important but narrative matters more
                "authority_weight": 0.5   # Less concerned with traditional authority
            },
            "teddy": {
                # Devotion paradigm prefers community, care-focused sources
                "preferred_sources": [
                    "npr.org", "pbs.org", "unitedway.org", "redcross.org",
                    "who.int", "unicef.org", "doctorswithoutborders.org"
                ],
                "bias_preference": "center",  # Neutral perspective preferred
                "factual_weight": 0.8,  # High importance on factual accuracy
                "authority_weight": 0.6   # Moderate respect for authority
            },
            "bernard": {
                # Analytical paradigm strongly prefers academic, research sources
                "preferred_sources": [
                    "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
                    "scholar.google.com", "jstor.org", "researchgate.net"
                ],
                "bias_preference": "center",  # Strict neutrality required
                "factual_weight": 0.95,  # Extremely high importance on facts
                "authority_weight": 0.9    # High respect for academic authority
            },
            "maeve": {
                # Strategic paradigm prefers business, industry sources
                "preferred_sources": [
                    "wsj.com", "ft.com", "bloomberg.com", "forbes.com",
                    "hbr.org", "mckinsey.com", "bcg.com", "strategy-business.com"
                ],
                "bias_preference": "center",  # Balanced perspective for strategy
                "factual_weight": 0.8,   # Facts important for good strategy
                "authority_weight": 0.8   # Values established business authority
            }
        }
    
    async def calculate_credibility_score(self, domain: str, paradigm: str = "bernard") -> CredibilityScore:
        """Calculate comprehensive credibility score for a domain"""
        
        # Check cache first
        cached_data = await cache_manager.get_source_credibility(domain)
        if cached_data:
            # Convert back to CredibilityScore object
            cached_data["last_updated"] = datetime.fromisoformat(cached_data["last_updated"])
            return CredibilityScore(**cached_data)
        
        async with self.domain_checker:
            # Get domain authority
            domain_authority = await self.domain_checker.get_domain_authority(domain)
            
            # Get bias information
            bias_rating, bias_score, fact_check_rating = await self.bias_detector.get_bias_score(domain)
            
            # Calculate paradigm alignment
            paradigm_alignment = self._calculate_paradigm_alignment(
                domain, bias_rating, fact_check_rating, paradigm
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                domain_authority, bias_score, fact_check_rating, paradigm, domain
            )
            
            # Identify reputation factors
            reputation_factors = self._identify_reputation_factors(
                domain, domain_authority, bias_rating, fact_check_rating
            )
            
            # Create credibility score object
            credibility = CredibilityScore(
                domain=domain,
                overall_score=overall_score,
                domain_authority=domain_authority,
                bias_rating=bias_rating,
                bias_score=bias_score,
                fact_check_rating=fact_check_rating,
                paradigm_alignment=paradigm_alignment,
                reputation_factors=reputation_factors
            )
            
            # Cache the result
            await cache_manager.set_source_credibility(domain, credibility.to_dict())
            
            return credibility
    
    def _calculate_paradigm_alignment(self, domain: str, bias_rating: Optional[str], 
                                    fact_check_rating: Optional[str], paradigm: str) -> Dict[str, float]:
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
                elif preferences["bias_preference"] == "center" and bias_rating == "center":
                    score += 0.2
                elif preferences["bias_preference"] != "center" and bias_rating != "center":
                    score += 0.1  # Some bias tolerance
            
            # Factual accuracy bonus
            if fact_check_rating == "high":
                score += 0.2 * preferences["factual_weight"]
            elif fact_check_rating == "medium":
                score += 0.1 * preferences["factual_weight"]
            
            alignment[para_name] = min(1.0, score)
        
        return alignment
    
    def _calculate_overall_score(self, domain_authority: Optional[float], bias_score: Optional[float],
                               fact_check_rating: Optional[str], paradigm: str, domain: str) -> float:
        """Calculate overall credibility score (0.0 to 1.0)"""
        
        weights = self.paradigm_preferences.get(paradigm, self.paradigm_preferences["bernard"])
        
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
    
    def _identify_reputation_factors(self, domain: str, domain_authority: Optional[float],
                                   bias_rating: Optional[str], fact_check_rating: Optional[str]) -> List[str]:
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
        
        if domain.endswith(".gov"):
            factors.append("Government Source")
        elif domain.endswith(".edu"):
            factors.append("Academic Institution")
        elif domain.endswith(".org"):
            factors.append("Non-profit Organization")
        
        return factors

# Global instance
credibility_system = SourceReputationDatabase()

async def get_source_credibility(domain: str, paradigm: str = "bernard") -> CredibilityScore:
    """Convenience function to get source credibility"""
    return await credibility_system.calculate_credibility_score(domain, paradigm)

# Example usage and testing
async def test_credibility_system():
    """Test the credibility system"""
    print("Testing Source Credibility System...")
    print("=" * 50)
    
    test_domains = [
        "nytimes.com",
        "foxnews.com", 
        "reuters.com",
        "nature.com",
        "breitbart.com",
        "wikipedia.org",
        "unknowndomain.com"
    ]
    
    for domain in test_domains:
        print(f"\nTesting: {domain}")
        
        for paradigm in ["dolores", "teddy", "bernard", "maeve"]:
            credibility = await get_source_credibility(domain, paradigm)
            alignment = credibility.paradigm_alignment.get(paradigm, 0.0)
            
            print(f"  {paradigm.upper()}: {credibility.overall_score:.2f} "
                  f"(alignment: {alignment:.2f})")
        
        print(f"  Domain Authority: {credibility.domain_authority}")
        print(f"  Bias: {credibility.bias_rating} ({credibility.bias_score})")
        print(f"  Factual: {credibility.fact_check_rating}")
        print(f"  Factors: {', '.join(credibility.reputation_factors)}")

if __name__ == "__main__":
    asyncio.run(test_credibility_system())