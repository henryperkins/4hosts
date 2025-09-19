"""
Centralized registry of preferred source domains for each paradigm.

This module is the single canonical place to maintain per-paradigm
preferred sources. Other modules (e.g., paradigm_search, credibility)
must import from here to avoid duplication.
"""

from __future__ import annotations

from typing import Dict, List

PREFERRED_SOURCES: Dict[str, List[str]] = {
    # Revolutionary paradigm - investigative/alternative sources
    "dolores": [
        "propublica.org",
        "theintercept.com",
        "democracynow.org",
        "jacobinmag.com",
        "commondreams.org",
        "truthout.org",
        "motherjones.com",
        "thenation.com",
        "theguardian.com",
        "washingtonpost.com",
        "nytimes.com",
        "icij.org",
    ],

    # Devotion paradigm - community/care-focused sources
    "teddy": [
        "npr.org",
        "pbs.org",
        "unitedway.org",
        "redcross.org",
        "who.int",
        "unicef.org",
        "doctorswithoutborders.org",
        "goodwill.org",
        "salvationarmy.org",
        "feedingamerica.org",
        "habitat.org",
        "americanredcross.org",
    ],

    # Analytical paradigm - academic/research sources
    "bernard": [
        "nature.com",
        "science.org",
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "jstor.org",
        "researchgate.net",
        "springerlink.com",
        "sciencedirect.com",
        "wiley.com",
        "tandfonline.com",
        "cambridge.org",
    ],

    # Strategic paradigm - business/industry sources
    "maeve": [
        "wsj.com",
        "ft.com",
        "bloomberg.com",
        "forbes.com",
        "hbr.org",
        "mckinsey.com",
        "bcg.com",
        "strategy-business.com",
        "bain.com",
        "deloitte.com",
        "pwc.com",
        "kpmg.com",
        "gartner.com",
        "forrester.com",
    ],
}

__all__ = ["PREFERRED_SOURCES"]