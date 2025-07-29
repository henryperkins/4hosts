```python
# Four Hosts Research System - Complete Integration
# Combines Classification Engine and Context Engineering Pipeline

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Mock LLM Service for Classification ---

class MockLLMService:
    """Mock LLM service for paradigm classification"""
    
    async def classify_with_llm(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM-based classification"""
        await asyncio.sleep(0.2)  # Simulate API delay
        
        # Simulate sophisticated LLM analysis
        query_lower = query.lower()
        
        scores = {
            'DOLORES': 0.1,
            'TEDDY': 0.1,
            'BERNARD': 0.1,
            'MAEVE': 0.1
        }
        
        # Simulate pattern recognition
        if any(word in query_lower for word in ['expose', 'unfair', 'corrupt', 'fight']):
            scores['DOLORES'] += 0.4
        if any(word in query_lower for word in ['help', 'support', 'care', 'protect']):
            scores['TEDDY'] += 0.4
        if any(word in query_lower for word in ['analyze', 'research', 'study', 'data']):
            scores['BERNARD'] += 0.4
        if any(word in query_lower for word in ['strategy', 'compete', 'optimize', 'influence']):
            scores['MAEVE'] += 0.4
            
        # Normalize
        total = sum(scores.values())
        normalized = {k: v/total for k, v in scores.items()}
        
        return {
            'scores': normalized,
            'reasoning': "LLM analysis based on semantic patterns and context",
            'confidence': max(normalized.values())
        }

# --- Complete Research Query Processor ---

@dataclass
class ResearchRequest:
    """Complete research request with all parameters"""
    query: str
    options: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.options is None:
            self.options = {}

@dataclass 
class ResearchResult:
    """Complete research result with all outputs"""
    request: ResearchRequest
    classification: Any  # ClassificationResult
    context_engineering: Any  # ContextEngineeredQuery
    paradigm: str
    confidence: float
    search_queries: List[Dict[str, Any]]
    processing_metrics: Dict[str, float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'query': self.request.query,
            'paradigm': self.paradigm,
            'confidence': self.confidence,
            'search_queries_count': len(self.search_queries),
            'search_queries': self.search_queries[:5],  # First 5 queries
            'processing_metrics': self.processing_metrics,
            'timestamp': self.timestamp.isoformat(),
            'context_summary': {
                'documentation_focus': self.context_engineering.write_output.documentation_focus,
                'compression_ratio': self.context_engineering.compress_output.compression_ratio,
                'token_budget': self.context_engineering.compress_output.token_budget,
                'isolation_strategy': self.context_engineering.isolate_output.isolation_strategy
            }
        }

class FourHostsResearchSystem:
    """Complete integrated research system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_service = MockLLMService()
        
        # Initialize components
        self._initialize_components()
        
        # Processing history
        self.processing_history: List[ResearchResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'average_classification_time': 0,
            'average_context_time': 0,
            'average_total_time': 0,
            'paradigm_distribution': {}
        }
        
    def _initialize_components(self):
        """Initialize classification and context engineering components"""
        # Import and initialize classification engine
        from classification_engine import (
            QueryAnalyzer, ParadigmClassifier, ClassificationEngine,
            HostParadigm, ParadigmScore
        )
        
        # Monkey patch the LLM classification method
        original_llm_method = ParadigmClassifier._llm_classification
        
        async def enhanced_llm_classification(self, query, features):
            # Use our mock LLM service
            llm_result = await self.llm_service.classify_with_llm(query, features)
            
            # Convert to expected format
            scores = {}
            for paradigm in HostParadigm:
                score_value = llm_result['scores'].get(paradigm.name, 0.1) * 10
                scores[paradigm] = ParadigmScore(
                    paradigm=paradigm,
                    score=score_value,
                    confidence=llm_result['confidence'],
                    reasoning=[llm_result['reasoning']],
                    keyword_matches=[]
                )
            
            return scores
        
        # Apply monkey patch
        ParadigmClassifier._llm_classification = enhanced_llm_classification
        
        # Initialize engines
        self.classification_engine = ClassificationEngine(use_llm=True, cache_enabled=True)
        self.classification_engine.classifier.llm_service = self.llm_service
        
        # Initialize context pipeline
        from context_engineering_pipeline import ContextEngineeringPipeline
        self.context_pipeline = ContextEngineeringPipeline()
        
    async def process_research_request(self, request: ResearchRequest) -> ResearchResult:
        """Process a complete research request"""
        start_time = time.time()
        metrics = {}
        
        logger.info(f"Processing research request: {request.query[:50]}...")
        
        try:
            # Step 1: Classification
            classification_start = time.time()
            classification = await self.classification_engine.classify_query(request.query)
            metrics['classification_time'] = time.time() - classification_start
            
            # Step 2: Context Engineering
            context_start = time.time()
            context_result = await self.context_pipeline.process_query(classification)
            metrics['context_engineering_time'] = time.time() - context_start
            
            # Step 3: Prepare search queries
            search_queries = self._prepare_search_queries(context_result)
            
            # Calculate total time
            metrics['total_time'] = time.time() - start_time
            
            # Create result
            result = ResearchResult(
                request=request,
                classification=classification,
                context_engineering=context_result,
                paradigm=classification.primary_paradigm.value,
                confidence=classification.confidence,
                search_queries=search_queries,
                processing_metrics=metrics
            )
            
            # Update metrics
            self._update_metrics(result)
            
            # Store in history
            self.processing_history.append(result)
            
            logger.info(f"Research request completed in {metrics['total_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing research request: {str(e)}")
            raise
    
    def _prepare_search_queries(self, context_result) -> List[Dict[str, Any]]:
        """Prepare final search queries from context engineering"""
        queries = []
        
        for sq in context_result.select_output.search_queries:
            query_data = {
                'query': sq['query'],
                'type': sq['type'],
                'weight': sq['weight'],
                'paradigm': context_result.classification.primary_paradigm.value,
                'source_filter': sq.get('source_filter'),
                'tools': context_result.select_output.tool_selections[:2]  # Top 2 tools
            }
            queries.append(query_data)
            
        return queries
    
    def _update_metrics(self, result: ResearchResult):
        """Update performance metrics"""
        self.performance_metrics['total_queries'] += 1
        
        # Update averages
        n = self.performance_metrics['total_queries']
        metrics = result.processing_metrics
        
        # Running average calculation
        for key in ['classification_time', 'context_engineering_time', 'total_time']:
            avg_key = f'average_{key}'.replace('_time', '')
            old_avg = self.performance_metrics.get(avg_key, 0)
            new_value = metrics.get(key, 0)
            self.performance_metrics[avg_key] = (old_avg * (n-1) + new_value) / n
        
        # Update paradigm distribution
        paradigm = result.paradigm
        dist = self.performance_metrics.get('paradigm_distribution', {})
        dist[paradigm] = dist.get(paradigm, 0) + 1
        self.performance_metrics['paradigm_distribution'] = dist
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add cache metrics
        cache_metrics = self.classification_engine.get_classification_metrics()
        metrics['cache_metrics'] = cache_metrics
        
        # Add pipeline metrics
        pipeline_metrics = self.context_pipeline.get_pipeline_metrics()
        metrics['pipeline_metrics'] = pipeline_metrics
        
        return metrics
    
    def export_results(self, filepath: str, last_n: Optional[int] = None):
        """Export research results to JSON"""
        results_to_export = self.processing_history[-last_n:] if last_n else self.processing_history
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_results': len(results_to_export),
            'results': [r.to_dict() for r in results_to_export],
            'system_metrics': self.get_system_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported {len(results_to_export)} results to {filepath}")

# --- Example Usage and Testing ---

async def demonstrate_system():
    """Demonstrate the complete system with various queries"""
    
    print("=" * 80)
    print("FOUR HOSTS RESEARCH SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = FourHostsResearchSystem()
    
    # Test queries covering all paradigms
    test_cases = [
        {
            'query': "How can small businesses compete with Amazon's monopolistic practices?",
            'expected_paradigm': 'strategic',  # MAEVE with DOLORES influence
            'description': 'Business competition with justice angle'
        },
        {
            'query': "What support resources exist for homeless veterans in Chicago?",
            'expected_paradigm': 'devotion',  # TEDDY
            'description': 'Community support query'
        },
        {
            'query': "Analyze the correlation between social media usage and teenage depression",
            'expected_paradigm': 'analytical',  # BERNARD
            'description': 'Academic research query'
        },
        {
            'query': "Expose how pharmaceutical companies manipulate drug prices",
            'expected_paradigm': 'revolutionary',  # DOLORES
            'description': 'Justice-oriented investigation'
        },
        {
            'query': "Best strategies to influence policy makers on climate change",
            'expected_paradigm': 'strategic',  # MAEVE
            'description': 'Strategic influence query'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test_case['description']}")
        print(f"QUERY: {test_case['query']}")
        print(f"EXPECTED: {test_case['expected_paradigm'].upper()}")
        
        # Create request
        request = ResearchRequest(
            query=test_case['query'],
            options={'test_mode': True}
        )
        
        # Process request
        result = await system.process_research_request(request)
        results.append(result)
        
        # Display results
        print(f"\nRESULTS:")
        print(f"  Paradigm: {result.paradigm.upper()} (confidence: {result.confidence:.1%})")
        print(f"  Classification Time: {result.processing_metrics['classification_time']:.3f}s")
        print(f"  Context Engineering Time: {result.processing_metrics['context_engineering_time']:.3f}s")
        print(f"  Total Time: {result.processing_metrics['total_time']:.3f}s")
        
        print(f"\n  Context Engineering Summary:")
        print(f"    - Documentation Focus: {result.context_engineering.write_output.documentation_focus[:60]}...")
        print(f"    - Search Queries Generated: {len(result.search_queries)}")
        print(f"    - Compression Ratio: {result.context_engineering.compress_output.compression_ratio:.0%}")
        print(f"    - Token Budget: {result.context_engineering.compress_output.token_budget}")
        print(f"    - Isolation Strategy: {result.context_engineering.isolate_output.isolation_strategy}")
        
        print(f"\n  Top 3 Search Queries:")
        for i, sq in enumerate(result.search_queries[:3], 1):
            print(f"    {i}. {sq['query'][:70]}...")
            print(f"       Type: {sq['type']}, Weight: {sq['weight']}")
    
    # Display system metrics
    print(f"\n{'='*80}")
    print("SYSTEM PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    metrics = system.get_system_metrics()
    print(f"Total Queries Processed: {metrics['total_queries']}")
    print(f"Average Classification Time: {metrics['average_classification']:.3f}s")
    print(f"Average Context Engineering Time: {metrics['average_context_engineering']:.3f}s")
    print(f"Average Total Time: {metrics['average_total']:.3f}s")
    
    print(f"\nParadigm Distribution:")
    for paradigm, count in metrics['paradigm_distribution'].items():
        percentage = (count / metrics['total_queries']) * 100
        print(f"  {paradigm:15} {count:3d} ({percentage:5.1f}%)")
    
    # Export results
    export_path = "research_results_demo.json"
    system.export_results(export_path)
    print(f"\nResults exported to: {export_path}")
    
    return system, results

# --- Performance Testing ---

async def performance_test():
    """Test system performance with multiple queries"""
    
    print("\n" + "="*80)
    print("PERFORMANCE TESTING")
    print("="*80)
    
    system = FourHostsResearchSystem()
    
    # Generate test queries
    test_queries = [
        "How to fight corporate monopolies?",
        "Support programs for disabled children",
        "Statistical analysis of income inequality",
        "Strategies to dominate market share",
        "Protecting endangered species",
        "Research on quantum computing applications",
        "Expose government surveillance programs",
        "How to optimize supply chain efficiency"
    ] * 5  # 40 queries total
    
    print(f"Processing {len(test_queries)} queries...")
    
    start_time = time.time()
    
    for i, query in enumerate(test_queries):
        request = ResearchRequest(query=query)
        await system.process_research_request(request)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_queries)} queries...")
    
    total_time = time.time() - start_time
    
    print(f"\nPerformance Results:")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per query: {total_time/len(test_queries):.3f}s")
    print(f"  Queries per second: {len(test_queries)/total_time:.2f}")
    
    # Cache hit rate
    cache_metrics = system.classification_engine.get_classification_metrics()
    if cache_metrics:
        cache_size = cache_metrics.get('cache_size', 0)
        hit_rate = (len(test_queries) - cache_size) / len(test_queries) * 100
        print(f"  Cache hit rate: {hit_rate:.1f}%")

# --- Main execution ---

async def main():
    """Main execution function"""
    
    # Run demonstration
    system, results = await demonstrate_system()
    
    # Run performance test
    await performance_test()
    
    print("\n" + "="*80)
    print("CLASSIFICATION & CONTEXT ENGINEERING SYSTEM READY")
    print("="*80)
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'quit' to exit.")
    
    while True:
        try:
            query = input("\nEnter research query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
                
            request = ResearchRequest(query=query)
            result = await system.process_research_request(request)
            
            print(f"\nParadigm: {result.paradigm.upper()} ({result.confidence:.1%} confidence)")
            print(f"Processing time: {result.processing_metrics['total_time']:.2f}s")
            print(f"\nTop search queries:")
            for i, sq in enumerate(result.search_queries[:3], 1):
                print(f"  {i}. {sq['query'][:60]}...")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    asyncio.run(main())
```