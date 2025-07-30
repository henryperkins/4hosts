"""
Self-Healing Paradigm Switching System for Four Hosts Research Application
Monitors performance and automatically switches paradigms for better results
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .classification_engine import HostParadigm, ClassificationResult
from .answer_generator import GeneratedAnswer
from .monitoring import monitoring_service

logger = logging.getLogger(__name__)


@dataclass
class ParadigmPerformanceMetrics:
    """Tracks performance metrics for a specific paradigm"""
    paradigm: HostParadigm
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_confidence_score: float = 0.0
    avg_synthesis_quality: float = 0.0
    avg_user_satisfaction: float = 0.0
    avg_response_time: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class QueryPerformanceRecord:
    """Records performance of a single query"""
    query_id: str
    query_text: str
    original_paradigm: HostParadigm
    switched_paradigm: Optional[HostParadigm]
    confidence_score: float
    synthesis_quality: float
    response_time: float
    user_feedback: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParadigmSwitchDecision:
    """Represents a decision to switch paradigms"""
    original_paradigm: HostParadigm
    recommended_paradigm: HostParadigm
    confidence: float
    reasons: List[str]
    expected_improvement: float
    risk_score: float


class SelfHealingSystem:
    """
    Monitors system performance and automatically switches paradigms
    when better results can be achieved with a different approach
    """
    
    def __init__(self):
        self.performance_metrics: Dict[HostParadigm, ParadigmPerformanceMetrics] = {
            paradigm: ParadigmPerformanceMetrics(paradigm=paradigm)
            for paradigm in HostParadigm
        }
        self.query_history: deque = deque(maxlen=10000)
        self.switch_history: List[Dict[str, Any]] = []
        self.learning_enabled = True
        self.min_queries_for_switch = 50  # Minimum queries before considering switch
        self.performance_threshold = 0.7  # Below this, consider switching
        self.improvement_threshold = 0.15  # Minimum expected improvement to switch
        
        # Performance weights for scoring
        self.weights = {
            "confidence": 0.3,
            "synthesis_quality": 0.3,
            "user_satisfaction": 0.2,
            "response_time": 0.1,
            "error_rate": 0.1,
        }
        
        # Paradigm affinity matrix (how well paradigms handle certain query types)
        self.paradigm_affinities = self._initialize_paradigm_affinities()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    def _initialize_paradigm_affinities(self) -> Dict[str, Dict[HostParadigm, float]]:
        """Initialize paradigm affinities for different query types"""
        return {
            "analytical": {
                HostParadigm.BERNARD: 0.9,
                HostParadigm.MAEVE: 0.7,
                HostParadigm.DOLORES: 0.4,
                HostParadigm.TEDDY: 0.3,
            },
            "strategic": {
                HostParadigm.MAEVE: 0.9,
                HostParadigm.BERNARD: 0.6,
                HostParadigm.DOLORES: 0.5,
                HostParadigm.TEDDY: 0.3,
            },
            "emotional": {
                HostParadigm.TEDDY: 0.9,
                HostParadigm.DOLORES: 0.7,
                HostParadigm.MAEVE: 0.4,
                HostParadigm.BERNARD: 0.2,
            },
            "revolutionary": {
                HostParadigm.DOLORES: 0.9,
                HostParadigm.TEDDY: 0.6,
                HostParadigm.MAEVE: 0.5,
                HostParadigm.BERNARD: 0.3,
            },
        }

    async def record_query_performance(
        self,
        query_id: str,
        query_text: str,
        paradigm: HostParadigm,
        answer: Optional[GeneratedAnswer] = None,
        error: Optional[str] = None,
        response_time: float = 0.0,
    ) -> None:
        """Record performance metrics for a query"""
        # Create performance record
        record = QueryPerformanceRecord(
            query_id=query_id,
            query_text=query_text,
            original_paradigm=paradigm,
            switched_paradigm=None,
            confidence_score=answer.confidence_score if answer else 0.0,
            synthesis_quality=answer.synthesis_quality if answer else 0.0,
            response_time=response_time,
            error=error,
        )
        
        # Update paradigm metrics
        metrics = self.performance_metrics[paradigm]
        metrics.total_queries += 1
        
        if error:
            metrics.failed_queries += 1
            error_type = self._categorize_error(error)
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
        else:
            metrics.successful_queries += 1
            
            # Update running averages
            n = metrics.successful_queries
            metrics.avg_confidence_score = self._update_average(
                metrics.avg_confidence_score, record.confidence_score, n
            )
            metrics.avg_synthesis_quality = self._update_average(
                metrics.avg_synthesis_quality, record.synthesis_quality, n
            )
            metrics.avg_response_time = self._update_average(
                metrics.avg_response_time, response_time, n
            )
            
            # Add to recent scores for trend analysis
            metrics.recent_scores.append({
                "confidence": record.confidence_score,
                "quality": record.synthesis_quality,
                "time": response_time,
                "timestamp": datetime.now(),
            })
        
        metrics.last_updated = datetime.now()
        
        # Store in query history
        self.query_history.append(record)
        
        # Check if paradigm switch needed
        if self.learning_enabled and metrics.total_queries >= self.min_queries_for_switch:
            switch_decision = await self._evaluate_paradigm_switch(paradigm, record)
            if switch_decision:
                await self._execute_paradigm_switch(record, switch_decision)
        
        # Send metrics to monitoring service
        await monitoring_service.track_paradigm_performance(paradigm, record.__dict__)

    async def record_user_feedback(
        self, query_id: str, satisfaction_score: float
    ) -> None:
        """Record user feedback for a query"""
        # Find the query record
        for record in self.query_history:
            if record.query_id == query_id:
                record.user_feedback = satisfaction_score
                
                # Update paradigm satisfaction metric
                paradigm = record.switched_paradigm or record.original_paradigm
                metrics = self.performance_metrics[paradigm]
                
                # Update average satisfaction
                feedback_count = sum(
                    1 for r in self.query_history 
                    if r.user_feedback is not None and 
                    (r.original_paradigm == paradigm or r.switched_paradigm == paradigm)
                )
                
                metrics.avg_user_satisfaction = self._update_average(
                    metrics.avg_user_satisfaction, satisfaction_score, feedback_count
                )
                
                # If satisfaction is low, increase switch likelihood
                if satisfaction_score < 0.5:
                    await self._analyze_failure_pattern(record)
                
                break

    async def _evaluate_paradigm_switch(
        self, current_paradigm: HostParadigm, record: QueryPerformanceRecord
    ) -> Optional[ParadigmSwitchDecision]:
        """Evaluate whether to switch paradigms based on performance"""
        current_metrics = self.performance_metrics[current_paradigm]
        
        # Calculate current paradigm performance score
        current_score = self._calculate_paradigm_score(current_metrics)
        
        # Check if performance is below threshold
        if current_score >= self.performance_threshold:
            return None
        
        # Identify query type for affinity matching
        query_type = self._identify_query_type(record.query_text)
        
        # Find best alternative paradigm
        best_alternative = None
        best_expected_score = current_score
        
        for paradigm in HostParadigm:
            if paradigm == current_paradigm:
                continue
            
            # Calculate expected performance
            alternative_metrics = self.performance_metrics[paradigm]
            base_score = self._calculate_paradigm_score(alternative_metrics)
            
            # Apply affinity bonus
            affinity_bonus = self.paradigm_affinities.get(query_type, {}).get(paradigm, 0.5) * 0.2
            expected_score = base_score + affinity_bonus
            
            # Check if this is better than current best
            if expected_score > best_expected_score + self.improvement_threshold:
                best_alternative = paradigm
                best_expected_score = expected_score
        
        if not best_alternative:
            return None
        
        # Calculate switch confidence and risk
        improvement = best_expected_score - current_score
        confidence = min(0.95, improvement * 2)  # Higher improvement = higher confidence
        
        # Calculate risk based on alternative paradigm's consistency
        alt_metrics = self.performance_metrics[best_alternative]
        risk_score = self._calculate_risk_score(alt_metrics)
        
        # Generate reasons for switch
        reasons = self._generate_switch_reasons(
            current_paradigm, best_alternative, current_metrics, alt_metrics, query_type
        )
        
        return ParadigmSwitchDecision(
            original_paradigm=current_paradigm,
            recommended_paradigm=best_alternative,
            confidence=confidence,
            reasons=reasons,
            expected_improvement=improvement,
            risk_score=risk_score,
        )

    async def _execute_paradigm_switch(
        self, record: QueryPerformanceRecord, decision: ParadigmSwitchDecision
    ) -> None:
        """Execute a paradigm switch"""
        logger.info(
            f"Switching paradigm from {decision.original_paradigm} to {decision.recommended_paradigm} "
            f"for query {record.query_id}. Expected improvement: {decision.expected_improvement:.2f}"
        )
        
        # Record the switch
        record.switched_paradigm = decision.recommended_paradigm
        
        # Store switch history
        self.switch_history.append({
            "query_id": record.query_id,
            "timestamp": datetime.now(),
            "original": decision.original_paradigm.value,
            "switched_to": decision.recommended_paradigm.value,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "expected_improvement": decision.expected_improvement,
            "risk_score": decision.risk_score,
        })
        
        # Notify monitoring service
        await monitoring_service.track_paradigm_switch(decision)

    def _calculate_paradigm_score(self, metrics: ParadigmPerformanceMetrics) -> float:
        """Calculate overall performance score for a paradigm"""
        if metrics.total_queries == 0:
            return 0.5  # Neutral score for unknown performance
        
        # Calculate error rate
        error_rate = metrics.failed_queries / metrics.total_queries
        
        # Normalize response time (lower is better, assume 10s is very bad)
        normalized_response_time = max(0, 1 - (metrics.avg_response_time / 10))
        
        # Calculate weighted score
        score = (
            self.weights["confidence"] * metrics.avg_confidence_score +
            self.weights["synthesis_quality"] * metrics.avg_synthesis_quality +
            self.weights["user_satisfaction"] * metrics.avg_user_satisfaction +
            self.weights["response_time"] * normalized_response_time +
            self.weights["error_rate"] * (1 - error_rate)
        )
        
        # Apply recency bias - recent performance matters more
        if metrics.recent_scores:
            recent_avg_confidence = sum(s["confidence"] for s in metrics.recent_scores) / len(metrics.recent_scores)
            recent_avg_quality = sum(s["quality"] for s in metrics.recent_scores) / len(metrics.recent_scores)
            
            recent_score = (recent_avg_confidence + recent_avg_quality) / 2
            score = score * 0.7 + recent_score * 0.3
        
        return score

    def _calculate_risk_score(self, metrics: ParadigmPerformanceMetrics) -> float:
        """Calculate risk score for switching to a paradigm"""
        if metrics.total_queries < 10:
            return 0.8  # High risk due to limited data
        
        # Calculate variance in recent performance
        if len(metrics.recent_scores) < 10:
            return 0.5  # Medium risk
        
        confidences = [s["confidence"] for s in metrics.recent_scores]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Higher variance = higher risk
        risk = min(0.9, variance * 2)
        
        # Factor in error rate
        error_rate = metrics.failed_queries / metrics.total_queries
        risk = risk * 0.7 + error_rate * 0.3
        
        return risk

    def _identify_query_type(self, query_text: str) -> str:
        """Identify the type of query for affinity matching"""
        query_lower = query_text.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ["analyze", "data", "statistics", "research", "study"]):
            return "analytical"
        elif any(word in query_lower for word in ["strategy", "plan", "compete", "business", "market"]):
            return "strategic"
        elif any(word in query_lower for word in ["help", "support", "feel", "care", "community"]):
            return "emotional"
        elif any(word in query_lower for word in ["change", "revolution", "fight", "justice", "system"]):
            return "revolutionary"
        else:
            return "general"

    def _generate_switch_reasons(
        self,
        current: HostParadigm,
        alternative: HostParadigm,
        current_metrics: ParadigmPerformanceMetrics,
        alt_metrics: ParadigmPerformanceMetrics,
        query_type: str,
    ) -> List[str]:
        """Generate human-readable reasons for paradigm switch"""
        reasons = []
        
        # Performance comparison
        if alt_metrics.avg_confidence_score > current_metrics.avg_confidence_score + 0.1:
            reasons.append(
                f"{alternative.value} shows {(alt_metrics.avg_confidence_score - current_metrics.avg_confidence_score)*100:.1f}% "
                f"higher confidence scores"
            )
        
        # Error rate comparison
        if current_metrics.total_queries > 0 and alt_metrics.total_queries > 0:
            current_error_rate = current_metrics.failed_queries / current_metrics.total_queries
            alt_error_rate = alt_metrics.failed_queries / alt_metrics.total_queries
            
            if alt_error_rate < current_error_rate - 0.05:
                reasons.append(
                    f"{alternative.value} has {(current_error_rate - alt_error_rate)*100:.1f}% "
                    f"lower error rate"
                )
        
        # Query type affinity
        current_affinity = self.paradigm_affinities.get(query_type, {}).get(current, 0.5)
        alt_affinity = self.paradigm_affinities.get(query_type, {}).get(alternative, 0.5)
        
        if alt_affinity > current_affinity:
            reasons.append(
                f"{alternative.value} is better suited for {query_type} queries"
            )
        
        # User satisfaction
        if alt_metrics.avg_user_satisfaction > current_metrics.avg_user_satisfaction + 0.1:
            reasons.append(
                f"Users report higher satisfaction with {alternative.value} paradigm"
            )
        
        # Response time
        if alt_metrics.avg_response_time < current_metrics.avg_response_time * 0.8:
            reasons.append(
                f"{alternative.value} responds {(1 - alt_metrics.avg_response_time/current_metrics.avg_response_time)*100:.0f}% faster"
            )
        
        if not reasons:
            reasons.append(f"Overall performance metrics favor {alternative.value}")
        
        return reasons

    async def _analyze_failure_pattern(self, record: QueryPerformanceRecord) -> None:
        """Analyze failure patterns to improve future switches"""
        # Group recent failures by paradigm
        recent_failures = [
            r for r in self.query_history
            if r.timestamp > datetime.now() - timedelta(hours=24) and
            (r.error is not None or (r.user_feedback is not None and r.user_feedback < 0.5))
        ]
        
        failure_patterns = defaultdict(list)
        for failure in recent_failures:
            paradigm = failure.switched_paradigm or failure.original_paradigm
            failure_patterns[paradigm].append({
                "query_type": self._identify_query_type(failure.query_text),
                "error_type": self._categorize_error(failure.error) if failure.error else "low_satisfaction",
                "confidence": failure.confidence_score,
            })
        
        # Update paradigm affinities based on failures
        for paradigm, failures in failure_patterns.items():
            if len(failures) >= 5:  # Significant pattern
                # Reduce affinity for query types that frequently fail
                query_type_counts = defaultdict(int)
                for failure in failures:
                    query_type_counts[failure["query_type"]] += 1
                
                for query_type, count in query_type_counts.items():
                    if count >= 3:
                        current_affinity = self.paradigm_affinities[query_type][paradigm]
                        # Reduce affinity by 5% per pattern
                        self.paradigm_affinities[query_type][paradigm] = max(0.1, current_affinity * 0.95)
                        
                        logger.info(
                            f"Reduced {paradigm.value} affinity for {query_type} queries "
                            f"to {self.paradigm_affinities[query_type][paradigm]:.2f}"
                        )

    def _categorize_error(self, error: str) -> str:
        """Categorize error types for analysis"""
        error_lower = error.lower() if error else ""
        
        if "timeout" in error_lower:
            return "timeout"
        elif "api" in error_lower or "rate limit" in error_lower:
            return "api_error"
        elif "memory" in error_lower or "resource" in error_lower:
            return "resource_error"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation_error"
        else:
            return "unknown"

    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average"""
        if count <= 1:
            return new_value
        return (current_avg * (count - 1) + new_value) / count

    async def _monitoring_loop(self) -> None:
        """Background loop for monitoring and optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Analyze overall system health
                await self._analyze_system_health()
                
                # Optimize paradigm affinities based on recent performance
                await self._optimize_affinities()
                
                # Clean up old data
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _analyze_system_health(self) -> None:
        """Analyze overall system health and performance"""
        total_queries = sum(m.total_queries for m in self.performance_metrics.values())
        if total_queries == 0:
            return
        
        # Calculate system-wide metrics
        total_errors = sum(m.failed_queries for m in self.performance_metrics.values())
        system_error_rate = total_errors / total_queries
        
        avg_confidence = sum(
            m.avg_confidence_score * m.successful_queries 
            for m in self.performance_metrics.values()
        ) / (total_queries - total_errors)
        
        # Log system health
        logger.info(
            f"System Health - Total Queries: {total_queries}, "
            f"Error Rate: {system_error_rate:.2%}, "
            f"Avg Confidence: {avg_confidence:.2f}"
        )
        
        # Alert if system-wide issues detected
        if system_error_rate > 0.1:
            logger.warning(f"High system-wide error rate: {system_error_rate:.2%}")
        
        if avg_confidence < 0.6:
            logger.warning(f"Low system-wide confidence: {avg_confidence:.2f}")

    async def _optimize_affinities(self) -> None:
        """Optimize paradigm affinities based on recent performance"""
        # Analyze successful queries from the last hour
        recent_successes = [
            r for r in self.query_history
            if r.timestamp > datetime.now() - timedelta(hours=1) and
            r.error is None and
            r.confidence_score > 0.8
        ]
        
        if len(recent_successes) < 20:
            return  # Not enough data
        
        # Group by paradigm and query type
        success_patterns = defaultdict(lambda: defaultdict(list))
        for success in recent_successes:
            paradigm = success.switched_paradigm or success.original_paradigm
            query_type = self._identify_query_type(success.query_text)
            success_patterns[paradigm][query_type].append(success.confidence_score)
        
        # Update affinities based on success patterns
        for paradigm, query_types in success_patterns.items():
            for query_type, scores in query_types.items():
                if len(scores) >= 3:
                    avg_score = sum(scores) / len(scores)
                    current_affinity = self.paradigm_affinities[query_type][paradigm]
                    
                    # Increase affinity if performing well
                    if avg_score > 0.85:
                        new_affinity = min(0.95, current_affinity * 1.05)
                        self.paradigm_affinities[query_type][paradigm] = new_affinity
                        
                        logger.info(
                            f"Increased {paradigm.value} affinity for {query_type} queries "
                            f"to {new_affinity:.2f} based on high performance"
                        )

    async def _cleanup_old_data(self) -> None:
        """Clean up old performance data"""
        # Keep detailed metrics for recent queries only
        for metrics in self.performance_metrics.values():
            # Clear old error types
            if len(metrics.error_types) > 50:
                # Keep only the most common error types
                sorted_errors = sorted(
                    metrics.error_types.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
                metrics.error_types = dict(sorted_errors)

    def get_paradigm_recommendation(
        self, query: str, current_paradigm: HostParadigm
    ) -> Optional[HostParadigm]:
        """Get paradigm recommendation for a query"""
        # Quick recommendation without full evaluation
        query_type = self._identify_query_type(query)
        
        # Get affinities for this query type
        affinities = self.paradigm_affinities.get(query_type, {})
        
        # Find paradigm with highest affinity and good performance
        best_paradigm = current_paradigm
        best_score = 0.0
        
        for paradigm, affinity in affinities.items():
            metrics = self.performance_metrics[paradigm]
            
            # Skip if insufficient data
            if metrics.total_queries < 10:
                continue
            
            # Calculate combined score
            performance_score = self._calculate_paradigm_score(metrics)
            combined_score = affinity * 0.4 + performance_score * 0.6
            
            if combined_score > best_score:
                best_score = combined_score
                best_paradigm = paradigm
        
        # Only recommend if significantly better
        current_metrics = self.performance_metrics[current_paradigm]
        current_score = self._calculate_paradigm_score(current_metrics)
        current_affinity = affinities.get(current_paradigm, 0.5)
        current_combined = current_affinity * 0.4 + current_score * 0.6
        
        if best_score > current_combined + 0.1:
            return best_paradigm
        
        return None

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "paradigm_metrics": {},
            "switch_statistics": {
                "total_switches": len(self.switch_history),
                "recent_switches": len([
                    s for s in self.switch_history 
                    if datetime.fromisoformat(s["timestamp"]) > datetime.now() - timedelta(hours=24)
                ]),
            },
            "system_health": {},
            "recommendations": [],
        }
        
        # Paradigm-specific metrics
        for paradigm, metrics in self.performance_metrics.items():
            if metrics.total_queries > 0:
                report["paradigm_metrics"][paradigm.value] = {
                    "total_queries": metrics.total_queries,
                    "success_rate": metrics.successful_queries / metrics.total_queries,
                    "avg_confidence": metrics.avg_confidence_score,
                    "avg_quality": metrics.avg_synthesis_quality,
                    "avg_satisfaction": metrics.avg_user_satisfaction,
                    "avg_response_time": metrics.avg_response_time,
                    "performance_score": self._calculate_paradigm_score(metrics),
                }
        
        # System health metrics
        total_queries = sum(m.total_queries for m in self.performance_metrics.values())
        if total_queries > 0:
            total_errors = sum(m.failed_queries for m in self.performance_metrics.values())
            report["system_health"] = {
                "total_queries": total_queries,
                "overall_success_rate": 1 - (total_errors / total_queries),
                "paradigm_distribution": {
                    p.value: m.total_queries / total_queries 
                    for p, m in self.performance_metrics.items()
                },
            }
        
        # Generate recommendations
        for paradigm, metrics in self.performance_metrics.items():
            score = self._calculate_paradigm_score(metrics)
            if score < 0.6 and metrics.total_queries > 20:
                report["recommendations"].append({
                    "paradigm": paradigm.value,
                    "issue": "Low performance score",
                    "score": score,
                    "suggestion": f"Consider reducing usage of {paradigm.value} paradigm or investigating performance issues",
                })
        
        return report

    async def force_paradigm_switch(
        self, query_id: str, new_paradigm: HostParadigm, reason: str
    ) -> None:
        """Manually force a paradigm switch (for admin intervention)"""
        # Find the query record
        for record in self.query_history:
            if record.query_id == query_id:
                original = record.switched_paradigm or record.original_paradigm
                record.switched_paradigm = new_paradigm
                
                # Record manual switch
                self.switch_history.append({
                    "query_id": query_id,
                    "timestamp": datetime.now().isoformat(),
                    "original": original.value,
                    "switched_to": new_paradigm.value,
                    "confidence": 1.0,
                    "reasons": [f"Manual switch: {reason}"],
                    "expected_improvement": 0.0,
                    "risk_score": 0.0,
                    "manual": True,
                })
                
                logger.info(
                    f"Manual paradigm switch for query {query_id}: "
                    f"{original.value} -> {new_paradigm.value}. Reason: {reason}"
                )
                break


# Create singleton instance
self_healing_system = SelfHealingSystem()