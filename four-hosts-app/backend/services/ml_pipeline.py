"""
Machine Learning Pipeline for Continuous Improvement
Trains and updates models based on user feedback and system performance
"""

import asyncio
import structlog
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

# ML imports (assuming scikit-learn is available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    from logging_config import configure_logging
    configure_logging()
    logger = structlog.get_logger(__name__)
    logger.warning("Scikit-learn not available - ML pipeline will use mock training")

# Import HF zero-shot classifier
try:
    from .hf_zero_shot import async_predict_paradigm as hf_predict_async
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    from logging_config import configure_logging
    configure_logging()
    logger = structlog.get_logger(__name__)
    logger.info("HF zero-shot classifier not available")

from .classification_engine import HostParadigm, QueryFeatures
from .self_healing_system import self_healing_system

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


@dataclass
class TrainingExample:
    """Represents a training example for the ML pipeline"""
    query_id: str
    query_text: str
    features: QueryFeatures
    true_paradigm: HostParadigm
    predicted_paradigm: HostParadigm
    confidence_score: float
    user_satisfaction: Optional[float] = None
    synthesis_quality: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Tracks model performance metrics"""
    model_version: str
    training_date: datetime
    training_samples: int
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]


@dataclass
class ModelUpdate:
    """Represents a model update event"""
    update_id: str
    previous_version: str
    new_version: str
    improvement: float
    changes: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class MLPipeline:
    """
    Manages the machine learning pipeline for continuous model improvement
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.min_training_samples = 100
        self.retrain_interval = timedelta(days=7)
        self.performance_threshold = 0.85
        self.improvement_threshold = 0.02  # 2% improvement required
        
        # Model components
        self.vectorizer = None
        self.scaler = None
        self.paradigm_classifier = None
        self.feature_extractors = {}
        
        # Training data
        self.training_examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        
        # Model tracking
        self.current_model_version = "v1.0.0"
        self.model_performance_history: List[ModelPerformance] = []
        self.model_updates: List[ModelUpdate] = []
        self.last_training_date = None
        
        # Feature engineering
        self.feature_generators = self._initialize_feature_generators()
        self.engineered_feature_keys = [
            "urgency_score",
            "complexity_score",
            "num_entities",
            "num_intent_signals",
            "domain_encoded",
            "confidence_score",
            "user_satisfaction",
            "synthesis_quality",
        ]
        self.additional_feature_keys = list(self.feature_generators.keys()) + self.engineered_feature_keys

        # Concurrency for retraining
        self._retrain_lock = asyncio.Lock()
        self._retraining = False
        
        # Load existing model if available
        self._load_existing_model()
        
        # Start training loop
        # asyncio.create_task(self._training_loop())  # Commented out - needs to be started after event loop is running

    def _initialize_feature_generators(self) -> Dict[str, Any]:
        """Initialize feature generation functions"""
        return {
            "query_length": lambda q: len(q.split()),
            "avg_word_length": lambda q: np.mean([len(w) for w in q.split()]) if q.split() else 0,
            "question_marks": lambda q: q.count("?"),
            "exclamation_marks": lambda q: q.count("!"),
            "capital_ratio": lambda q: sum(1 for c in q if c.isupper()) / len(q) if q else 0,
            "numeric_count": lambda q: sum(1 for c in q if c.isdigit()),
            "special_chars": lambda q: sum(1 for c in q if not c.isalnum() and not c.isspace()),
        }

    def _load_existing_model(self) -> None:
        """Load existing model from disk if available"""
        model_path = self.model_dir / f"paradigm_classifier_{self.current_model_version}.pkl"
        vectorizer_path = self.model_dir / f"vectorizer_{self.current_model_version}.pkl"
        
        if ML_AVAILABLE and model_path.exists() and vectorizer_path.exists():
            try:
                self.paradigm_classifier = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Load performance history
                perf_path = self.model_dir / "performance_history.json"
                if perf_path.exists():
                    with open(perf_path, "r") as f:
                        data = json.load(f)
                        loaded: List[ModelPerformance] = []
                        for perf in data:
                            try:
                                if isinstance(perf.get("training_date"), str):
                                    perf["training_date"] = datetime.fromisoformat(perf["training_date"])
                            except Exception:
                                # Keep original if parsing fails
                                pass
                            loaded.append(ModelPerformance(**perf))
                        self.model_performance_history = loaded
                
                logger.info(f"Loaded existing model {self.current_model_version}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._initialize_new_model()
        else:
            self._initialize_new_model()

    def _initialize_new_model(self) -> None:
        """Initialize a new model from scratch"""
        if ML_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words="english",
                min_df=2,
            )
            
            # StandardScaler is unnecessary for tree-based models; omit to reduce memory/CPU
            self.scaler = None
            
            # Use ensemble of classifiers
            self.paradigm_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
            
            logger.info("Initialized new ML model")
        else:
            logger.info("ML libraries not available - using mock model")

    async def record_training_example(
        self,
        query_id: str,
        query_text: str,
        features: QueryFeatures,
        predicted_paradigm: HostParadigm,
        true_paradigm: Optional[HostParadigm] = None,
        user_feedback: Optional[float] = None,
        synthesis_quality: Optional[float] = None,
    ) -> None:
        """Record a training example from system usage"""
        # If no true paradigm provided, infer from feedback
        if true_paradigm is None and user_feedback is not None:
            # High satisfaction suggests correct paradigm
            if user_feedback >= 0.8:
                true_paradigm = predicted_paradigm
            else:
                # Low satisfaction might indicate wrong paradigm
                # Use self-healing system recommendation
                recommended = self_healing_system.get_paradigm_recommendation(
                    query_text, predicted_paradigm
                )
                if recommended:
                    true_paradigm = recommended
        
        example = TrainingExample(
            query_id=query_id,
            query_text=query_text,
            features=features,
            true_paradigm=true_paradigm or predicted_paradigm,
            predicted_paradigm=predicted_paradigm,
            confidence_score=features.confidence_score if hasattr(features, 'confidence_score') else 0.5,
            user_satisfaction=user_feedback,
            synthesis_quality=synthesis_quality,
        )
        
        self.training_examples.append(example)
        
        # Trigger retraining if enough new examples
        if len(self.training_examples) >= self.min_training_samples:
            if (
                self.last_training_date is None or
                datetime.now() - self.last_training_date > self.retrain_interval
            ):
                # Serialize retraining to avoid overlapping tasks
                if not self._retraining:
                    self._retraining = True
                    asyncio.create_task(self._retrain_model())

    async def _retrain_model(self) -> None:
        """Retrain the classification model with accumulated examples"""
        if not ML_AVAILABLE:
            logger.info("ML libraries not available - skipping retraining")
            return
        
        async with self._retrain_lock:
            logger.info(f"Starting model retraining with {len(self.training_examples)} examples")
            
            try:
                # Prepare training data
                X_text = [ex.query_text for ex in self.training_examples]
                X_features = self._extract_additional_features(self.training_examples)
                y = [ex.true_paradigm.value for ex in self.training_examples]
                
                # Split data
                # Use stratify only if each class has at least 2 samples to avoid ValueError
                class_counts = Counter(y)
                can_stratify = all(cnt >= 2 for cnt in class_counts.values())
                X_text_train, X_text_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
                    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
                )
                
                # Vectorize text
                X_text_train_vec = self.vectorizer.fit_transform(X_text_train)
                X_text_val_vec = self.vectorizer.transform(X_text_val)
                
                # Combine text and additional features
                X_train = self._combine_features(X_text_train_vec, X_feat_train)
                X_val = self._combine_features(X_text_val_vec, X_feat_val)
                
                # Scaling omitted for tree-based model to reduce memory and CPU
                # (GradientBoosting handles unscaled numeric features adequately)
                
                # Train new model
                new_classifier = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                )
                
                # Class balancing via sample weights (inverse frequency)
                class_counts_train = Counter(y_train)
                n_classes_train = max(1, len(class_counts_train))
                total_train = len(y_train)
                class_weights = {
                    cls: (total_train / (n_classes_train * cnt))
                    for cls, cnt in class_counts_train.items()
                }
                sample_weight = np.array([class_weights[label] for label in y_train], dtype=float)
                
                new_classifier.fit(X_train, y_train, sample_weight=sample_weight)
                
                # Evaluate new model
                y_pred = new_classifier.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                # Compare with current model
                should_update = await self._should_update_model(
                    new_classifier, accuracy, X_val, y_val
                )
                
                if should_update:
                    await self._update_model(new_classifier, accuracy, y_val, y_pred)
                else:
                    logger.info(
                        f"New model accuracy ({accuracy:.3f}) not sufficient improvement. "
                        f"Keeping current model."
                    )
                
                self.last_training_date = datetime.now()
            
            except Exception as e:
                logger.error(f"Error during model retraining: {e}")
            finally:
                # Ensure flag is cleared even if errors occur
                self._retraining = False

    def _extract_additional_features(
        self, examples: List[TrainingExample]
    ) -> np.ndarray:
        """Extract additional features beyond text"""
        features = []
        
        for ex in examples:
            feat_dict: Dict[str, float] = {}

            # Query-based features in deterministic order
            for name in self.feature_generators.keys():
                try:
                    feat_dict[name] = float(self.feature_generators[name](ex.query_text))
                except Exception:
                    feat_dict[name] = 0.0

            # Features from QueryFeatures object
            if ex.features:
                try:
                    feat_dict["urgency_score"] = float(ex.features.urgency_score)
                    feat_dict["complexity_score"] = float(ex.features.complexity_score)
                    feat_dict["num_entities"] = float(len(ex.features.entities))
                    feat_dict["num_intent_signals"] = float(len(ex.features.intent_signals))
                except Exception:
                    feat_dict.setdefault("urgency_score", 0.0)
                    feat_dict.setdefault("complexity_score", 0.0)
                    feat_dict.setdefault("num_entities", 0.0)
                    feat_dict.setdefault("num_intent_signals", 0.0)

                # Domain features
                domain_map = {
                    "technical": 0, "business": 1, "social": 2,
                    "personal": 3, "academic": 4, "other": 5
                }
                try:
                    dom = ex.features.domain or "other"
                except Exception:
                    dom = "other"
                feat_dict["domain_encoded"] = float(domain_map.get(dom, 5))
            else:
                # Initialize engineered features when features are missing
                feat_dict.update({
                    "urgency_score": 0.0,
                    "complexity_score": 0.0,
                    "num_entities": 0.0,
                    "num_intent_signals": 0.0,
                    "domain_encoded": 5.0,
                })

            # Performance features (always present)
            feat_dict["confidence_score"] = float(ex.confidence_score)
            feat_dict["user_satisfaction"] = float(ex.user_satisfaction or 0.5)
            feat_dict["synthesis_quality"] = float(ex.synthesis_quality or 0.5)

            # Build row in deterministic order aligned with self.additional_feature_keys
            features.append([feat_dict.get(k, 0.0) for k in self.additional_feature_keys])

        return np.array(features, dtype=float)

    def _combine_features(
        self, text_features: Any, additional_features: np.ndarray
    ) -> np.ndarray:
        """Combine text and additional features"""
        if hasattr(text_features, "toarray"):
            text_array = text_features.toarray()
        else:
            text_array = text_features
        
        return np.hstack([text_array, additional_features])

    async def _should_update_model(
        self, new_model: Any, new_accuracy: float, X_val: np.ndarray, y_val: List[str]
    ) -> bool:
        """Determine if the new model should replace the current one"""
        if self.paradigm_classifier is None:
            return True  # No existing model
        
        # Test current model on validation set
        try:
            y_pred_current = self.paradigm_classifier.predict(X_val)
            current_accuracy = accuracy_score(y_val, y_pred_current)
            
            improvement = new_accuracy - current_accuracy
            
            # Update if significant improvement
            if improvement > self.improvement_threshold:
                logger.info(
                    f"New model shows {improvement:.3f} improvement in accuracy"
                )
                return True
            
            # Update if current model is performing poorly
            if current_accuracy < self.performance_threshold and new_accuracy > current_accuracy:
                logger.info(
                    f"Current model below threshold ({current_accuracy:.3f}). "
                    f"Updating to new model ({new_accuracy:.3f})"
                )
                return True
            
            # Check for specific paradigm improvements
            new_report = classification_report(y_val, new_model.predict(X_val), output_dict=True)
            current_report = classification_report(y_val, y_pred_current, output_dict=True)
            
            paradigm_improvements = 0
            for paradigm in HostParadigm:
                if paradigm.value in new_report and paradigm.value in current_report:
                    new_f1 = new_report[paradigm.value]["f1-score"]
                    current_f1 = current_report[paradigm.value]["f1-score"]
                    if new_f1 > current_f1 + 0.05:  # 5% F1 improvement
                        paradigm_improvements += 1
            
            if paradigm_improvements >= 2:
                logger.info(
                    f"New model shows improvement in {paradigm_improvements} paradigms"
                )
                return True
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return False
        
        return False

    async def _update_model(
        self, new_model: Any, accuracy: float, y_true: List[str], y_pred: List[str]
    ) -> None:
        """Update the current model with the new one"""
        # Generate new version number (robust semver patch bump, preserving leading 'v')
        current = self.current_model_version or "v1.0.0"
        has_v = current.startswith("v")
        core = current[1:] if has_v else current
        parts = core.split(".")
        while len(parts) < 3:
            parts.append("0")
        try:
            major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            major, minor, patch = 1, 0, 0
        patch += 1
        new_core = f"{major}.{minor}.{patch}"
        new_version = f"v{new_core}" if has_v else new_core
        
        # Calculate detailed metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract feature importance
        feature_importance = {}
        if hasattr(new_model, "feature_importances_"):
            # Get feature names from vectorizer
            feature_names = []
            if self.vectorizer is not None:
                feature_names.extend(self.vectorizer.get_feature_names_out())
            feature_names.extend(self.additional_feature_keys)
            
            # Map importance to names
            for idx, importance in enumerate(new_model.feature_importances_):
                if idx < len(feature_names):
                    feature_importance[feature_names[idx]] = float(importance)
        
        # Create performance record
        performance = ModelPerformance(
            model_version=new_version,
            training_date=datetime.now(),
            training_samples=len(self.training_examples),
            accuracy=accuracy,
            precision_per_class={
                cls: report[cls]["precision"] 
                for cls in report if cls not in ["accuracy", "macro avg", "weighted avg"]
            },
            recall_per_class={
                cls: report[cls]["recall"] 
                for cls in report if cls not in ["accuracy", "macro avg", "weighted avg"]
            },
            f1_per_class={
                cls: report[cls]["f1-score"] 
                for cls in report if cls not in ["accuracy", "macro avg", "weighted avg"]
            },
            confusion_matrix=cm.tolist(),
            feature_importance=dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]),  # Top 20 features
        )
        
        # Record update
        changes = self._identify_model_changes(performance)
        update = ModelUpdate(
            update_id=f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            previous_version=self.current_model_version,
            new_version=new_version,
            improvement=accuracy - (
                self.model_performance_history[-1].accuracy
                if self.model_performance_history else 0.0
            ),
            changes=changes,
        )
        
        # Save new model
        model_path = self.model_dir / f"paradigm_classifier_{new_version}.pkl"
        vectorizer_path = self.model_dir / f"vectorizer_{new_version}.pkl"
        
        joblib.dump(new_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.scaler, self.model_dir / f"scaler_{new_version}.pkl")
        
        # Update state
        self.paradigm_classifier = new_model
        self.current_model_version = new_version
        self.model_performance_history.append(performance)
        self.model_updates.append(update)
        
        # Save performance history
        self._save_performance_history()
        
        # Move training examples to validation
        self.validation_examples = self.training_examples[-1000:]  # Keep last 1000
        self.training_examples = []
        
        logger.info(
            f"Model updated to {new_version}. "
            f"Accuracy: {accuracy:.3f}, Improvement: {update.improvement:.3f}"
        )

    def _identify_model_changes(self, performance: ModelPerformance) -> List[str]:
        """Identify what changed in the new model"""
        changes = []
        
        if not self.model_performance_history:
            changes.append("Initial model training")
            return changes
        
        prev_perf = self.model_performance_history[-1]
        
        # Overall accuracy change
        acc_change = performance.accuracy - prev_perf.accuracy
        if abs(acc_change) > 0.01:
            changes.append(
                f"Overall accuracy {'improved' if acc_change > 0 else 'decreased'} "
                f"by {abs(acc_change):.1%}"
            )
        
        # Per-paradigm changes
        for paradigm in HostParadigm:
            if paradigm.value in performance.f1_per_class and paradigm.value in prev_perf.f1_per_class:
                f1_change = performance.f1_per_class[paradigm.value] - prev_perf.f1_per_class[paradigm.value]
                if abs(f1_change) > 0.05:
                    changes.append(
                        f"{paradigm.value} F1-score {'improved' if f1_change > 0 else 'decreased'} "
                        f"by {abs(f1_change):.1%}"
                    )
        
        # Feature importance changes
        if performance.feature_importance and prev_perf.feature_importance:
            top_features_new = set(list(performance.feature_importance.keys())[:5])
            top_features_old = set(list(prev_perf.feature_importance.keys())[:5])
            
            new_important = top_features_new - top_features_old
            if new_important:
                changes.append(f"New important features: {', '.join(new_important)}")
        
        return changes

    def _save_performance_history(self) -> None:
        """Save performance history to disk"""
        perf_data = [
            {
                "model_version": p.model_version,
                "training_date": p.training_date.isoformat(),
                "training_samples": p.training_samples,
                "accuracy": p.accuracy,
                "precision_per_class": p.precision_per_class,
                "recall_per_class": p.recall_per_class,
                "f1_per_class": p.f1_per_class,
                "confusion_matrix": p.confusion_matrix,
                "feature_importance": p.feature_importance,
            }
            for p in self.model_performance_history
        ]
        
        tmp_path = self.model_dir / "performance_history.json.tmp"
        final_path = self.model_dir / "performance_history.json"
        with open(tmp_path, "w") as f:
            json.dump(perf_data, f, indent=2)
        try:
            tmp_path.replace(final_path)
        except Exception:
            # Fallback to regular write if atomic replace fails
            with open(final_path, "w") as f:
                json.dump(perf_data, f, indent=2)

    async def _training_loop(self) -> None:
        """Background loop for periodic model evaluation and retraining"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Analyze recent performance
                await self._analyze_recent_performance()
                
                # Check if retraining needed
                if self._should_retrain():
                    await self._retrain_model()
                
                # Clean up old data
                await self._cleanup_old_examples()
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")

    async def _analyze_recent_performance(self) -> None:
        """Analyze recent model performance"""
        if not self.validation_examples:
            return
        
        recent_examples = [
            ex for ex in self.validation_examples
            if ex.timestamp > datetime.now() - timedelta(days=1)
        ]
        
        if len(recent_examples) < 50:
            return
        
        # Calculate recent accuracy
        correct = sum(
            1 for ex in recent_examples 
            if ex.predicted_paradigm == ex.true_paradigm
        )
        recent_accuracy = correct / len(recent_examples)
        
        # Log if performance is degrading
        if recent_accuracy < self.performance_threshold * 0.9:
            logger.warning(
                f"Recent model accuracy ({recent_accuracy:.3f}) "
                f"below threshold. Consider retraining."
            )
            
            # Analyze which paradigms are problematic
            paradigm_errors = defaultdict(int)
            for ex in recent_examples:
                if ex.predicted_paradigm != ex.true_paradigm:
                    paradigm_errors[ex.true_paradigm.value] += 1
            
            worst_paradigm = max(paradigm_errors.items(), key=lambda x: x[1])
            logger.info(
                f"Paradigm with most errors: {worst_paradigm[0]} "
                f"({worst_paradigm[1]} errors)"
            )

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained"""
        # Check if enough new examples
        if len(self.training_examples) < self.min_training_samples:
            return False
        
        # Check if enough time has passed
        if self.last_training_date:
            time_since_training = datetime.now() - self.last_training_date
            if time_since_training < self.retrain_interval:
                return False
        
        # Check if performance is degrading
        if self.model_performance_history:
            recent_perf = self.model_performance_history[-1]
            if recent_perf.accuracy < self.performance_threshold:
                return True
        
        return True

    async def _cleanup_old_examples(self) -> None:
        """Clean up old training examples"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Keep recent examples
        self.training_examples = [
            ex for ex in self.training_examples
            if ex.timestamp > cutoff_date
        ]
        
        self.validation_examples = [
            ex for ex in self.validation_examples
            if ex.timestamp > cutoff_date
        ]

    async def predict_paradigm(
        self, query_text: str, features: QueryFeatures
    ) -> Tuple[HostParadigm, float]:
        """Predict paradigm using the current model"""
        # Try HF zero-shot classifier first if available
        if HF_AVAILABLE:
            try:
                label, score = await hf_predict_async(query_text)
                paradigm = HostParadigm(label)
                logger.debug(f"HF zero-shot prediction: {paradigm.value} (confidence: {score:.3f})")
                return paradigm, score
            except Exception as e:
                logger.debug(f"HF zero-shot fallback ({e}); trying ML model")
        
        # Fall back to ML model if HF not available or failed
        if not ML_AVAILABLE or self.paradigm_classifier is None:
            # Fallback to rule-based
            return self._rule_based_prediction(query_text, features)
        
        try:
            # Ensure vectorizer is fitted before use
            if not hasattr(self.vectorizer, "vocabulary_") or self.vectorizer.vocabulary_ is None:
                # Not enough data to train – fall back to rule-based classification
                raise ValueError("Vectorizer_not_fitted")

            # Vectorize query
            X_text = self.vectorizer.transform([query_text])
            
            # Extract additional features
            example = TrainingExample(
                query_id="predict",
                query_text=query_text,
                features=features,
                true_paradigm=HostParadigm.DOLORES,  # Dummy
                predicted_paradigm=HostParadigm.DOLORES,  # Dummy
                confidence_score=0.5,
            )
            
            X_features = self._extract_additional_features([example])
            
            # Combine features
            X = self._combine_features(X_text, X_features)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Predict
            prediction = self.paradigm_classifier.predict(X)[0]
            
            # Get confidence (probability of predicted class)
            probabilities = self.paradigm_classifier.predict_proba(X)[0]
            paradigm_index = list(self.paradigm_classifier.classes_).index(prediction)
            confidence = probabilities[paradigm_index]
            
            # Convert string to enum
            paradigm = HostParadigm(prediction)
            
            return paradigm, confidence
            
        except Exception as e:
            # Log as debug for expected not-fitted cases to avoid flooding error logs
            log_method = logger.debug if "Vectorizer_not_fitted" in str(e) else logger.error
            log_method(f"Error in ML prediction: {e}")
            return self._rule_based_prediction(query_text, features)

    def _rule_based_prediction(
        self, query_text: str, features: QueryFeatures
    ) -> Tuple[HostParadigm, float]:
        """Fallback rule-based prediction"""
        query_lower = query_text.lower()
        
        # Simple keyword matching
        if any(word in query_lower for word in ["analyze", "data", "research", "study"]):
            return HostParadigm.BERNARD, 0.7
        elif any(word in query_lower for word in ["strategy", "business", "compete", "market"]):
            return HostParadigm.MAEVE, 0.7
        elif any(word in query_lower for word in ["help", "support", "care", "community"]):
            return HostParadigm.TEDDY, 0.7
        elif any(word in query_lower for word in ["change", "fight", "justice", "system"]):
            return HostParadigm.DOLORES, 0.7
        else:
            # Default based on complexity
            if features.complexity_score > 0.7:
                return HostParadigm.BERNARD, 0.5
            else:
                return HostParadigm.TEDDY, 0.5

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        info = {
            "current_version": self.current_model_version,
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
            "training_examples": len(self.training_examples),
            "validation_examples": len(self.validation_examples),
            "ml_available": ML_AVAILABLE,
        }
        
        if self.model_performance_history:
            latest_perf = self.model_performance_history[-1]
            info["current_accuracy"] = latest_perf.accuracy
            info["paradigm_performance"] = latest_perf.f1_per_class
            info["top_features"] = list(latest_perf.feature_importance.keys())[:10]
        
        if self.model_updates:
            info["recent_updates"] = [
                {
                    "version": update.new_version,
                    "date": update.timestamp.isoformat(),
                    "improvement": update.improvement,
                    "changes": update.changes[:3],  # Top 3 changes
                }
                for update in self.model_updates[-5:]  # Last 5 updates
            ]
        
        return info

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            "total_examples": len(self.training_examples) + len(self.validation_examples),
            "examples_by_paradigm": defaultdict(int),
            "avg_confidence": 0.0,
            "avg_user_satisfaction": 0.0,
            "paradigm_accuracy": {},
        }
        
        all_examples = self.training_examples + self.validation_examples
        
        # Count by paradigm
        for ex in all_examples:
            stats["examples_by_paradigm"][ex.true_paradigm.value] += 1
        
        # Calculate averages
        if all_examples:
            stats["avg_confidence"] = np.mean([ex.confidence_score for ex in all_examples])
            
            satisfaction_scores = [ex.user_satisfaction for ex in all_examples if ex.user_satisfaction]
            if satisfaction_scores:
                stats["avg_user_satisfaction"] = np.mean(satisfaction_scores)
        
        # Calculate per-paradigm accuracy
        for paradigm in HostParadigm:
            paradigm_examples = [
                ex for ex in all_examples 
                if ex.true_paradigm == paradigm
            ]
            if paradigm_examples:
                correct = sum(
                    1 for ex in paradigm_examples 
                    if ex.predicted_paradigm == ex.true_paradigm
                )
                stats["paradigm_accuracy"][paradigm.value] = correct / len(paradigm_examples)
        
        stats["examples_by_paradigm"] = dict(stats["examples_by_paradigm"])
        return stats


# Create singleton instance
ml_pipeline = MLPipeline()
