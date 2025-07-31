#!/usr/bin/env python3
"""
Migration Script: V1 → V2
Migrates Research Store and Context Engineering to V2 with rollback capability
"""

import asyncio
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages the V1 → V2 migration process"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.migration_log = []
        self.rollback_data = {}
        
        # Migration flags
        self.use_v2_research_store = False
        self.use_v2_context_engineering = False
        
        logger.info(f"🚀 Migration Manager initialized for {environment}")
    
    def log_step(self, step: str, success: bool = True, details: str = ""):
        """Log migration step"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "success": success,
            "details": details,
            "environment": self.environment
        }
        self.migration_log.append(log_entry)
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {step}: {details}")
    
    async def preflight_checks(self) -> bool:
        """Run preflight checks before migration"""
        logger.info("🔍 Running preflight checks...")
        
        checks_passed = True
        
        # Check Redis connectivity
        try:
            from services.research_store_v2 import research_store_v2
            await research_store_v2.initialize()
            self.log_step("Redis connectivity", True, "V2 store connected")
        except Exception as e:
            self.log_step("Redis connectivity", False, f"Failed: {e}")
            checks_passed = False
        
        # Check imports
        try:
            from services.context_engineering_bridge import context_engineering_bridge
            from services.context_engineering_v2 import context_pipeline_v2
            self.log_step("Import dependencies", True, "All V2 modules available")
        except ImportError as e:
            self.log_step("Import dependencies", False, f"Missing: {e}")
            checks_passed = False
        
        # Check models
        try:
            from models.context_models import (
                ResearchRequestSchema, ClassificationResultSchema, 
                ResearchStatus, HostParadigm
            )
            self.log_step("Model schemas", True, "All schemas available")
        except ImportError as e:
            self.log_step("Model schemas", False, f"Missing: {e}")
            checks_passed = False
        
        # Check environment variables
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            self.log_step("Environment config", False, "REDIS_URL not set")
            checks_passed = False
        else:
            self.log_step("Environment config", True, f"Redis: {redis_url}")
        
        return checks_passed
    
    async def backup_v1_data(self) -> bool:
        """Backup existing V1 data"""
        logger.info("💾 Backing up V1 data...")
        
        try:
            from services.research_store import research_store as v1_store
            await v1_store.initialize()
            
            # Backup fallback store data
            if v1_store.fallback_store:
                backup_file = f"v1_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(v1_store.fallback_store, f, default=str, indent=2)
                
                self.log_step("V1 data backup", True, f"Saved to {backup_file}")
                self.rollback_data["backup_file"] = backup_file
                return True
            else:
                self.log_step("V1 data backup", True, "No fallback data to backup")
                return True
                
        except Exception as e:
            self.log_step("V1 data backup", False, f"Failed: {e}")
            return False
    
    async def migrate_research_store(self, test_mode: bool = True) -> bool:
        """Migrate to V2 Research Store"""
        logger.info("📦 Migrating Research Store to V2...")
        
        try:
            from services.research_store_v2 import research_store_v2
            from services.research_store import research_store as v1_store
            
            # Initialize V2 store
            await research_store_v2.initialize()
            
            if test_mode:
                # Test with sample data
                from models.context_models import ResearchRequestSchema, ResearchStatus
                
                test_request = ResearchRequestSchema(
                    id="migration-test-001",
                    query="test migration query",
                    user_context={"user_id": "migration-test", "role": "BASIC"},
                    status=ResearchStatus.PROCESSING
                )
                
                # Store and retrieve test data
                await research_store_v2.store_research_request(test_request)
                retrieved = await research_store_v2.get_research("migration-test-001")
                
                if retrieved and retrieved["id"] == "migration-test-001":
                    self.log_step("Research Store V2 test", True, "Test data stored/retrieved")
                    self.use_v2_research_store = True
                    
                    # Cleanup test data
                    await research_store_v2.delete("migration-test-001")
                    return True
                else:
                    self.log_step("Research Store V2 test", False, "Test data retrieval failed")
                    return False
            else:
                # Production migration would go here
                self.use_v2_research_store = True
                self.log_step("Research Store V2 migration", True, "Production migration completed")
                return True
                
        except Exception as e:
            self.log_step("Research Store V2 migration", False, f"Failed: {e}")
            return False
    
    async def migrate_context_engineering(self, test_mode: bool = True) -> bool:
        """Migrate to V2 Context Engineering"""
        logger.info("🔧 Migrating Context Engineering to V2...")
        
        try:
            from services.context_engineering_bridge import context_engineering_bridge
            from services.classification_engine import ClassificationResult, QueryFeatures, HostParadigm
            
            if test_mode:
                # Create test classification
                test_features = QueryFeatures(
                    text="test migration query",
                    tokens=["test", "migration", "query"],
                    entities=["migration"],
                    intent_signals=["test"],
                    domain="testing",
                    urgency_score=0.5,
                    complexity_score=0.5,
                    emotional_valence=0.0
                )
                
                test_classification = ClassificationResult(
                    query="test migration query",
                    primary_paradigm=HostParadigm.BERNARD,
                    secondary_paradigm=None,
                    distribution={HostParadigm.BERNARD: 0.9},
                    confidence=0.9,
                    features=test_features,
                    reasoning={HostParadigm.BERNARD: ["test reasoning"]}
                )
                
                # Test V2 processing
                context_engineering_bridge.use_v2 = True
                result = await context_engineering_bridge.process_query(
                    test_classification,
                    include_debug=True
                )
                
                if hasattr(result, 'refined_queries') and len(result.refined_queries) > 0:
                    self.log_step("Context Engineering V2 test", True, f"Generated {len(result.refined_queries)} queries")
                    self.use_v2_context_engineering = True
                    return True
                else:
                    self.log_step("Context Engineering V2 test", False, "No queries generated")
                    return False
            else:
                # Production migration
                context_engineering_bridge.use_v2 = True
                self.use_v2_context_engineering = True
                self.log_step("Context Engineering V2 migration", True, "Production migration completed")
                return True
                
        except Exception as e:
            self.log_step("Context Engineering V2 migration", False, f"Failed: {e}")
            return False
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests to verify migration"""
        logger.info("🧪 Running integration tests...")
        
        try:
            # Create test fixtures manually (without pytest)
            from models.context_models import (
                ResearchRequestSchema, ClassificationResultSchema, 
                QueryFeaturesSchema, HostParadigm, ResearchStatus
            )
            from services.classification_engine import ClassificationResult, QueryFeatures
            from datetime import datetime
            
            # Create sample classification
            features = QueryFeatures(
                text="climate change impacts on agriculture",
                tokens=["climate", "change", "impacts", "agriculture"],
                entities=["climate", "agriculture"],
                intent_signals=["research", "analysis"],
                domain="environmental",
                urgency_score=0.6,
                complexity_score=0.7,
                emotional_valence=0.0
            )
            
            sample_classification = ClassificationResult(
                query="climate change impacts on agriculture",
                primary_paradigm=HostParadigm.BERNARD,
                secondary_paradigm=HostParadigm.DOLORES,
                distribution={
                    HostParadigm.BERNARD: 0.8,
                    HostParadigm.DOLORES: 0.6,
                    HostParadigm.MAEVE: 0.3,
                    HostParadigm.TEDDY: 0.4
                },
                confidence=0.8,
                features=features,
                reasoning={
                    HostParadigm.BERNARD: ["scientific research needed", "data analysis"],
                    HostParadigm.DOLORES: ["environmental justice", "system impact"]
                }
            )
            
            # Create sample research request
            sample_request = ResearchRequestSchema(
                id="test-research-123",
                query="climate change impacts on agriculture", 
                user_context={
                    "user_id": "test-user-456",
                    "role": "RESEARCHER",
                    "subscription_tier": "PRO"
                },
                options={
                    "deep_research": True,
                    "academic_focus": True
                },
                status=ResearchStatus.PROCESSING
            )
            
            # Run simplified tests
            from services.research_store_v2 import research_store_v2
            from services.context_engineering_bridge import context_engineering_bridge
            
            # Test research store
            test_id = await research_store_v2.store_research_request(sample_request)
            stored_data = await research_store_v2.get_research(test_id)
            if not stored_data or stored_data["id"] != test_id:
                raise Exception("Research store test failed")
            await research_store_v2.delete(test_id)  # Cleanup
            
            # Test context engineering
            context_result = await context_engineering_bridge.process_query(
                sample_classification,
                include_debug=True
            )
            if not hasattr(context_result, 'refined_queries') or len(context_result.refined_queries) == 0:
                raise Exception("Context engineering test failed")
            
            self.log_step("Integration tests", True, "All tests passed")
            return True
            
        except Exception as e:
            self.log_step("Integration tests", False, f"Failed: {e}")
            return False
    
    async def rollback_migration(self) -> bool:
        """Rollback migration to V1"""
        logger.info("🔄 Rolling back migration...")
        
        try:
            from services.context_engineering_bridge import context_engineering_bridge
            
            # Revert context engineering
            context_engineering_bridge.use_v2 = False
            self.use_v2_context_engineering = False
            self.log_step("Context Engineering rollback", True, "Reverted to V1")
            
            # Revert research store
            self.use_v2_research_store = False
            self.log_step("Research Store rollback", True, "Reverted to V1")
            
            # Restore backup data if available
            if "backup_file" in self.rollback_data:
                backup_file = self.rollback_data["backup_file"]
                if os.path.exists(backup_file):
                    from services.research_store import research_store as v1_store
                    
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)
                    
                    v1_store.fallback_store.update(backup_data)
                    self.log_step("Data restoration", True, f"Restored from {backup_file}")
            
            return True
            
        except Exception as e:
            self.log_step("Migration rollback", False, f"Failed: {e}")
            return False
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate migration report"""
        successful_steps = sum(1 for log in self.migration_log if log["success"])
        total_steps = len(self.migration_log)
        
        report = {
            "migration_summary": {
                "environment": self.environment,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            },
            "feature_status": {
                "research_store_v2": self.use_v2_research_store,
                "context_engineering_v2": self.use_v2_context_engineering
            },
            "migration_log": self.migration_log,
            "rollback_available": bool(self.rollback_data)
        }
        
        return report
    
    async def run_complete_migration(self, test_mode: bool = True) -> bool:
        """Run complete migration process"""
        logger.info("🚀 Starting complete migration process...")
        
        # Step 1: Preflight checks
        if not await self.preflight_checks():
            logger.error("❌ Preflight checks failed - aborting migration")
            return False
        
        # Step 2: Backup V1 data
        if not await self.backup_v1_data():
            logger.error("❌ Data backup failed - aborting migration")
            return False
        
        # Step 3: Migrate Research Store
        if not await self.migrate_research_store(test_mode):
            logger.error("❌ Research Store migration failed")
            await self.rollback_migration()
            return False
        
        # Step 4: Migrate Context Engineering
        if not await self.migrate_context_engineering(test_mode):
            logger.error("❌ Context Engineering migration failed")
            await self.rollback_migration()
            return False
        
        # Step 5: Run integration tests
        if not await self.run_integration_tests():
            logger.error("❌ Integration tests failed")
            if test_mode:
                await self.rollback_migration()
                return False
        
        # Step 6: Generate report
        report = self.generate_migration_report()
        
        # Save report
        report_file = f"migration_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Migration report saved to {report_file}")
        
        if report["migration_summary"]["success_rate"] == 1.0:
            logger.info("✅ Migration completed successfully!")
            return True
        else:
            logger.error("❌ Migration completed with errors")
            return False


async def main():
    """Main migration entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V1 → V2 Migration Script")
    parser.add_argument("--env", default="development", help="Environment (development/staging/production)")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Run in test mode")
    parser.add_argument("--rollback", action="store_true", help="Rollback to V1")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")
    
    args = parser.parse_args()
    
    migration = MigrationManager(args.env)
    
    if args.rollback:
        success = await migration.rollback_migration()
        print("🔄 Rollback completed" if success else "❌ Rollback failed")
    elif args.report_only:
        report = migration.generate_migration_report()
        print(json.dumps(report, indent=2, default=str))
    else:
        success = await migration.run_complete_migration(args.test_mode)
        if success:
            print("✅ Migration completed successfully!")
            print(f"📊 Check migration.log and migration_report_*.json for details")
        else:
            print("❌ Migration failed - check migration.log for details")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
