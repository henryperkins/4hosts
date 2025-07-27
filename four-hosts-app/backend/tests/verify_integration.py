#!/usr/bin/env python3
"""
Integration verification script for the combined Four Hosts API
Tests component connections and data flow
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name: str, status: bool, message: str = ""):
    """Print test result with color"""
    icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    status_text = f"{GREEN}PASS{RESET}" if status else f"{RED}FAIL{RESET}"
    print(f"{icon} {name}: {status_text} {message}")

def print_section(title: str):
    """Print section header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

class IntegrationVerifier:
    def __init__(self):
        self.results = {
            "imports": [],
            "models": [],
            "services": [],
            "endpoints": [],
            "integration": []
        }
    
    def verify_imports(self):
        """Verify all necessary imports work"""
        print_section("1. IMPORT VERIFICATION")
        
        # Core imports
        core_imports = [
            ("FastAPI", "from fastapi import FastAPI"),
            ("Pydantic", "from pydantic import BaseModel"),
            ("Environment", "from dotenv import load_dotenv"),
            ("Async", "import asyncio"),
            ("UUID", "import uuid")
        ]
        
        for name, import_stmt in core_imports:
            try:
                exec(import_stmt)
                print_test(f"Core: {name}", True)
                self.results["imports"].append((name, True))
            except ImportError as e:
                print_test(f"Core: {name}", False, str(e))
                self.results["imports"].append((name, False))
        
        # Main module
        try:
            import main
            print_test("Main module", True)
            
            # Check feature flags
            print(f"\n  Feature Flags:")
            print(f"    - Production: {YELLOW}{main.PRODUCTION_FEATURES}{RESET}")
            print(f"    - Research: {YELLOW}{main.RESEARCH_FEATURES}{RESET}")
            print(f"    - AI: {YELLOW}{main.AI_FEATURES}{RESET}")
            print(f"    - Custom Docs: {YELLOW}{main.CUSTOM_DOCS}{RESET}")
            
        except Exception as e:
            print_test("Main module", False, str(e))
    
    def verify_data_models(self):
        """Verify data models are properly defined"""
        print_section("2. DATA MODEL VERIFICATION")
        
        try:
            from main import (
                Paradigm, ResearchDepth, ResearchOptions, 
                ResearchQuery, ParadigmClassification,
                ResearchStatus, SourceResult, ResearchResult
            )
            
            # Test enums
            paradigms = [p.value for p in Paradigm]
            assert paradigms == ["dolores", "teddy", "bernard", "maeve"]
            print_test("Paradigm enum", True, f"Values: {paradigms}")
            
            depths = [d.value for d in ResearchDepth]
            assert depths == ["quick", "standard", "deep"]
            print_test("ResearchDepth enum", True, f"Values: {depths}")
            
            # Test model creation
            query = ResearchQuery(query="Test query about climate change")
            assert query.query == "Test query about climate change"
            assert query.options.depth == ResearchDepth.STANDARD
            print_test("ResearchQuery model", True)
            
            # Test optional fields
            options = ResearchOptions(depth=ResearchDepth.DEEP, max_sources=100)
            assert options.max_sources == 100
            print_test("ResearchOptions model", True)
            
            self.results["models"].append(("All models", True))
            
        except Exception as e:
            print_test("Data models", False, str(e))
            self.results["models"].append(("Models", False))
    
    async def verify_services(self):
        """Verify service integrations"""
        print_section("3. SERVICE INTEGRATION VERIFICATION")
        
        try:
            import main
            
            # Test classification
            print("\n  Classification Service:")
            test_queries = [
                ("Expose corruption in government", "dolores"),
                ("Help vulnerable communities", "teddy"),
                ("Analyze statistical data", "bernard"),
                ("Optimize business strategy", "maeve")
            ]
            
            for query, expected in test_queries:
                try:
                    classification = await main.classify_query(query)
                    actual = classification.primary.value
                    matches = actual == expected
                    print_test(f"  Classify: {expected}", matches, 
                              f"'{query[:30]}...' → {actual}")
                    self.results["services"].append((f"classify_{expected}", matches))
                except Exception as e:
                    print_test(f"  Classify: {expected}", False, str(e))
                    self.results["services"].append((f"classify_{expected}", False))
            
            # Test helper functions
            print("\n  Helper Functions:")
            helpers = [
                ("generate_paradigm_queries", lambda: main.generate_paradigm_queries("test", "bernard")),
                ("get_paradigm_approach", lambda: main.get_paradigm_approach(main.Paradigm.DOLORES)),
                ("get_paradigm_focus", lambda: main.get_paradigm_focus(main.Paradigm.TEDDY)),
                ("generate_paradigm_summary", lambda: main.generate_paradigm_summary("test", main.Paradigm.MAEVE))
            ]
            
            for name, func in helpers:
                try:
                    result = func()
                    print_test(f"  {name}", True)
                    self.results["services"].append((name, True))
                except Exception as e:
                    print_test(f"  {name}", False, str(e))
                    self.results["services"].append((name, False))
            
        except Exception as e:
            print_test("Service verification", False, str(e))
    
    def verify_endpoints(self):
        """Verify endpoint definitions"""
        print_section("4. ENDPOINT VERIFICATION")
        
        try:
            from main import app
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, "path") and hasattr(route, "methods"):
                    routes.append((route.path, list(route.methods)))
            
            # Essential endpoints
            essential_endpoints = [
                ("/", ["GET"]),
                ("/health", ["GET"]),
                ("/paradigms/classify", ["POST"]),
                ("/research/query", ["POST"]),
                ("/research/status/{research_id}", ["GET"]),
                ("/research/results/{research_id}", ["GET"]),
                ("/system/stats", ["GET"])
            ]
            
            print(f"\n  Found {len(routes)} total endpoints")
            
            for path, methods in essential_endpoints:
                found = any(r[0] == path for r in routes)
                print_test(f"  {path}", found)
                self.results["endpoints"].append((path, found))
            
            # Production endpoints (conditional)
            import main
            if main.PRODUCTION_FEATURES:
                prod_endpoints = [
                    ("/auth/register", ["POST"]),
                    ("/auth/login", ["POST"]),
                    ("/metrics", ["GET"])
                ]
                print("\n  Production Endpoints:")
                for path, methods in prod_endpoints:
                    found = any(r[0] == path for r in routes)
                    print_test(f"  {path}", found)
            
            # Research endpoints (conditional)
            if main.RESEARCH_FEATURES:
                research_endpoints = [
                    ("/sources/credibility/{domain}", ["GET"])
                ]
                print("\n  Research Endpoints:")
                for path, methods in research_endpoints:
                    found = any(r[0] == path for r in routes)
                    print_test(f"  {path}", found)
            
        except Exception as e:
            print_test("Endpoint verification", False, str(e))
    
    async def verify_integration_flow(self):
        """Verify complete integration flow"""
        print_section("5. INTEGRATION FLOW VERIFICATION")
        
        try:
            import main
            
            # Test research flow
            print("\n  Research Flow Test:")
            
            # 1. Classification
            query = "How can we analyze climate change data?"
            classification = await main.classify_query(query)
            print_test("  1. Query classification", True, 
                      f"Primary: {classification.primary.value}")
            
            # 2. Generate paradigm queries
            queries = main.generate_paradigm_queries(query, classification.primary.value)
            print_test("  2. Generate search queries", len(queries) > 0,
                      f"Generated {len(queries)} queries")
            
            # 3. Mock research execution
            research_data = {
                "id": "test_123",
                "query": query,
                "options": {"depth": "standard", "max_sources": 50},
                "status": "processing",
                "paradigm_classification": classification.dict(),
                "created_at": datetime.utcnow().isoformat(),
                "results": None
            }
            main.research_store["test_123"] = research_data
            
            # Execute mock research
            await main.execute_mock_research("test_123", query, classification)
            
            # Check results
            result = main.research_store.get("test_123", {})
            success = result.get("status") == main.ResearchStatus.COMPLETED
            print_test("  3. Mock research execution", success)
            
            if success and result.get("results"):
                results = result["results"]
                print_test("  4. Results structure", 
                          "sources" in results and "answer" in results,
                          f"Sources: {len(results.get('sources', []))}")
            
            self.results["integration"].append(("research_flow", success))
            
        except Exception as e:
            print_test("Integration flow", False, str(e))
            self.results["integration"].append(("research_flow", False))
    
    def generate_report(self):
        """Generate final report"""
        print_section("INTEGRATION VERIFICATION REPORT")
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            if tests:
                category_passed = sum(1 for _, passed in tests if passed)
                category_total = len(tests)
                total_tests += category_total
                passed_tests += category_passed
                
                status = f"{GREEN}PASS{RESET}" if category_passed == category_total else f"{YELLOW}PARTIAL{RESET}"
                print(f"\n{category.upper()}: {status} ({category_passed}/{category_total})")
        
        # Overall status
        print(f"\n{BLUE}{'='*60}{RESET}")
        percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if percentage == 100:
            print(f"{GREEN}✓ ALL TESTS PASSED!{RESET} ({passed_tests}/{total_tests})")
            print(f"\nThe combined main.py is fully integrated and ready for use.")
        elif percentage >= 80:
            print(f"{YELLOW}⚠ MOSTLY INTEGRATED{RESET} ({passed_tests}/{total_tests} - {percentage:.1f}%)")
            print(f"\nThe core functionality works but some features may be limited.")
        else:
            print(f"{RED}✗ INTEGRATION ISSUES{RESET} ({passed_tests}/{total_tests} - {percentage:.1f}%)")
            print(f"\nSignificant integration problems detected. Review the failed tests.")
        
        # Recommendations
        print(f"\n{BLUE}Recommendations:{RESET}")
        if not all(r[1] for r in self.results.get("imports", [])):
            print("  - Install missing dependencies: pip install -r requirements.txt")
        if percentage < 100:
            print("  - Review failed tests and fix integration issues")
            print("  - Check service configurations in .env file")
        else:
            print("  - All components are properly integrated")
            print("  - Ready for deployment")

async def main():
    """Run integration verification"""
    print(f"{BLUE}Four Hosts API Integration Verification{RESET}")
    print(f"{BLUE}Testing combined main.py integration...{RESET}")
    
    verifier = IntegrationVerifier()
    
    # Run verifications
    verifier.verify_imports()
    verifier.verify_data_models()
    await verifier.verify_services()
    verifier.verify_endpoints()
    await verifier.verify_integration_flow()
    
    # Generate report
    verifier.generate_report()

if __name__ == "__main__":
    asyncio.run(main())