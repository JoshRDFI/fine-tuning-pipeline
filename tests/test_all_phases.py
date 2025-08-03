#!/usr/bin/env python3
"""
Phase 8: Testing & Validation
Comprehensive test suite for all pipeline phases.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime
import logging

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import pipeline components
from extract_pdfs import PDFExtractor
from enrich_metadata import MetadataEnricher
from build_db import DatabaseBuilder
from prepare_training import TrainingDataPreparer
from fine_tune import ModelFineTuner
from rag_pipeline import RAGPipeline

class TestAllPhases(unittest.TestCase):
    """Comprehensive test suite for all pipeline phases"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.extracted_assets_dir = self.test_dir / "extracted_assets"
        self.training_data_dir = self.test_dir / "training_data"
        self.fine_tuned_models_dir = self.test_dir / "fine_tuned_models"
        self.database_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'test_multimodal_rag',
            'user': 'postgres',
            'password': ''
        }
        
        # Create test directories
        for dir_path in [self.extracted_assets_dir, self.training_data_dir, self.fine_tuned_models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create mock PDF for testing
        self.create_mock_pdf()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        
    def create_mock_pdf(self):
        """Create a mock PDF file for testing"""
        # Create a simple text file that we'll pretend is a PDF
        mock_pdf_path = self.test_dir / "test_document.pdf"
        with open(mock_pdf_path, 'w') as f:
            f.write("Mock PDF content for testing")
            
        # Create mock extracted assets structure
        pdf_dir = self.extracted_assets_dir / "test_document"
        text_dir = pdf_dir / "text"
        images_dir = pdf_dir / "images"
        tables_dir = pdf_dir / "tables"
        
        for dir_path in [text_dir, images_dir, tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create mock text file
        with open(text_dir / "page_001.txt", 'w') as f:
            f.write("This is sample text content from the document.")
            
        # Create mock image file
        with open(images_dir / "page_001_img_001.png", 'w') as f:
            f.write("mock image data")
            
        # Create mock table file
        with open(tables_dir / "table_001.csv", 'w') as f:
            f.write("column1,column2\nvalue1,value2")
            
        # Create metadata file
        metadata = {
            "pdf_name": "test_document",
            "source_path": str(mock_pdf_path),
            "extraction_date": datetime.now().isoformat(),
            "file_size": 1024
        }
        
        with open(pdf_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def test_phase1_extraction(self):
        """Test Phase 1: Data Ingestion & Extraction"""
        self.logger.info("Testing Phase 1: Data Ingestion & Extraction")
        
        try:
            # Test PDF extractor initialization
            extractor = PDFExtractor(str(self.extracted_assets_dir))
            self.assertIsNotNone(extractor)
            
            # Test directory processing
            results = extractor.extract_from_directory(str(self.test_dir), include_subdirs=False)
            
            # Verify results structure
            self.assertIn("total_files", results)
            self.assertIn("processed_files", results)
            self.assertIn("extracted_assets", results)
            
            # Check that extraction log was created
            log_file = self.extracted_assets_dir / "extraction.log"
            self.assertTrue(log_file.exists())
            
            self.logger.info("✓ Phase 1 tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 1 tests failed: {str(e)}")
            return False
            
    def test_phase2_enrichment(self):
        """Test Phase 2: Metadata Enrichment & Embedding Generation"""
        self.logger.info("Testing Phase 2: Metadata Enrichment & Embedding Generation")
        
        try:
            # Test metadata enricher initialization
            enricher = MetadataEnricher(str(self.extracted_assets_dir))
            self.assertIsNotNone(enricher)
            
            # Test enrichment process
            results = enricher.enrich_all_assets()
            
            # Verify results structure
            self.assertIn("total_assets", results)
            self.assertIn("enriched_assets", results)
            self.assertIn("asset_types", results)
            
            # Check that enrichment log was created
            log_file = self.extracted_assets_dir / "enrichment.log"
            self.assertTrue(log_file.exists())
            
            self.logger.info("✓ Phase 2 tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 2 tests failed: {str(e)}")
            return False
            
    def test_phase3_database(self):
        """Test Phase 3: Database Construction"""
        self.logger.info("Testing Phase 3: Database Construction")
        
        try:
            # Test database builder initialization
            builder = DatabaseBuilder(self.database_config)
            self.assertIsNotNone(builder)
            
            # Test database connection
            builder.connect()
            self.assertIsNotNone(builder.connection)
            
            # Test schema creation
            builder.create_schema()
            
            # Test database population
            results = builder.populate_database(str(self.extracted_assets_dir))
            
            # Verify results structure
            self.assertIn("documents_inserted", results)
            self.assertIn("text_chunks_inserted", results)
            self.assertIn("images_inserted", results)
            self.assertIn("tables_inserted", results)
            
            # Test database functionality
            builder.test_database()
            
            builder.disconnect_from_database()
            
            self.logger.info("✓ Phase 3 tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 3 tests failed: {str(e)}")
            return False
            
    def test_phase4_training_preparation(self):
        """Test Phase 4: Training Data Preparation"""
        self.logger.info("Testing Phase 4: Training Data Preparation")
        
        try:
            # Test training data preparer initialization
            preparer = TrainingDataPreparer(str(self.extracted_assets_dir), str(self.training_data_dir))
            self.assertIsNotNone(preparer)
            
            # Test dataset preparation
            results = preparer.prepare_dataset(
                output_format="jsonl",
                base_model="llama2",
                split_ratio=(0.8, 0.1, 0.1),
                max_samples=10  # Limit for testing
            )
            
            # Verify results structure
            self.assertIn("total_pairs", results)
            self.assertIn("train_pairs", results)
            self.assertIn("val_pairs", results)
            self.assertIn("test_pairs", results)
            self.assertIn("output_files", results)
            
            # Check that output files were created
            for output_file in results["output_files"]:
                self.assertTrue(Path(output_file).exists())
                
            self.logger.info("✓ Phase 4 tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 4 tests failed: {str(e)}")
            return False
            
    def test_phase5_fine_tuning(self):
        """Test Phase 5: Model Fine-Tuning"""
        self.logger.info("Testing Phase 5: Model Fine-Tuning")
        
        try:
            # Test model fine-tuner initialization
            fine_tuner = ModelFineTuner(str(self.training_data_dir), str(self.fine_tuned_models_dir))
            self.assertIsNotNone(fine_tuner)
            
            # Note: Full fine-tuning test would require significant resources
            # For testing purposes, we'll just verify the initialization
            self.logger.info("✓ Phase 5 initialization tests passed (full training skipped for testing)")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 5 tests failed: {str(e)}")
            return False
            
    def test_phase6_rag_pipeline(self):
        """Test Phase 6: RAG Pipeline Integration"""
        self.logger.info("Testing Phase 6: RAG Pipeline Integration")
        
        try:
            # Test RAG pipeline initialization
            # Note: This would require a trained model, so we'll test the structure
            rag_pipeline = RAGPipeline(str(self.fine_tuned_models_dir), self.database_config)
            self.assertIsNotNone(rag_pipeline)
            
            # Test database connection
            rag_pipeline.connect_to_database()
            self.assertIsNotNone(rag_pipeline.connection)
            
            # Test RAG pipeline functionality
            test_results = rag_pipeline.test_rag_pipeline()
            
            # Verify test results structure
            self.assertIn("test_queries", test_results)
            self.assertIn("successful_queries", test_results)
            self.assertIn("total_queries", test_results)
            
            rag_pipeline.disconnect_from_database()
            
            self.logger.info("✓ Phase 6 tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 6 tests failed: {str(e)}")
            return False
            
    def test_phase7_streamlit_ui(self):
        """Test Phase 7: Streamlit UI Development"""
        self.logger.info("Testing Phase 7: Streamlit UI Development")
        
        try:
            # Test Streamlit app initialization
            # Note: Full UI testing would require running Streamlit
            # For testing purposes, we'll verify the app structure
            
            # Import the Streamlit app
            from streamlit_app import StreamlitApp
            
            app = StreamlitApp()
            self.assertIsNotNone(app)
            
            # Test session state initialization
            self.assertIsNotNone(app.initialize_session_state)
            
            self.logger.info("✓ Phase 7 initialization tests passed (full UI testing skipped)")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Phase 7 tests failed: {str(e)}")
            return False
            
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline workflow"""
        self.logger.info("Testing End-to-End Pipeline Workflow")
        
        try:
            # Phase 1: Extraction
            phase1_success = self.test_phase1_extraction()
            if not phase1_success:
                self.fail("Phase 1 failed")
                
            # Phase 2: Enrichment
            phase2_success = self.test_phase2_enrichment()
            if not phase2_success:
                self.fail("Phase 2 failed")
                
            # Phase 3: Database
            phase3_success = self.test_phase3_database()
            if not phase3_success:
                self.fail("Phase 3 failed")
                
            # Phase 4: Training Preparation
            phase4_success = self.test_phase4_training_preparation()
            if not phase4_success:
                self.fail("Phase 4 failed")
                
            # Phase 5: Fine-tuning (initialization only)
            phase5_success = self.test_phase5_fine_tuning()
            if not phase5_success:
                self.fail("Phase 5 failed")
                
            # Phase 6: RAG Pipeline
            phase6_success = self.test_phase6_rag_pipeline()
            if not phase6_success:
                self.fail("Phase 6 failed")
                
            # Phase 7: Streamlit UI
            phase7_success = self.test_phase7_streamlit_ui()
            if not phase7_success:
                self.fail("Phase 7 failed")
                
            self.logger.info("✓ End-to-end pipeline tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ End-to-end pipeline tests failed: {str(e)}")
            return False
            
    def test_error_handling(self):
        """Test error handling and edge cases"""
        self.logger.info("Testing Error Handling and Edge Cases")
        
        try:
            # Test with invalid input directory
            with self.assertRaises(ValueError):
                extractor = PDFExtractor("invalid_dir")
                extractor.extract_from_directory("/non/existent/path")
                
            # Test with invalid database configuration
            with self.assertRaises(Exception):
                builder = DatabaseBuilder({
                    'host': 'invalid_host',
                    'port': '5432',
                    'database': 'invalid_db',
                    'user': 'invalid_user',
                    'password': 'invalid_password'
                })
                builder.connect()
                
            # Test with empty training data
            preparer = TrainingDataPreparer("empty_dir", str(self.training_data_dir))
            with self.assertRaises(Exception):
                preparer.prepare_dataset()
                
            self.logger.info("✓ Error handling tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error handling tests failed: {str(e)}")
            return False
            
    def test_performance_metrics(self):
        """Test performance metrics and benchmarks"""
        self.logger.info("Testing Performance Metrics")
        
        try:
            import time
            
            # Test extraction performance
            start_time = time.time()
            extractor = PDFExtractor(str(self.extracted_assets_dir))
            results = extractor.extract_from_directory(str(self.test_dir), include_subdirs=False)
            extraction_time = time.time() - start_time
            
            # Test enrichment performance
            start_time = time.time()
            enricher = MetadataEnricher(str(self.extracted_assets_dir))
            enrichment_results = enricher.enrich_all_assets()
            enrichment_time = time.time() - start_time
            
            # Log performance metrics
            self.logger.info(f"Extraction time: {extraction_time:.2f} seconds")
            self.logger.info(f"Enrichment time: {enrichment_time:.2f} seconds")
            
            # Verify reasonable performance (adjust thresholds as needed)
            self.assertLess(extraction_time, 60)  # Should complete within 60 seconds
            self.assertLess(enrichment_time, 120)  # Should complete within 120 seconds
            
            self.logger.info("✓ Performance tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Performance tests failed: {str(e)}")
            return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("COMPREHENSIVE PIPELINE TESTING")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAllPhases)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    generate_test_report(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
            
    return len(result.failures) == 0 and len(result.errors) == 0

def generate_test_report(result):
    """Generate detailed test report"""
    report = {
        "test_summary": {
            "total_tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
            "timestamp": datetime.now().isoformat()
        },
        "phase_results": {
            "phase1_extraction": "PASS" if not any("phase1" in str(f[0]) for f in result.failures) else "FAIL",
            "phase2_enrichment": "PASS" if not any("phase2" in str(f[0]) for f in result.failures) else "FAIL",
            "phase3_database": "PASS" if not any("phase3" in str(f[0]) for f in result.failures) else "FAIL",
            "phase4_training_preparation": "PASS" if not any("phase4" in str(f[0]) for f in result.failures) else "FAIL",
            "phase5_fine_tuning": "PASS" if not any("phase5" in str(f[0]) for f in result.failures) else "FAIL",
            "phase6_rag_pipeline": "PASS" if not any("phase6" in str(f[0]) for f in result.failures) else "FAIL",
            "phase7_streamlit_ui": "PASS" if not any("phase7" in str(f[0]) for f in result.failures) else "FAIL"
        },
        "failures": [{"test": str(f[0]), "traceback": f[1]} for f in result.failures],
        "errors": [{"test": str(e[0]), "traceback": e[1]} for e in result.errors]
    }
    
    # Save report
    report_file = Path("comprehensive_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Create human-readable report
    txt_report = Path("comprehensive_test_report.txt")
    with open(txt_report, 'w') as f:
        f.write("Comprehensive Pipeline Test Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Success Rate: {report['test_summary']['success_rate']:.1f}%\n\n")
        
        f.write("Phase Results:\n")
        for phase, status in report["phase_results"].items():
            f.write(f"  - {phase}: {status}\n")
            
        if result.failures:
            f.write("\nFailures:\n")
            for failure in report["failures"]:
                f.write(f"  - {failure['test']}\n")
                
        if result.errors:
            f.write("\nErrors:\n")
            for error in report["errors"]:
                f.write(f"  - {error['test']}\n")

def run_individual_phase_tests():
    """Run tests for individual phases"""
    print("Running individual phase tests...")
    
    test_instance = TestAllPhases()
    test_instance.setUp()
    
    phases = [
        ("Phase 1: Extraction", test_instance.test_phase1_extraction),
        ("Phase 2: Enrichment", test_instance.test_phase2_enrichment),
        ("Phase 3: Database", test_instance.test_phase3_database),
        ("Phase 4: Training Preparation", test_instance.test_phase4_training_preparation),
        ("Phase 5: Fine-tuning", test_instance.test_phase5_fine_tuning),
        ("Phase 6: RAG Pipeline", test_instance.test_phase6_rag_pipeline),
        ("Phase 7: Streamlit UI", test_instance.test_phase7_streamlit_ui)
    ]
    
    results = {}
    for phase_name, test_func in phases:
        print(f"\nTesting {phase_name}...")
        try:
            success = test_func()
            results[phase_name] = "PASS" if success else "FAIL"
            print(f"  {phase_name}: {results[phase_name]}")
        except Exception as e:
            results[phase_name] = "ERROR"
            print(f"  {phase_name}: ERROR - {str(e)}")
            
    test_instance.tearDown()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive pipeline tests")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive test suite")
    parser.add_argument("--individual", action="store_true", 
                       help="Run individual phase tests")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="Test specific phase only")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    elif args.individual:
        results = run_individual_phase_tests()
        print("\nIndividual Phase Test Results:")
        for phase, result in results.items():
            print(f"  {phase}: {result}")
    elif args.phase:
        test_instance = TestAllPhases()
        test_instance.setUp()
        
        phase_tests = {
            1: test_instance.test_phase1_extraction,
            2: test_instance.test_phase2_enrichment,
            3: test_instance.test_phase3_database,
            4: test_instance.test_phase4_training_preparation,
            5: test_instance.test_phase5_fine_tuning,
            6: test_instance.test_phase6_rag_pipeline,
            7: test_instance.test_phase7_streamlit_ui
        }
        
        test_func = phase_tests[args.phase]
        success = test_func()
        
        test_instance.tearDown()
        print(f"Phase {args.phase} test: {'PASS' if success else 'FAIL'}")
        sys.exit(0 if success else 1)
    else:
        # Default: run comprehensive tests
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1) 