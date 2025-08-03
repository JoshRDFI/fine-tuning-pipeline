#!/usr/bin/env python3
"""
Test script for Phase 1: Data Ingestion & Extraction
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract_pdfs import PDFExtractor

class TestPDFExtraction(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.sample_pdfs_dir = self.test_dir / "sample_pdfs"
        self.sample_pdfs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock PDF file for testing
        self.create_mock_pdf()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        
    def create_mock_pdf(self):
        """Create a mock PDF file for testing"""
        # This is a simplified mock - in real testing you'd use actual PDF files
        mock_pdf_path = self.sample_pdfs_dir / "test_document.pdf"
        
        # Create a simple text file that we'll pretend is a PDF for testing
        with open(mock_pdf_path, 'w') as f:
            f.write("Mock PDF content")
            
    def test_extractor_initialization(self):
        """Test PDFExtractor initialization"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Check that directories were created
        self.assertTrue(self.output_dir.exists())
        self.assertTrue((self.output_dir / "text").exists())
        self.assertTrue((self.output_dir / "images").exists())
        self.assertTrue((self.output_dir / "tables").exists())
        self.assertTrue((self.output_dir / "metadata").exists())
        
    def test_directory_processing(self):
        """Test processing a directory of PDFs"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Test with empty directory
        results = extractor.extract_from_directory(str(self.sample_pdfs_dir))
        self.assertEqual(results["total_files"], 1)  # Our mock PDF
        self.assertEqual(results["processed_files"], 0)  # Should fail to process mock
        self.assertEqual(results["failed_files"], 1)
        
    def test_output_structure(self):
        """Test that output structure is created correctly"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Verify directory structure
        expected_dirs = ["text", "images", "tables", "metadata"]
        for dir_name in expected_dirs:
            dir_path = self.output_dir / dir_name
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())
            
    def test_logging_setup(self):
        """Test that logging is properly configured"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Check that log file was created
        log_file = self.output_dir / "extraction.log"
        self.assertTrue(log_file.exists())
        
    def test_metadata_saving(self):
        """Test that metadata is saved correctly"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Create a mock result
        mock_result = {
            "pdf_name": "test",
            "text_sections": 5,
            "images": 3,
            "tables": 2,
            "uncertain_items": [],
            "metadata": {
                "source_path": "/path/to/test.pdf",
                "extraction_date": "2024-01-01T00:00:00",
                "file_size": 1024
            }
        }
        
        # Test saving summary report
        extractor.save_summary_report(mock_result)
        
        # Check that files were created
        summary_json = self.output_dir / "extraction_summary.json"
        summary_txt = self.output_dir / "extraction_summary.txt"
        
        self.assertTrue(summary_json.exists())
        self.assertTrue(summary_txt.exists())
        
        # Verify JSON content
        with open(summary_json, 'r') as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data["processed_files"], 0)
            self.assertEqual(saved_data["extracted_assets"]["text_sections"], 0)
            
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Test with non-existent directory
        with self.assertRaises(ValueError):
            extractor.extract_from_directory("/non/existent/path")
            
    def test_file_references(self):
        """Test that text files are updated with references to images and tables"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Create mock directories and files
        text_dir = self.output_dir / "test_pdf" / "text"
        images_dir = self.output_dir / "test_pdf" / "images"
        tables_dir = self.output_dir / "test_pdf" / "tables"
        
        for dir_path in [text_dir, images_dir, tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create mock files
        (text_dir / "page_001.txt").write_text("Sample text content")
        (images_dir / "page_001_img_001.png").write_text("mock image")
        (tables_dir / "table_001.csv").write_text("mock table")
        
        # Test reference updating
        extractor.update_text_with_references(text_dir, images_dir, tables_dir)
        
        # Check that text file was updated
        updated_content = (text_dir / "page_001.txt").read_text()
        self.assertIn("## Extracted Images", updated_content)
        self.assertIn("## Extracted Tables", updated_content)
        
    def test_extraction_logging(self):
        """Test that extraction process is properly logged"""
        extractor = PDFExtractor(str(self.output_dir))
        
        # Run extraction on empty directory
        results = extractor.extract_from_directory(str(self.sample_pdfs_dir))
        
        # Check that log file contains entries
        log_file = self.output_dir / "extraction.log"
        log_content = log_file.read_text()
        
        self.assertIn("Found", log_content)
        self.assertIn("PDF files to process", log_content)

def run_extraction_tests():
    """Run all extraction tests"""
    print("Running Phase 1 Extraction Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFExtraction)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
            
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_extraction_tests()
    sys.exit(0 if success else 1) 