#!/usr/bin/env python3
"""
Phase 1: Data Ingestion & Extraction
Extract all text, images, and tables from PDFs, preserving structure and references.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

# PDF processing libraries
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import camelot
import tabula

# Image processing
import cv2
import numpy as np

# Data handling
import pandas as pd

class PDFExtractor:
    def __init__(self, output_dir: str, use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.extraction_log = []
        
        # Create output directories
        self.text_dir = self.output_dir / "text"
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.text_dir, self.images_dir, self.tables_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "extraction.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_from_directory(self, input_dir: str, include_subdirs: bool = True) -> Dict:
        """Extract from all PDFs in a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
            
        pdf_files = []
        if include_subdirs:
            pdf_files = list(input_path.rglob("*.pdf"))
        else:
            pdf_files = list(input_path.glob("*.pdf"))
            
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            "total_files": len(pdf_files),
            "processed_files": 0,
            "failed_files": 0,
            "extracted_assets": {
                "text_sections": 0,
                "images": 0,
                "tables": 0
            },
            "uncertain_items": []
        }
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing: {pdf_file}")
                file_result = self.extract_single_pdf(pdf_file)
                results["processed_files"] += 1
                results["extracted_assets"]["text_sections"] += file_result["text_sections"]
                results["extracted_assets"]["images"] += file_result["images"]
                results["extracted_assets"]["tables"] += file_result["tables"]
                results["uncertain_items"].extend(file_result["uncertain_items"])
                
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {str(e)}")
                results["failed_files"] += 1
                
        # Save summary report
        self.save_summary_report(results)
        return results
    
    def extract_single_pdf(self, pdf_path: Path) -> Dict:
        """Extract all content from a single PDF"""
        pdf_name = pdf_path.stem
        self.logger.info(f"Starting extraction for {pdf_name}")
        
        result = {
            "pdf_name": pdf_name,
            "text_sections": 0,
            "images": 0,
            "tables": 0,
            "uncertain_items": [],
            "metadata": {
                "source_path": str(pdf_path),
                "extraction_date": datetime.now().isoformat(),
                "file_size": pdf_path.stat().st_size
            }
        }
        
        # Create PDF-specific directories
        pdf_output_dir = self.output_dir / pdf_name
        pdf_text_dir = pdf_output_dir / "text"
        pdf_images_dir = pdf_output_dir / "images"
        pdf_tables_dir = pdf_output_dir / "tables"
        
        for dir_path in [pdf_text_dir, pdf_images_dir, pdf_tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract text with structure
            text_result = self.extract_text_with_structure(pdf_path, pdf_text_dir)
            result["text_sections"] = text_result["sections"]
            result["uncertain_items"].extend(text_result["uncertain_items"])
            
            # Extract images
            images_result = self.extract_images(pdf_path, pdf_images_dir)
            result["images"] = images_result["count"]
            result["uncertain_items"].extend(images_result["uncertain_items"])
            
            # Extract tables
            tables_result = self.extract_tables(pdf_path, pdf_tables_dir)
            result["tables"] = tables_result["count"]
            result["uncertain_items"].extend(tables_result["uncertain_items"])
            
            # Update text with references to images and tables
            self.update_text_with_references(pdf_text_dir, pdf_images_dir, pdf_tables_dir)
            
            # Save metadata
            metadata_file = pdf_output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Completed extraction for {pdf_name}")
            
        except Exception as e:
            self.logger.error(f"Error extracting {pdf_name}: {str(e)}")
            result["uncertain_items"].append({
                "type": "extraction_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
        return result
    
    def extract_text_with_structure(self, pdf_path: Path, output_dir: Path) -> Dict:
        """Extract text with headings, paragraphs, and structure"""
        result = {"sections": 0, "uncertain_items": []}
        
        try:
            doc = fitz.open(pdf_path)
            current_section = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text blocks with their properties
                blocks = page.get_text("dict")["blocks"]
                
                page_text = []
                for block in blocks:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Determine if this is a heading based on font size
                                    font_size = span["size"]
                                    if font_size > 14:  # Likely a heading
                                        current_section += 1
                                        section_file = output_dir / f"section_{current_section:03d}.md"
                                        with open(section_file, 'w', encoding='utf-8') as f:
                                            f.write(f"# {text}\n\n")
                                        result["sections"] += 1
                                    else:
                                        page_text.append(text)
                
                # Save page text
                if page_text:
                    page_file = output_dir / f"page_{page_num + 1:03d}.txt"
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(page_text))
                        
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            result["uncertain_items"].append({
                "type": "text_extraction_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
        return result
    
    def extract_images(self, pdf_path: Path, output_dir: Path) -> Dict:
        """Extract images from PDF"""
        result = {"count": 0, "uncertain_items": []}
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get image list
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:03d}.png"
                            img_path = output_dir / img_filename
                            pix.save(str(img_path))
                            result["count"] += 1
                            
                        pix = None
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {str(e)}")
                        result["uncertain_items"].append({
                            "type": "image_extraction_error",
                            "page": page_num + 1,
                            "image_index": img_index,
                            "message": str(e)
                        })
                        
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
            result["uncertain_items"].append({
                "type": "image_extraction_error",
                "message": str(e)
            })
            
        return result
    
    def extract_tables(self, pdf_path: Path, output_dir: Path) -> Dict:
        """Extract tables from PDF using multiple methods"""
        result = {"count": 0, "uncertain_items": []}
        
        try:
            # Method 1: Using camelot
            try:
                tables = camelot.read_pdf(str(pdf_path), pages='all')
                for i, table in enumerate(tables):
                    if table.parsing_report['accuracy'] > 80:  # Only keep accurate tables
                        # Save as CSV
                        csv_filename = f"table_{i + 1:03d}.csv"
                        csv_path = output_dir / csv_filename
                        table.df.to_csv(csv_path, index=False)
                        
                        # Save as JSON
                        json_filename = f"table_{i + 1:03d}.json"
                        json_path = output_dir / json_filename
                        table.df.to_json(json_path, orient='records', indent=2)
                        
                        result["count"] += 1
                    else:
                        result["uncertain_items"].append({
                            "type": "table_accuracy_low",
                            "table_index": i,
                            "accuracy": table.parsing_report['accuracy']
                        })
                        
            except Exception as e:
                self.logger.warning(f"Camelot table extraction failed: {str(e)}")
                result["uncertain_items"].append({
                    "type": "camelot_extraction_error",
                    "message": str(e)
                })
            
            # Method 2: Using tabula-py as backup
            try:
                tables = tabula.read_pdf(str(pdf_path), pages='all')
                for i, table in enumerate(tables):
                    if not table.empty:
                        # Save as CSV
                        csv_filename = f"tabula_table_{i + 1:03d}.csv"
                        csv_path = output_dir / csv_filename
                        table.to_csv(csv_path, index=False)
                        
                        # Save as JSON
                        json_filename = f"tabula_table_{i + 1:03d}.json"
                        json_path = output_dir / json_filename
                        table.to_json(json_path, orient='records', indent=2)
                        
                        result["count"] += 1
                        
            except Exception as e:
                self.logger.warning(f"Tabula table extraction failed: {str(e)}")
                result["uncertain_items"].append({
                    "type": "tabula_extraction_error",
                    "message": str(e)
                })
                
        except Exception as e:
            self.logger.error(f"Error extracting tables from {pdf_path}: {str(e)}")
            result["uncertain_items"].append({
                "type": "table_extraction_error",
                "message": str(e)
            })
            
        return result
    
    def update_text_with_references(self, text_dir: Path, images_dir: Path, tables_dir: Path):
        """Update text files with references to extracted images and tables"""
        try:
            # Get list of extracted assets
            images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            tables = list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.json"))
            
            # Update text files with references
            for text_file in text_dir.glob("*.txt"):
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add references section
                references = []
                if images:
                    references.append("\n## Extracted Images\n")
                    for img in images:
                        references.append(f"![{img.stem}]({img.relative_to(text_dir)})\n")
                
                if tables:
                    references.append("\n## Extracted Tables\n")
                    for table in tables:
                        references.append(f"[Table: {table.stem}]({table.relative_to(text_dir)})\n")
                
                if references:
                    with open(text_file, 'a', encoding='utf-8') as f:
                        f.write('\n'.join(references))
                        
        except Exception as e:
            self.logger.error(f"Error updating text with references: {str(e)}")
    
    def save_summary_report(self, results: Dict):
        """Save a summary report of the extraction process"""
        report_file = self.output_dir / "extraction_summary.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Also create a human-readable report
        txt_report = self.output_dir / "extraction_summary.txt"
        with open(txt_report, 'w') as f:
            f.write("PDF Extraction Summary Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total PDF files found: {results['total_files']}\n")
            f.write(f"Successfully processed: {results['processed_files']}\n")
            f.write(f"Failed to process: {results['failed_files']}\n\n")
            f.write("Extracted Assets:\n")
            f.write(f"  - Text sections: {results['extracted_assets']['text_sections']}\n")
            f.write(f"  - Images: {results['extracted_assets']['images']}\n")
            f.write(f"  - Tables: {results['extracted_assets']['tables']}\n\n")
            f.write(f"Uncertain items: {len(results['uncertain_items'])}\n")
            
            if results['uncertain_items']:
                f.write("\nUncertain Items:\n")
                for item in results['uncertain_items']:
                    f.write(f"  - {item}\n")

def main():
    parser = argparse.ArgumentParser(description="Extract text, images, and tables from PDFs")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", default="extracted_assets", help="Output directory for extracted assets")
    parser.add_argument("--no_subdirs", action="store_true", help="Don't process subdirectories")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU acceleration if available")
    
    args = parser.parse_args()
    
    # Create extractor and run extraction
    extractor = PDFExtractor(args.output_dir, args.use_gpu)
    results = extractor.extract_from_directory(args.input_dir, not args.no_subdirs)
    
    print(f"\nExtraction completed!")
    print(f"Processed: {results['processed_files']}/{results['total_files']} files")
    print(f"Extracted: {results['extracted_assets']['text_sections']} text sections, "
          f"{results['extracted_assets']['images']} images, "
          f"{results['extracted_assets']['tables']} tables")
    print(f"Uncertain items: {len(results['uncertain_items'])}")
    print(f"Check {args.output_dir}/extraction_summary.txt for details")

if __name__ == "__main__":
    main() 