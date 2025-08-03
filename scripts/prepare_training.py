#!/usr/bin/env python3
"""
Phase 4: Training Data Preparation
Prepare dataset for multi-modal fine-tuning.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import random
import numpy as np
import pandas as pd

# Data handling
import torch
from PIL import Image

class TrainingDataPreparer:
    def __init__(self, extracted_assets_dir: str, output_dir: str = "training_data"):
        self.extracted_assets_dir = Path(extracted_assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training data formats
        self.supported_formats = ["jsonl", "csv", "huggingface"]
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "training_prep.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_dataset(self, 
                       output_format: str = "jsonl",
                       base_model: str = "llama2",
                       split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                       max_samples: Optional[int] = None) -> Dict:
        """Prepare training dataset from extracted assets"""
        
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}. Supported: {self.supported_formats}")
            
        self.logger.info(f"Preparing training dataset in {output_format} format")
        
        # Collect all training pairs
        training_pairs = self.collect_training_pairs()
        
        if max_samples and len(training_pairs) > max_samples:
            random.shuffle(training_pairs)
            training_pairs = training_pairs[:max_samples]
            
        self.logger.info(f"Collected {len(training_pairs)} training pairs")
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(training_pairs, split_ratio)
        
        # Format data according to output format
        results = {
            "total_pairs": len(training_pairs),
            "train_pairs": len(train_data),
            "val_pairs": len(val_data),
            "test_pairs": len(test_data),
            "output_files": []
        }
        
        if output_format == "jsonl":
            results["output_files"] = self.save_jsonl_format(train_data, val_data, test_data)
        elif output_format == "csv":
            results["output_files"] = self.save_csv_format(train_data, val_data, test_data)
        elif output_format == "huggingface":
            results["output_files"] = self.save_huggingface_format(train_data, val_data, test_data)
            
        # Save dataset metadata
        self.save_dataset_metadata(results, base_model, output_format)
        
        return results
        
    def collect_training_pairs(self) -> List[Dict]:
        """Collect training pairs from extracted assets"""
        training_pairs = []
        
        # Process each PDF directory
        for pdf_dir in self.extracted_assets_dir.iterdir():
            if pdf_dir.is_dir() and pdf_dir.name not in ["text", "images", "tables", "metadata"]:
                self.logger.info(f"Processing PDF directory: {pdf_dir.name}")
                pdf_pairs = self.collect_pdf_training_pairs(pdf_dir)
                training_pairs.extend(pdf_pairs)
                
        return training_pairs
        
    def collect_pdf_training_pairs(self, pdf_dir: Path) -> List[Dict]:
        """Collect training pairs from a single PDF"""
        pairs = []
        
        # Load PDF metadata
        metadata_file = pdf_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"pdf_name": pdf_dir.name}
            
        # Text-to-text pairs (from text chunks)
        text_pairs = self.create_text_pairs(pdf_dir, metadata)
        pairs.extend(text_pairs)
        
        # Text-to-image pairs
        image_pairs = self.create_image_pairs(pdf_dir, metadata)
        pairs.extend(image_pairs)
        
        # Text-to-table pairs
        table_pairs = self.create_table_pairs(pdf_dir, metadata)
        pairs.extend(table_pairs)
        
        # Multi-modal pairs (text + image + table)
        multimodal_pairs = self.create_multimodal_pairs(pdf_dir, metadata)
        pairs.extend(multimodal_pairs)
        
        return pairs
        
    def create_text_pairs(self, pdf_dir: Path, metadata: Dict) -> List[Dict]:
        """Create text-to-text training pairs"""
        pairs = []
        text_dir = pdf_dir / "text"
        
        if not text_dir.exists():
            return pairs
            
        # Get all text files
        text_files = list(text_dir.glob("*.txt"))
        
        for i, text_file in enumerate(text_files):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if not content.strip():
                    continue
                    
                # Create instruction-response pair
                instruction = f"Please summarize the following text from {metadata.get('pdf_name', 'the document')}:"
                response = content[:500] + "..." if len(content) > 500 else content
                
                pairs.append({
                    "type": "text_to_text",
                    "instruction": instruction,
                    "response": response,
                    "source_file": str(text_file),
                    "pdf_name": metadata.get("pdf_name", pdf_dir.name),
                    "metadata": {
                        "content_length": len(content),
                        "file_type": "text"
                    }
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process text file {text_file}: {str(e)}")
                
        return pairs
        
    def create_image_pairs(self, pdf_dir: Path, metadata: Dict) -> List[Dict]:
        """Create text-to-image training pairs"""
        pairs = []
        images_dir = pdf_dir / "images"
        text_dir = pdf_dir / "text"
        
        if not images_dir.exists():
            return pairs
            
        # Get all image files
        image_files = list(images_dir.glob("*.png"))
        
        for image_file in image_files:
            try:
                # Load image metadata
                metadata_file = image_file.with_suffix('.json')
                image_metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        image_metadata = json.load(f)
                        
                # Create instruction-response pair
                instruction = f"Describe the image from {metadata.get('pdf_name', 'the document')}:"
                response = image_metadata.get("description", "an image from the document")
                
                pairs.append({
                    "type": "text_to_image",
                    "instruction": instruction,
                    "response": response,
                    "image_path": str(image_file),
                    "source_file": str(image_file),
                    "pdf_name": metadata.get("pdf_name", pdf_dir.name),
                    "metadata": {
                        "image_width": image_metadata.get("width", 0),
                        "image_height": image_metadata.get("height", 0),
                        "file_type": "image"
                    }
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process image {image_file}: {str(e)}")
                
        return pairs
        
    def create_table_pairs(self, pdf_dir: Path, metadata: Dict) -> List[Dict]:
        """Create text-to-table training pairs"""
        pairs = []
        tables_dir = pdf_dir / "tables"
        
        if not tables_dir.exists():
            return pairs
            
        # Get all CSV files
        csv_files = list(tables_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                # Load table metadata
                metadata_file = csv_file.with_suffix('.json')
                table_metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        table_metadata = json.load(f)
                        
                # Read table data
                df = pd.read_csv(csv_file)
                
                # Create instruction-response pair
                instruction = f"Analyze the table from {metadata.get('pdf_name', 'the document')}:"
                response = table_metadata.get("description", f"Table with {len(df)} rows and {len(df.columns)} columns")
                
                pairs.append({
                    "type": "text_to_table",
                    "instruction": instruction,
                    "response": response,
                    "table_path": str(csv_file),
                    "source_file": str(csv_file),
                    "pdf_name": metadata.get("pdf_name", pdf_dir.name),
                    "metadata": {
                        "table_rows": len(df),
                        "table_columns": len(df.columns),
                        "column_names": df.columns.tolist(),
                        "file_type": "table"
                    }
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process table {csv_file}: {str(e)}")
                
        return pairs
        
    def create_multimodal_pairs(self, pdf_dir: Path, metadata: Dict) -> List[Dict]:
        """Create multi-modal training pairs"""
        pairs = []
        
        # Get all asset types
        text_dir = pdf_dir / "text"
        images_dir = pdf_dir / "images"
        tables_dir = pdf_dir / "tables"
        
        text_files = list(text_dir.glob("*.txt")) if text_dir.exists() else []
        image_files = list(images_dir.glob("*.png")) if images_dir.exists() else []
        table_files = list(tables_dir.glob("*.csv")) if tables_dir.exists() else []
        
        # Create multi-modal pairs when we have multiple asset types
        if len(text_files) > 0 and (len(image_files) > 0 or len(table_files) > 0):
            for text_file in text_files[:5]:  # Limit to avoid too many pairs
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        
                    if not text_content.strip():
                        continue
                        
                    # Create instruction with multiple modalities
                    instruction = f"Based on the text and visual content from {metadata.get('pdf_name', 'the document')}, provide a comprehensive analysis."
                    
                    # Combine text with image/table descriptions
                    response_parts = [text_content[:200] + "..." if len(text_content) > 200 else text_content]
                    
                    if image_files:
                        response_parts.append(f"The document contains {len(image_files)} images.")
                        
                    if table_files:
                        response_parts.append(f"The document contains {len(table_files)} data tables.")
                        
                    response = " ".join(response_parts)
                    
                    pairs.append({
                        "type": "multimodal",
                        "instruction": instruction,
                        "response": response,
                        "text_path": str(text_file),
                        "image_paths": [str(f) for f in image_files[:3]],  # Limit images
                        "table_paths": [str(f) for f in table_files[:2]],  # Limit tables
                        "source_file": str(text_file),
                        "pdf_name": metadata.get("pdf_name", pdf_dir.name),
                        "metadata": {
                            "text_length": len(text_content),
                            "num_images": len(image_files),
                            "num_tables": len(table_files),
                            "file_type": "multimodal"
                        }
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process multimodal pair for {text_file}: {str(e)}")
                    
        return pairs
        
    def split_dataset(self, data: List[Dict], split_ratio: Tuple[float, float, float]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets"""
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        self.logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
        
    def save_jsonl_format(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> List[str]:
        """Save data in JSONL format"""
        output_files = []
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_file = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                    
            output_files.append(str(output_file))
            self.logger.info(f"Saved {len(split_data)} items to {output_file}")
            
        return output_files
        
    def save_csv_format(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> List[str]:
        """Save data in CSV format"""
        output_files = []
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_file = self.output_dir / f"{split_name}.csv"
            
            # Convert to DataFrame
            df = pd.DataFrame(split_data)
            
            # Flatten metadata column
            if 'metadata' in df.columns:
                metadata_df = pd.json_normalize(df['metadata'])
                df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
                
            df.to_csv(output_file, index=False)
            output_files.append(str(output_file))
            self.logger.info(f"Saved {len(split_data)} items to {output_file}")
            
        return output_files
        
    def save_huggingface_format(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> List[str]:
        """Save data in HuggingFace dataset format"""
        try:
            from datasets import Dataset, DatasetDict
            
            output_files = []
            
            # Convert to HuggingFace format
            datasets = {}
            for split_name, split_data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
                if split_data:
                    # Convert to format expected by HuggingFace
                    hf_data = []
                    for item in split_data:
                        hf_item = {
                            "instruction": item["instruction"],
                            "response": item["response"],
                            "type": item["type"],
                            "source_file": item["source_file"],
                            "pdf_name": item["pdf_name"]
                        }
                        
                        # Add metadata as additional fields
                        if "metadata" in item:
                            for key, value in item["metadata"].items():
                                hf_item[f"metadata_{key}"] = value
                                
                        hf_data.append(hf_item)
                        
                    datasets[split_name] = Dataset.from_list(hf_data)
                    
            # Create dataset dictionary
            dataset_dict = DatasetDict(datasets)
            
            # Save to disk
            output_dir = self.output_dir / "huggingface_dataset"
            dataset_dict.save_to_disk(str(output_dir))
            
            output_files.append(str(output_dir))
            self.logger.info(f"Saved HuggingFace dataset to {output_dir}")
            
        except ImportError:
            self.logger.warning("HuggingFace datasets not available, falling back to JSONL format")
            return self.save_jsonl_format(train_data, val_data, test_data)
            
        return output_files
        
    def save_dataset_metadata(self, results: Dict, base_model: str, output_format: str):
        """Save dataset metadata"""
        metadata = {
            "dataset_info": {
                "base_model": base_model,
                "output_format": output_format,
                "creation_date": datetime.now().isoformat(),
                "total_pairs": results["total_pairs"],
                "train_pairs": results["train_pairs"],
                "val_pairs": results["val_pairs"],
                "test_pairs": results["test_pairs"]
            },
            "output_files": results["output_files"],
            "source_directory": str(self.extracted_assets_dir)
        }
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create human-readable summary
        summary_file = self.output_dir / "dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Training Dataset Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Base model: {base_model}\n")
            f.write(f"Output format: {output_format}\n")
            f.write(f"Creation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Dataset splits:\n")
            f.write(f"  - Training: {results['train_pairs']} pairs\n")
            f.write(f"  - Validation: {results['val_pairs']} pairs\n")
            f.write(f"  - Test: {results['test_pairs']} pairs\n")
            f.write(f"  - Total: {results['total_pairs']} pairs\n\n")
            f.write("Output files:\n")
            for file_path in results["output_files"]:
                f.write(f"  - {file_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset for multi-modal fine-tuning")
    parser.add_argument("--input_dir", default="extracted_assets", 
                       help="Directory containing extracted and enriched assets")
    parser.add_argument("--output_dir", default="training_data", 
                       help="Output directory for training data")
    parser.add_argument("--format", choices=["jsonl", "csv", "huggingface"], default="jsonl",
                       help="Output format for training data")
    parser.add_argument("--base_model", default="llama2", 
                       help="Base model for fine-tuning")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, 
                       help="Test set ratio")
    parser.add_argument("--max_samples", type=int, 
                       help="Maximum number of training pairs to generate")
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
    # Create preparer and run
    preparer = TrainingDataPreparer(args.input_dir, args.output_dir)
    results = preparer.prepare_dataset(
        output_format=args.format,
        base_model=args.base_model,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        max_samples=args.max_samples
    )
    
    print(f"\nTraining data preparation completed!")
    print(f"Total pairs: {results['total_pairs']}")
    print(f"Train: {results['train_pairs']}, Val: {results['val_pairs']}, Test: {results['test_pairs']}")
    print(f"Output format: {args.format}")
    print(f"Check {args.output_dir}/dataset_summary.txt for details")

if __name__ == "__main__":
    main() 