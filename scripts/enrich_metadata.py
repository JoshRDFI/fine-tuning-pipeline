#!/usr/bin/env python3
"""
Phase 2: Metadata Enrichment & Embedding Generation
Enrich assets with descriptions and generate vector embeddings.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import numpy as np

# Vision and embedding models
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image

# Data handling
import pandas as pd
import cv2

class MetadataEnricher:
    def __init__(self, extracted_assets_dir: str, use_gpu: bool = True):
        self.extracted_assets_dir = Path(extracted_assets_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Initialize models
        self.setup_models()
        
        # Setup logging
        self.setup_logging()
        
        # Track enrichment progress
        self.enrichment_log = []
        
    def setup_models(self):
        """Initialize vision and text models"""
        try:
            # CLIP model for image understanding and embeddings
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            # Sentence transformer for text embeddings
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_embedder.to(self.device)
            
            self.logger.info(f"Models loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.extracted_assets_dir / "enrichment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def enrich_all_assets(self) -> Dict:
        """Enrich all assets in the extracted directory"""
        self.logger.info("Starting metadata enrichment process")
        
        results = {
            "total_assets": 0,
            "enriched_assets": 0,
            "failed_assets": 0,
            "uncertain_items": [],
            "asset_types": {
                "text": 0,
                "images": 0,
                "tables": 0
            }
        }
        
        # Process each PDF directory
        for pdf_dir in self.extracted_assets_dir.iterdir():
            if pdf_dir.is_dir() and pdf_dir.name not in ["text", "images", "tables", "metadata"]:
                self.logger.info(f"Processing PDF directory: {pdf_dir.name}")
                pdf_results = self.enrich_pdf_assets(pdf_dir)
                
                # Aggregate results
                results["total_assets"] += pdf_results["total_assets"]
                results["enriched_assets"] += pdf_results["enriched_assets"]
                results["failed_assets"] += pdf_results["failed_assets"]
                results["uncertain_items"].extend(pdf_results["uncertain_items"])
                
                for asset_type in results["asset_types"]:
                    results["asset_types"][asset_type] += pdf_results["asset_types"][asset_type]
                    
        # Save enrichment summary
        self.save_enrichment_summary(results)
        return results
        
    def enrich_pdf_assets(self, pdf_dir: Path) -> Dict:
        """Enrich all assets for a single PDF"""
        results = {
            "total_assets": 0,
            "enriched_assets": 0,
            "failed_assets": 0,
            "uncertain_items": [],
            "asset_types": {"text": 0, "images": 0, "tables": 0}
        }
        
        # Load existing metadata
        metadata_file = pdf_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"pdf_name": pdf_dir.name}
            
        # Process text assets
        text_dir = pdf_dir / "text"
        if text_dir.exists():
            text_results = self.enrich_text_assets(text_dir)
            results["asset_types"]["text"] = text_results["count"]
            results["enriched_assets"] += text_results["enriched"]
            results["failed_assets"] += text_results["failed"]
            results["uncertain_items"].extend(text_results["uncertain_items"])
            
            # Update metadata with text information
            metadata["text_assets"] = text_results["assets_info"]
            
        # Process image assets
        images_dir = pdf_dir / "images"
        if images_dir.exists():
            image_results = self.enrich_image_assets(images_dir)
            results["asset_types"]["images"] = image_results["count"]
            results["enriched_assets"] += image_results["enriched"]
            results["failed_assets"] += image_results["failed"]
            results["uncertain_items"].extend(image_results["uncertain_items"])
            
            # Update metadata with image information
            metadata["image_assets"] = image_results["assets_info"]
            
        # Process table assets
        tables_dir = pdf_dir / "tables"
        if tables_dir.exists():
            table_results = self.enrich_table_assets(tables_dir)
            results["asset_types"]["tables"] = table_results["count"]
            results["enriched_assets"] += table_results["enriched"]
            results["failed_assets"] += table_results["failed"]
            results["uncertain_items"].extend(table_results["uncertain_items"])
            
            # Update metadata with table information
            metadata["table_assets"] = table_results["assets_info"]
            
        # Update enrichment metadata
        metadata["enrichment_date"] = datetime.now().isoformat()
        metadata["enrichment_results"] = results
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        results["total_assets"] = sum(results["asset_types"].values())
        return results
        
    def enrich_text_assets(self, text_dir: Path) -> Dict:
        """Enrich text assets with embeddings and descriptions"""
        result = {
            "count": 0,
            "enriched": 0,
            "failed": 0,
            "uncertain_items": [],
            "assets_info": []
        }
        
        for text_file in text_dir.glob("*.txt"):
            try:
                result["count"] += 1
                
                # Read text content
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if not content.strip():
                    result["uncertain_items"].append({
                        "type": "empty_text_file",
                        "file": str(text_file),
                        "message": "Text file is empty"
                    })
                    continue
                    
                # Generate text embedding
                embedding = self.generate_text_embedding(content)
                
                # Generate description (first 100 characters as summary)
                description = content[:100] + "..." if len(content) > 100 else content
                
                # Create asset info
                asset_info = {
                    "file_path": str(text_file),
                    "content_length": len(content),
                    "description": description,
                    "embedding": embedding.tolist(),
                    "enriched_date": datetime.now().isoformat()
                }
                
                # Save enriched metadata
                metadata_file = text_file.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(asset_info, f, indent=2)
                    
                result["assets_info"].append(asset_info)
                result["enriched"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to enrich text file {text_file}: {str(e)}")
                result["failed"] += 1
                result["uncertain_items"].append({
                    "type": "text_enrichment_error",
                    "file": str(text_file),
                    "message": str(e)
                })
                
        return result
        
    def enrich_image_assets(self, images_dir: Path) -> Dict:
        """Enrich image assets with descriptions and embeddings"""
        result = {
            "count": 0,
            "enriched": 0,
            "failed": 0,
            "uncertain_items": [],
            "assets_info": []
        }
        
        for image_file in images_dir.glob("*.png"):
            try:
                result["count"] += 1
                
                # Load and process image
                image = Image.open(image_file)
                
                # Generate image description using CLIP
                description = self.generate_image_description(image)
                
                # Generate image embedding
                embedding = self.generate_image_embedding(image)
                
                # Get image metadata
                width, height = image.size
                
                # Create asset info
                asset_info = {
                    "file_path": str(image_file),
                    "width": width,
                    "height": height,
                    "description": description,
                    "embedding": embedding.tolist(),
                    "enriched_date": datetime.now().isoformat()
                }
                
                # Save enriched metadata
                metadata_file = image_file.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(asset_info, f, indent=2)
                    
                result["assets_info"].append(asset_info)
                result["enriched"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to enrich image {image_file}: {str(e)}")
                result["failed"] += 1
                result["uncertain_items"].append({
                    "type": "image_enrichment_error",
                    "file": str(image_file),
                    "message": str(e)
                })
                
        return result
        
    def enrich_table_assets(self, tables_dir: Path) -> Dict:
        """Enrich table assets with descriptions and embeddings"""
        result = {
            "count": 0,
            "enriched": 0,
            "failed": 0,
            "uncertain_items": [],
            "assets_info": []
        }
        
        for table_file in tables_dir.glob("*.csv"):
            try:
                result["count"] += 1
                
                # Read table data
                df = pd.read_csv(table_file)
                
                # Generate table description
                description = self.generate_table_description(df)
                
                # Generate table embedding (using text representation)
                table_text = df.to_string()
                embedding = self.generate_text_embedding(table_text)
                
                # Get table metadata
                rows, cols = df.shape
                
                # Create asset info
                asset_info = {
                    "file_path": str(table_file),
                    "rows": rows,
                    "columns": cols,
                    "column_names": df.columns.tolist(),
                    "description": description,
                    "embedding": embedding.tolist(),
                    "enriched_date": datetime.now().isoformat()
                }
                
                # Save enriched metadata
                metadata_file = table_file.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(asset_info, f, indent=2)
                    
                result["assets_info"].append(asset_info)
                result["enriched"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to enrich table {table_file}: {str(e)}")
                result["failed"] += 1
                result["uncertain_items"].append({
                    "type": "table_enrichment_error",
                    "file": str(table_file),
                    "message": str(e)
                })
                
        return result
        
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text content"""
        try:
            embedding = self.text_embedder.encode(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate text embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(384)  # all-MiniLM-L6-v2 dimension
            
    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for image using CLIP"""
        try:
            # Preprocess image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Failed to generate image embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(512)  # CLIP dimension
            
    def generate_image_description(self, image: Image.Image) -> str:
        """Generate description for image using CLIP"""
        try:
            # Preprocess image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Define possible descriptions
            possible_descriptions = [
                "a chart or graph", "a diagram", "a photograph", "an illustration",
                "a table", "a logo", "a screenshot", "a drawing", "a figure",
                "a technical diagram", "a flow chart", "a bar chart", "a pie chart"
            ]
            
            # Encode descriptions
            text_inputs = self.clip_processor(text=possible_descriptions, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Get text features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                image_features = self.clip_model.get_image_features(**inputs)
                
            # Calculate similarities
            similarities = torch.cosine_similarity(image_features, text_features)
            best_match_idx = similarities.argmax().item()
            
            return possible_descriptions[best_match_idx]
            
        except Exception as e:
            self.logger.error(f"Failed to generate image description: {str(e)}")
            return "an image"
            
    def generate_table_description(self, df: pd.DataFrame) -> str:
        """Generate description for table"""
        try:
            rows, cols = df.shape
            
            # Analyze table content
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            description = f"Table with {rows} rows and {cols} columns"
            
            if numeric_cols:
                description += f", containing {len(numeric_cols)} numeric columns"
            if text_cols:
                description += f" and {len(text_cols)} text columns"
                
            return description
            
        except Exception as e:
            self.logger.error(f"Failed to generate table description: {str(e)}")
            return "a data table"
            
    def save_enrichment_summary(self, results: Dict):
        """Save enrichment summary report"""
        summary_file = self.extracted_assets_dir / "enrichment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create human-readable report
        txt_report = self.extracted_assets_dir / "enrichment_summary.txt"
        with open(txt_report, 'w') as f:
            f.write("Metadata Enrichment Summary Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total assets found: {results['total_assets']}\n")
            f.write(f"Successfully enriched: {results['enriched_assets']}\n")
            f.write(f"Failed to enrich: {results['failed_assets']}\n\n")
            f.write("Assets by type:\n")
            for asset_type, count in results["asset_types"].items():
                f.write(f"  - {asset_type}: {count}\n")
            f.write(f"\nUncertain items: {len(results['uncertain_items'])}\n")
            
            if results['uncertain_items']:
                f.write("\nUncertain Items:\n")
                for item in results['uncertain_items']:
                    f.write(f"  - {item}\n")

def main():
    parser = argparse.ArgumentParser(description="Enrich extracted assets with metadata and embeddings")
    parser.add_argument("--input_dir", default="extracted_assets", help="Directory containing extracted assets")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU acceleration if available")
    
    args = parser.parse_args()
    
    # Create enricher and run enrichment
    enricher = MetadataEnricher(args.input_dir, args.use_gpu)
    results = enricher.enrich_all_assets()
    
    print(f"\nEnrichment completed!")
    print(f"Total assets: {results['total_assets']}")
    print(f"Enriched: {results['enriched_assets']}")
    print(f"Failed: {results['failed_assets']}")
    print(f"Uncertain items: {len(results['uncertain_items'])}")
    print(f"Check {args.input_dir}/enrichment_summary.txt for details")

if __name__ == "__main__":
    main() 