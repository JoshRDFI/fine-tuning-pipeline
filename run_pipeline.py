#!/usr/bin/env python3
"""
Multi-Modal RAG + Training Pipeline - Main Orchestration Script
Run the complete pipeline or individual phases.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import pipeline components
from extract_pdfs import PDFExtractor
from enrich_metadata import MetadataEnricher
from build_db import DatabaseBuilder
from prepare_training import TrainingDataPreparer
from fine_tune import ModelFineTuner
from rag_pipeline import RAGPipeline

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_phase1_extraction(input_dir, output_dir, include_subdirs=True, use_gpu=True):
    """Run Phase 1: Data Ingestion & Extraction"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 1: Data Ingestion & Extraction")
    
    try:
        extractor = PDFExtractor(output_dir, use_gpu)
        results = extractor.extract_from_directory(input_dir, include_subdirs)
        
        logger.info(f"Phase 1 completed successfully!")
        logger.info(f"Processed: {results['processed_files']}/{results['total_files']} files")
        logger.info(f"Extracted: {results['extracted_assets']['text_sections']} text sections, "
                   f"{results['extracted_assets']['images']} images, "
                   f"{results['extracted_assets']['tables']} tables")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {str(e)}")
        raise

def run_phase2_enrichment(extracted_assets_dir, use_gpu=True):
    """Run Phase 2: Metadata Enrichment & Embedding Generation"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 2: Metadata Enrichment & Embedding Generation")
    
    try:
        enricher = MetadataEnricher(extracted_assets_dir, use_gpu)
        results = enricher.enrich_all_assets()
        
        logger.info(f"Phase 2 completed successfully!")
        logger.info(f"Total assets: {results['total_assets']}")
        logger.info(f"Enriched: {results['enriched_assets']}")
        logger.info(f"Failed: {results['failed_assets']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}")
        raise

def run_phase3_database(extracted_assets_dir, db_config):
    """Run Phase 3: Database Construction"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 3: Database Construction")
    
    try:
        builder = DatabaseBuilder(db_config)
        builder.connect()
        builder.create_schema()
        results = builder.populate_database(extracted_assets_dir)
        builder.test_database()
        builder.disconnect_from_database()
        
        logger.info(f"Phase 3 completed successfully!")
        logger.info(f"Documents: {results['documents_inserted']}")
        logger.info(f"Text chunks: {results['text_chunks_inserted']}")
        logger.info(f"Images: {results['images_inserted']}")
        logger.info(f"Tables: {results['tables_inserted']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {str(e)}")
        raise

def run_phase4_training_preparation(extracted_assets_dir, training_data_dir, 
                                   output_format="jsonl", base_model="llama2"):
    """Run Phase 4: Training Data Preparation"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 4: Training Data Preparation")
    
    try:
        preparer = TrainingDataPreparer(extracted_assets_dir, training_data_dir)
        results = preparer.prepare_dataset(
            output_format=output_format,
            base_model=base_model,
            split_ratio=(0.8, 0.1, 0.1)
        )
        
        logger.info(f"Phase 4 completed successfully!")
        logger.info(f"Total pairs: {results['total_pairs']}")
        logger.info(f"Train: {results['train_pairs']}, Val: {results['val_pairs']}, Test: {results['test_pairs']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 4 failed: {str(e)}")
        raise

def run_phase5_fine_tuning(training_data_dir, output_dir, base_model="llama2", 
                          epochs=3, batch_size=4, learning_rate=2e-5):
    """Run Phase 5: Model Fine-Tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 5: Model Fine-Tuning")
    
    try:
        fine_tuner = ModelFineTuner(training_data_dir, output_dir)
        results = fine_tuner.fine_tune_model(
            base_model=base_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        logger.info(f"Phase 5 completed successfully!")
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Training loss: {results['training_results']['train_loss']:.4f}")
        logger.info(f"Evaluation loss: {results['training_results']['eval_loss']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 5 failed: {str(e)}")
        raise

def run_phase6_rag_pipeline(model_path, db_config):
    """Run Phase 6: RAG Pipeline Integration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 6: RAG Pipeline Integration")
    
    try:
        rag_pipeline = RAGPipeline(model_path, db_config)
        test_results = rag_pipeline.test_rag_pipeline()
        rag_pipeline.save_rag_summary(test_results)
        rag_pipeline.disconnect_from_database()
        
        logger.info(f"Phase 6 completed successfully!")
        logger.info(f"Successful queries: {test_results['successful_queries']}/{test_results['total_queries']}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Phase 6 failed: {str(e)}")
        raise

def run_complete_pipeline(config):
    """Run the complete pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete pipeline execution")
    
    pipeline_results = {
        "start_time": datetime.now().isoformat(),
        "phases": {},
        "overall_status": "running"
    }
    
    try:
        # Phase 1: Extraction
        logger.info("=" * 50)
        logger.info("PHASE 1: Data Ingestion & Extraction")
        logger.info("=" * 50)
        phase1_results = run_phase1_extraction(
            config["input_dir"],
            config["extracted_assets_dir"],
            config.get("include_subdirs", True),
            config.get("use_gpu", True)
        )
        pipeline_results["phases"]["phase1"] = {"status": "completed", "results": phase1_results}
        
        # Phase 2: Enrichment
        logger.info("=" * 50)
        logger.info("PHASE 2: Metadata Enrichment & Embedding Generation")
        logger.info("=" * 50)
        phase2_results = run_phase2_enrichment(
            config["extracted_assets_dir"],
            config.get("use_gpu", True)
        )
        pipeline_results["phases"]["phase2"] = {"status": "completed", "results": phase2_results}
        
        # Phase 3: Database
        logger.info("=" * 50)
        logger.info("PHASE 3: Database Construction")
        logger.info("=" * 50)
        phase3_results = run_phase3_database(
            config["extracted_assets_dir"],
            config["database_config"]
        )
        pipeline_results["phases"]["phase3"] = {"status": "completed", "results": phase3_results}
        
        # Phase 4: Training Preparation
        logger.info("=" * 50)
        logger.info("PHASE 4: Training Data Preparation")
        logger.info("=" * 50)
        phase4_results = run_phase4_training_preparation(
            config["extracted_assets_dir"],
            config["training_data_dir"],
            config.get("output_format", "jsonl"),
            config.get("base_model", "llama2")
        )
        pipeline_results["phases"]["phase4"] = {"status": "completed", "results": phase4_results}
        
        # Phase 5: Fine-tuning
        logger.info("=" * 50)
        logger.info("PHASE 5: Model Fine-Tuning")
        logger.info("=" * 50)
        phase5_results = run_phase5_fine_tuning(
            config["training_data_dir"],
            config["fine_tuned_models_dir"],
            config.get("base_model", "llama2"),
            config.get("epochs", 3),
            config.get("batch_size", 4),
            config.get("learning_rate", 2e-5)
        )
        pipeline_results["phases"]["phase5"] = {"status": "completed", "results": phase5_results}
        
        # Phase 6: RAG Pipeline
        logger.info("=" * 50)
        logger.info("PHASE 6: RAG Pipeline Integration")
        logger.info("=" * 50)
        phase6_results = run_phase6_rag_pipeline(
            phase5_results["model_path"],
            config["database_config"]
        )
        pipeline_results["phases"]["phase6"] = {"status": "completed", "results": phase6_results}
        
        # Pipeline completed successfully
        pipeline_results["overall_status"] = "completed"
        pipeline_results["end_time"] = datetime.now().isoformat()
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        # Save pipeline results
        with open("pipeline_results.json", "w") as f:
            json.dump(pipeline_results, f, indent=2)
            
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        pipeline_results["overall_status"] = "failed"
        pipeline_results["error"] = str(e)
        pipeline_results["end_time"] = datetime.now().isoformat()
        
        # Save pipeline results
        with open("pipeline_results.json", "w") as f:
            json.dump(pipeline_results, f, indent=2)
            
        raise

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal RAG + Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], 
                       help="Run specific phase only")
    parser.add_argument("--input_dir", type=str, help="Input directory with PDFs")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", 
                       help="Output directory for pipeline results")
    parser.add_argument("--base_model", type=str, default="llama2", 
                       help="Base model for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--include_subdirs", action="store_true", default=True, 
                       help="Include subdirectories in extraction")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "input_dir": args.input_dir or "data",
            "extracted_assets_dir": f"{args.output_dir}/extracted_assets",
            "training_data_dir": f"{args.output_dir}/training_data",
            "fine_tuned_models_dir": f"{args.output_dir}/fine_tuned_models",
            "database_config": {
                'host': 'localhost',
                'port': '5432',
                'database': 'multimodal_rag',
                'user': 'postgres',
                'password': ''
            },
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_gpu": args.use_gpu,
            "include_subdirs": args.include_subdirs,
            "output_format": "jsonl"
        }
    
    # Create output directories
    for dir_path in [config["extracted_assets_dir"], config["training_data_dir"], config["fine_tuned_models_dir"]]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.phase:
            # Run specific phase
            logger.info(f"Running Phase {args.phase} only")
            
            if args.phase == 1:
                results = run_phase1_extraction(
                    config["input_dir"],
                    config["extracted_assets_dir"],
                    config["include_subdirs"],
                    config["use_gpu"]
                )
            elif args.phase == 2:
                results = run_phase2_enrichment(
                    config["extracted_assets_dir"],
                    config["use_gpu"]
                )
            elif args.phase == 3:
                results = run_phase3_database(
                    config["extracted_assets_dir"],
                    config["database_config"]
                )
            elif args.phase == 4:
                results = run_phase4_training_preparation(
                    config["extracted_assets_dir"],
                    config["training_data_dir"],
                    config["output_format"],
                    config["base_model"]
                )
            elif args.phase == 5:
                results = run_phase5_fine_tuning(
                    config["training_data_dir"],
                    config["fine_tuned_models_dir"],
                    config["base_model"],
                    config["epochs"],
                    config["batch_size"],
                    config["learning_rate"]
                )
            elif args.phase == 6:
                # For phase 6, we need a model path
                model_path = input("Enter the path to the fine-tuned model: ")
                results = run_phase6_rag_pipeline(model_path, config["database_config"])
                
            logger.info(f"Phase {args.phase} completed successfully!")
            
        else:
            # Run complete pipeline
            results = run_complete_pipeline(config)
            logger.info("Complete pipeline executed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 