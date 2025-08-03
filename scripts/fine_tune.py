#!/usr/bin/env python3
"""
Phase 5: Model Fine-Tuning
Fine-tune the selected multi-modal model.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np

# Transformers and training
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_dataset
import evaluate

# Ollama integration
import requests

class ModelFineTuner:
    def __init__(self, training_data_dir: str, output_dir: str = "fine_tuned_models"):
        self.training_data_dir = Path(training_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "fine_tuning.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def fine_tune_model(self, 
                       base_model: str = "llama2",
                       training_format: str = "jsonl",
                       model_name: str = None,
                       epochs: int = 3,
                       batch_size: int = 4,
                       learning_rate: float = 2e-5,
                       max_length: int = 512,
                       gradient_accumulation_steps: int = 4,
                       save_steps: int = 500,
                       eval_steps: int = 500,
                       warmup_steps: int = 100) -> Dict:
        """Fine-tune a multi-modal model"""
        
        self.logger.info(f"Starting fine-tuning of {base_model}")
        
        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{base_model}_multimodal_{timestamp}"
            
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        train_dataset, val_dataset = self.load_training_data(training_format)
        
        # Load base model and tokenizer
        model, tokenizer = self.load_base_model(base_model)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset, tokenizer, max_length)
        val_dataset = self.prepare_dataset(val_dataset, tokenizer, max_length)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=100,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        self.logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(str(model_output_dir))
        
        # Evaluate model
        eval_result = trainer.evaluate()
        
        # Save training results
        results = {
            "model_name": model_name,
            "base_model": base_model,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            "training_results": {
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
            },
            "model_path": str(model_output_dir),
            "training_date": datetime.now().isoformat()
        }
        
        # Save results
        results_file = model_output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Test model inference
        inference_results = self.test_model_inference(model_output_dir, tokenizer)
        results["inference_test"] = inference_results
        
        # Save final results
        self.save_training_summary(results)
        
        self.logger.info(f"Fine-tuning completed! Model saved to {model_output_dir}")
        return results
        
    def load_training_data(self, training_format: str) -> Tuple[Dataset, Dataset]:
        """Load training data in the specified format"""
        if training_format == "jsonl":
            return self.load_jsonl_data()
        elif training_format == "huggingface":
            return self.load_huggingface_data()
        else:
            raise ValueError(f"Unsupported training format: {training_format}")
            
    def load_jsonl_data(self) -> Tuple[Dataset, Dataset]:
        """Load data from JSONL files"""
        train_file = self.training_data_dir / "train.jsonl"
        val_file = self.training_data_dir / "val.jsonl"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
            
        # Load training data
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))
                    
        # Load validation data
        val_data = []
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        val_data.append(json.loads(line))
        else:
            # Use a subset of training data for validation
            val_data = train_data[:len(train_data)//10]
            train_data = train_data[len(train_data)//10:]
            
        self.logger.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
        
        return Dataset.from_list(train_data), Dataset.from_list(val_data)
        
    def load_huggingface_data(self) -> Tuple[Dataset, Dataset]:
        """Load data from HuggingFace dataset format"""
        dataset_path = self.training_data_dir / "huggingface_dataset"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"HuggingFace dataset not found: {dataset_path}")
            
        dataset = load_dataset(str(dataset_path))
        
        train_dataset = dataset.get("train", dataset.get("training"))
        val_dataset = dataset.get("validation", dataset.get("val"))
        
        if train_dataset is None:
            raise ValueError("No training split found in dataset")
            
        if val_dataset is None:
            # Use a subset of training data for validation
            train_data = list(train_dataset)
            val_data = train_data[:len(train_data)//10]
            train_data = train_data[len(train_data)//10:]
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)
            
        self.logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
        
    def load_base_model(self, base_model: str):
        """Load base model and tokenizer"""
        try:
            # Map model names to HuggingFace model IDs
            model_mapping = {
                "llama2": "meta-llama/Llama-2-7b-hf",
                "llama2-7b": "meta-llama/Llama-2-7b-hf",
                "llama2-13b": "meta-llama/Llama-2-13b-hf",
                "mistral": "mistralai/Mistral-7B-v0.1",
                "mistral-7b": "mistralai/Mistral-7B-v0.1",
                "phi-2": "microsoft/phi-2",
                "qwen": "Qwen/Qwen-7B",
                "qwen-7b": "Qwen/Qwen-7B"
            }
            
            model_id = model_mapping.get(base_model.lower(), base_model)
            
            self.logger.info(f"Loading model: {model_id}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Resize token embeddings if needed
            model.resize_token_embeddings(len(tokenizer))
            
            self.logger.info(f"Model loaded successfully: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model {base_model}: {str(e)}")
            raise
            
    def prepare_dataset(self, dataset: Dataset, tokenizer, max_length: int) -> Dataset:
        """Prepare dataset for training"""
        def tokenize_function(examples):
            # Combine instruction and response
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                response = examples["response"][i]
                
                # Format as instruction-following format
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n### End\n"
                texts.append(text)
                
            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # Set labels to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
            
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def test_model_inference(self, model_path: Path, tokenizer) -> Dict:
        """Test model inference with sample prompts"""
        try:
            # Load fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Test prompts
            test_prompts = [
                "### Instruction:\nSummarize the main points from the document.\n\n### Response:\n",
                "### Instruction:\nDescribe the image in the document.\n\n### Response:\n",
                "### Instruction:\nAnalyze the table data.\n\n### Response:\n"
            ]
            
            results = []
            for prompt in test_prompts:
                try:
                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    
                    results.append({
                        "prompt": prompt,
                        "response": response,
                        "success": True
                    })
                    
                except Exception as e:
                    results.append({
                        "prompt": prompt,
                        "response": str(e),
                        "success": False
                    })
                    
            return {
                "test_results": results,
                "successful_tests": sum(1 for r in results if r["success"])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to test model inference: {str(e)}")
            return {
                "test_results": [],
                "successful_tests": 0,
                "error": str(e)
            }
            
    def save_training_summary(self, results: Dict):
        """Save training summary"""
        summary_file = self.output_dir / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Model Fine-Tuning Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Model name: {results['model_name']}\n")
            f.write(f"Base model: {results['base_model']}\n")
            f.write(f"Training date: {results['training_date']}\n\n")
            
            f.write("Training Configuration:\n")
            config = results['training_config']
            f.write(f"  - Epochs: {config['epochs']}\n")
            f.write(f"  - Batch size: {config['batch_size']}\n")
            f.write(f"  - Learning rate: {config['learning_rate']}\n")
            f.write(f"  - Max length: {config['max_length']}\n\n")
            
            f.write("Training Results:\n")
            train_results = results['training_results']
            f.write(f"  - Training loss: {train_results['train_loss']:.4f}\n")
            f.write(f"  - Evaluation loss: {train_results['eval_loss']:.4f}\n")
            f.write(f"  - Training time: {train_results['train_runtime']:.2f} seconds\n")
            f.write(f"  - Samples per second: {train_results['train_samples_per_second']:.2f}\n\n")
            
            f.write("Model Path:\n")
            f.write(f"  {results['model_path']}\n\n")
            
            f.write("Inference Test Results:\n")
            inference = results.get('inference_test', {})
            f.write(f"  - Successful tests: {inference.get('successful_tests', 0)}\n")
            
            if 'test_results' in inference:
                for i, test in enumerate(inference['test_results']):
                    f.write(f"  - Test {i+1}: {'✓' if test['success'] else '✗'}\n")
                    
    def convert_to_ollama_format(self, model_path: Path, model_name: str) -> bool:
        """Convert fine-tuned model to Ollama format"""
        try:
            self.logger.info(f"Converting model to Ollama format: {model_name}")
            
            # Create Ollama model directory
            ollama_dir = self.output_dir / f"{model_name}_ollama"
            ollama_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Modelfile
            template_content = '{{"prompt": "### Instruction:\\n{{.Input}}\\n\\n### Response:\\n", "response": "{{.Response}}\\n\\n### End\\n"}}'
            system_content = "You are a helpful AI assistant trained on multi-modal document data. You can analyze text, images, and tables from documents."
            
            modelfile_content = f"""FROM {model_path}
TEMPLATE {template_content}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "### End"
PARAMETER stop "### Instruction:"
SYSTEM {system_content}
"""
            
            modelfile_path = ollama_dir / "Modelfile"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
                
            # Copy model files
            import shutil
            shutil.copytree(model_path, ollama_dir / "model", dirs_exist_ok=True)
            
            self.logger.info(f"Ollama model created at: {ollama_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert to Ollama format: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Fine-tune multi-modal model")
    parser.add_argument("--training_data", default="training_data", 
                       help="Directory containing training data")
    parser.add_argument("--output_dir", default="fine_tuned_models", 
                       help="Output directory for fine-tuned models")
    parser.add_argument("--base_model", default="llama2", 
                       help="Base model to fine-tune")
    parser.add_argument("--model_name", 
                       help="Name for the fine-tuned model")
    parser.add_argument("--training_format", choices=["jsonl", "huggingface"], default="jsonl",
                       help="Format of training data")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")
    parser.add_argument("--convert_to_ollama", action="store_true", 
                       help="Convert model to Ollama format after training")
    
    args = parser.parse_args()
    
    # Create fine-tuner and run
    fine_tuner = ModelFineTuner(args.training_data, args.output_dir)
    results = fine_tuner.fine_tune_model(
        base_model=args.base_model,
        training_format=args.training_format,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    # Convert to Ollama format if requested
    if args.convert_to_ollama:
        model_path = Path(results["model_path"])
        model_name = results["model_name"]
        success = fine_tuner.convert_to_ollama_format(model_path, model_name)
        if success:
            print(f"Model converted to Ollama format successfully!")
    
    print(f"\nFine-tuning completed!")
    print(f"Model: {results['model_name']}")
    print(f"Training loss: {results['training_results']['train_loss']:.4f}")
    print(f"Evaluation loss: {results['training_results']['eval_loss']:.4f}")
    print(f"Model saved to: {results['model_path']}")
    print(f"Check {args.output_dir}/training_summary.txt for details")

if __name__ == "__main__":
    main() 