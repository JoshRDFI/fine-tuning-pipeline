#!/usr/bin/env python3
"""
Phase 6: RAG Pipeline Integration
Integrate model with RAG pipeline and database.
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
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Transformers and models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, model_path: str, db_config: Dict = None):
        self.model_path = Path(model_path)
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'multimodal_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        self.connection = None
        self.model = None
        self.tokenizer = None
        self.embedder = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize model, tokenizer, and database connection"""
        try:
            # Load fine-tuned model
            self.logger.info("Loading fine-tuned model...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load sentence transformer for embeddings
            self.logger.info("Loading sentence transformer...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                self.embedder = self.embedder.to('cuda')
                
            # Connect to database
            self.logger.info("Connecting to database...")
            self.connect_to_database()
            
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
            
    def connect_to_database(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
    def disconnect_from_database(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
            
    def query_rag(self, query: str, top_k: int = 5, include_images: bool = True, 
                  include_tables: bool = True) -> Dict:
        """Query the RAG pipeline"""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedder.encode(query)
            
            # Retrieve relevant text chunks
            text_results = self.retrieve_text_chunks(query_embedding, top_k)
            
            # Retrieve relevant images
            image_results = []
            if include_images:
                image_results = self.retrieve_images(query_embedding, top_k)
                
            # Retrieve relevant tables
            table_results = []
            if include_tables:
                table_results = self.retrieve_tables(query_embedding, top_k)
                
            # Generate response using retrieved context
            response = self.generate_response(query, text_results, image_results, table_results)
            
            # Log query
            self.log_query(query, response, text_results, image_results, table_results)
            
            return {
                "query": query,
                "response": response,
                "retrieved_text": text_results,
                "retrieved_images": image_results,
                "retrieved_tables": table_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG query: {str(e)}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "retrieved_text": [],
                "retrieved_images": [],
                "retrieved_tables": [],
                "timestamp": datetime.now().isoformat()
            }
            
    def retrieve_text_chunks(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Retrieve relevant text chunks using vector similarity"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query for similar text chunks
            cursor.execute("""
                SELECT tc.id, tc.content, tc.metadata, 
                       tc.embedding <-> %s::vector as distance,
                       s.section_title, d.doc_name
                FROM text_chunks tc
                JOIN sections s ON tc.section_id = s.id
                JOIN documents d ON s.document_id = d.id
                WHERE tc.embedding IS NOT NULL
                ORDER BY tc.embedding <-> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "distance": float(row["distance"]),
                    "section_title": row["section_title"],
                    "doc_name": row["doc_name"]
                })
                
            cursor.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving text chunks: {str(e)}")
            return []
            
    def retrieve_images(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Retrieve relevant images using vector similarity"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query for similar images
            cursor.execute("""
                SELECT i.id, i.file_path, i.caption, i.description, i.metadata,
                       i.embedding <-> %s::vector as distance,
                       s.section_title, d.doc_name
                FROM images i
                JOIN sections s ON i.section_id = s.id
                JOIN documents d ON s.document_id = d.id
                WHERE i.embedding IS NOT NULL
                ORDER BY i.embedding <-> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "file_path": row["file_path"],
                    "caption": row["caption"],
                    "description": row["description"],
                    "metadata": row["metadata"],
                    "distance": float(row["distance"]),
                    "section_title": row["section_title"],
                    "doc_name": row["doc_name"]
                })
                
            cursor.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving images: {str(e)}")
            return []
            
    def retrieve_tables(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Retrieve relevant tables using vector similarity"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query for similar tables
            cursor.execute("""
                SELECT t.id, t.csv_path, t.json_path, t.description, t.metadata,
                       t.embedding <-> %s::vector as distance,
                       s.section_title, d.doc_name
                FROM tables t
                JOIN sections s ON t.section_id = s.id
                JOIN documents d ON s.document_id = d.id
                WHERE t.embedding IS NOT NULL
                ORDER BY t.embedding <-> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "csv_path": row["csv_path"],
                    "json_path": row["json_path"],
                    "description": row["description"],
                    "metadata": row["metadata"],
                    "distance": float(row["distance"]),
                    "section_title": row["section_title"],
                    "doc_name": row["doc_name"]
                })
                
            cursor.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving tables: {str(e)}")
            return []
            
    def generate_response(self, query: str, text_results: List[Dict], 
                         image_results: List[Dict], table_results: List[Dict]) -> str:
        """Generate response using retrieved context"""
        try:
            # Build context from retrieved results
            context_parts = []
            
            # Add text context
            if text_results:
                context_parts.append("Relevant text content:")
                for i, result in enumerate(text_results[:3]):  # Limit to top 3
                    context_parts.append(f"{i+1}. {result['content'][:200]}...")
                    
            # Add image context
            if image_results:
                context_parts.append("\nRelevant images:")
                for i, result in enumerate(image_results[:2]):  # Limit to top 2
                    context_parts.append(f"{i+1}. {result['description']}")
                    
            # Add table context
            if table_results:
                context_parts.append("\nRelevant tables:")
                for i, result in enumerate(table_results[:2]):  # Limit to top 2
                    context_parts.append(f"{i+1}. {result['description']}")
                    
            context = "\n".join(context_parts)
            
            # Create prompt
            prompt = f"""### Instruction:
Based on the following context from documents, please answer the question: {query}

Context:
{context}

### Response:
"""
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
            
    def log_query(self, query: str, response: str, text_results: List[Dict], 
                  image_results: List[Dict], table_results: List[Dict]):
        """Log query and response to database"""
        try:
            cursor = self.connection.cursor()
            
            # Prepare response assets
            response_assets = {
                "text_chunks": [{"id": r["id"], "distance": r["distance"]} for r in text_results],
                "images": [{"id": r["id"], "distance": r["distance"]} for r in image_results],
                "tables": [{"id": r["id"], "distance": r["distance"]} for r in table_results]
            }
            
            # Insert into rag_queries table
            cursor.execute("""
                INSERT INTO rag_queries (user_id, query_text, response_text, response_assets)
                VALUES (%s, %s, %s, %s)
            """, ("default_user", query, response, json.dumps(response_assets)))
            
            self.connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error logging query: {str(e)}")
            
    def add_new_data(self, pdf_path: str) -> Dict:
        """Add new PDF data to the RAG pipeline"""
        try:
            self.logger.info(f"Adding new data: {pdf_path}")
            
            # This would integrate with the extraction and enrichment pipeline
            # For now, we'll just log the addition
            result = {
                "pdf_path": pdf_path,
                "status": "pending",
                "message": "New data addition requires running extraction and enrichment pipeline",
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"New data addition logged: {pdf_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding new data: {str(e)}")
            return {
                "pdf_path": pdf_path,
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_rag_pipeline(self) -> Dict:
        """Test the RAG pipeline with sample queries"""
        test_queries = [
            "What are the main topics discussed in the documents?",
            "Can you show me any charts or graphs from the documents?",
            "What data tables are available in the documents?",
            "Summarize the key findings from the documents."
        ]
        
        results = []
        for query in test_queries:
            try:
                result = self.query_rag(query, top_k=3)
                results.append({
                    "query": query,
                    "success": True,
                    "response_length": len(result["response"]),
                    "text_results": len(result["retrieved_text"]),
                    "image_results": len(result["retrieved_images"]),
                    "table_results": len(result["retrieved_tables"])
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
                
        return {
            "test_queries": results,
            "successful_queries": sum(1 for r in results if r["success"]),
            "total_queries": len(results)
        }
        
    def save_rag_summary(self, test_results: Dict):
        """Save RAG pipeline summary"""
        summary = {
            "rag_pipeline_info": {
                "model_path": str(self.model_path),
                "database_config": {
                    "host": self.db_config["host"],
                    "database": self.db_config["database"]
                },
                "initialization_date": datetime.now().isoformat()
            },
            "test_results": test_results
        }
        
        summary_file = Path("rag_pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Create human-readable report
        txt_report = Path("rag_pipeline_summary.txt")
        with open(txt_report, 'w') as f:
            f.write("RAG Pipeline Summary Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Model path: {self.model_path}\n")
            f.write(f"Database: {self.db_config['database']} on {self.db_config['host']}\n")
            f.write(f"Initialization date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Test Results:\n")
            f.write(f"  - Total queries: {test_results['total_queries']}\n")
            f.write(f"  - Successful queries: {test_results['successful_queries']}\n")
            f.write(f"  - Success rate: {test_results['successful_queries']/test_results['total_queries']*100:.1f}%\n\n")
            
            f.write("Query Details:\n")
            for i, result in enumerate(test_results['test_queries']):
                f.write(f"  {i+1}. {result['query']}\n")
                if result['success']:
                    f.write(f"     ✓ Success - Response: {result['response_length']} chars\n")
                    f.write(f"     Retrieved: {result['text_results']} text, {result['image_results']} images, {result['table_results']} tables\n")
                else:
                    f.write(f"     ✗ Failed - Error: {result['error']}\n")

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Integration")
    parser.add_argument("--model_path", required=True, 
                       help="Path to fine-tuned model")
    parser.add_argument("--host", help="Database host")
    parser.add_argument("--port", help="Database port")
    parser.add_argument("--database", help="Database name")
    parser.add_argument("--user", help="Database user")
    parser.add_argument("--password", help="Database password")
    parser.add_argument("--test", action="store_true", 
                       help="Run RAG pipeline tests")
    parser.add_argument("--query", 
                       help="Run a specific query")
    
    args = parser.parse_args()
    
    # Build database configuration
    db_config = {}
    if args.host:
        db_config['host'] = args.host
    if args.port:
        db_config['port'] = args.port
    if args.database:
        db_config['database'] = args.database
    if args.user:
        db_config['user'] = args.user
    if args.password:
        db_config['password'] = args.password
        
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(args.model_path, db_config)
    
    try:
        if args.query:
            # Run specific query
            result = rag_pipeline.query_rag(args.query)
            print(f"\nQuery: {result['query']}")
            print(f"Response: {result['response']}")
            print(f"Retrieved {len(result['retrieved_text'])} text chunks, "
                  f"{len(result['retrieved_images'])} images, "
                  f"{len(result['retrieved_tables'])} tables")
                  
        elif args.test:
            # Run tests
            test_results = rag_pipeline.test_rag_pipeline()
            rag_pipeline.save_rag_summary(test_results)
            
            print(f"\nRAG Pipeline Tests Completed!")
            print(f"Successful queries: {test_results['successful_queries']}/{test_results['total_queries']}")
            print("Check rag_pipeline_summary.txt for details")
            
        else:
            # Interactive mode
            print("RAG Pipeline initialized. Enter queries (type 'quit' to exit):")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if query:
                    result = rag_pipeline.query_rag(query)
                    print(f"\nResponse: {result['response']}")
                    print(f"Retrieved {len(result['retrieved_text'])} text chunks, "
                          f"{len(result['retrieved_images'])} images, "
                          f"{len(result['retrieved_tables'])} tables")
                          
    finally:
        rag_pipeline.disconnect_from_database()

if __name__ == "__main__":
    main() 