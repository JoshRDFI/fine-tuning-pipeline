#!/usr/bin/env python3
"""
Phase 3: Database Construction
Build and populate a PostgreSQL + pgvector database.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseBuilder:
    def __init__(self, db_config: Dict = None):
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'multimodal_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        self.connection = None
        self.setup_logging()
        
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
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
            
    def disconnect_from_database(self):
        """Alias for disconnect method for consistency"""
        self.disconnect()
            
    def create_schema(self):
        """Create database schema based on sample_schema.sql"""
        try:
            cursor = self.connection.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    date_added TIMESTAMP DEFAULT NOW(),
                    notes TEXT
                );
            """)
            
            # Create sections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    section_title TEXT,
                    section_order INTEGER,
                    page_number INTEGER,
                    notes TEXT
                );
            """)
            
            # Create text_chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id SERIAL PRIMARY KEY,
                    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                    chunk_order INTEGER,
                    content TEXT,
                    embedding vector(384),
                    metadata JSONB,
                    is_uncertain BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id SERIAL PRIMARY KEY,
                    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    caption TEXT,
                    description TEXT,
                    embedding vector(512),
                    tags TEXT[],
                    page_number INTEGER,
                    width INTEGER,
                    height INTEGER,
                    ocr_text TEXT,
                    metadata JSONB,
                    is_uncertain BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create tables table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id SERIAL PRIMARY KEY,
                    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                    csv_path TEXT,
                    json_path TEXT,
                    image_path TEXT,
                    description TEXT,
                    embedding vector(384),
                    tags TEXT[],
                    page_number INTEGER,
                    ocr_text TEXT,
                    metadata JSONB,
                    is_uncertain BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    asset_type TEXT CHECK (asset_type IN ('text', 'image', 'table')),
                    asset_id INTEGER,
                    user_id TEXT,
                    feedback_type TEXT,
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create model_versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT,
                    base_model TEXT,
                    date_trained TIMESTAMP DEFAULT NOW(),
                    training_data_path TEXT,
                    notes TEXT
                );
            """)
            
            # Create rag_queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_queries (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    query_text TEXT,
                    response_text TEXT,
                    response_assets JSONB,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)
            
            self.connection.commit()
            self.logger.info("Database schema created successfully")
            
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Failed to create schema: {str(e)}")
            raise
        finally:
            cursor.close()
            
    def populate_database(self, extracted_assets_dir: str):
        """Populate database with extracted and enriched assets"""
        extracted_assets_path = Path(extracted_assets_dir)
        
        if not extracted_assets_path.exists():
            raise ValueError(f"Extracted assets directory {extracted_assets_dir} does not exist")
            
        self.logger.info("Starting database population")
        
        results = {
            "documents_inserted": 0,
            "sections_inserted": 0,
            "text_chunks_inserted": 0,
            "images_inserted": 0,
            "tables_inserted": 0,
            "errors": []
        }
        
        # Process each PDF directory
        for pdf_dir in extracted_assets_path.iterdir():
            if pdf_dir.is_dir() and pdf_dir.name not in ["text", "images", "tables", "metadata"]:
                try:
                    self.logger.info(f"Processing PDF directory: {pdf_dir.name}")
                    pdf_results = self.insert_pdf_data(pdf_dir)
                    
                    # Aggregate results
                    for key in results:
                        if key != "errors":
                            results[key] += pdf_results.get(key, 0)
                    results["errors"].extend(pdf_results.get("errors", []))
                    
                except Exception as e:
                    self.logger.error(f"Failed to process PDF directory {pdf_dir.name}: {str(e)}")
                    results["errors"].append({
                        "pdf_dir": pdf_dir.name,
                        "error": str(e)
                    })
                    
        self.logger.info("Database population completed")
        return results
        
    def insert_pdf_data(self, pdf_dir: Path) -> Dict:
        """Insert data for a single PDF"""
        results = {
            "documents_inserted": 0,
            "sections_inserted": 0,
            "text_chunks_inserted": 0,
            "images_inserted": 0,
            "tables_inserted": 0,
            "errors": []
        }
        
        cursor = self.connection.cursor()
        
        try:
            # Load PDF metadata
            metadata_file = pdf_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"pdf_name": pdf_dir.name, "source_path": str(pdf_dir)}
                
            # Insert document record
            cursor.execute("""
                INSERT INTO documents (doc_name, source_path, notes)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (
                metadata.get("pdf_name", pdf_dir.name),
                metadata.get("source_path", str(pdf_dir)),
                json.dumps(metadata)
            ))
            
            document_id = cursor.fetchone()[0]
            results["documents_inserted"] += 1
            
            # Insert sections
            sections = self.create_sections_from_structure(pdf_dir, document_id)
            for section in sections:
                cursor.execute("""
                    INSERT INTO sections (document_id, section_title, section_order, page_number, notes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    document_id,
                    section["title"],
                    section["order"],
                    section["page_number"],
                    section["notes"]
                ))
                
                section_id = cursor.fetchone()[0]
                results["sections_inserted"] += 1
                
                # Insert text chunks for this section
                text_chunks = self.get_text_chunks_for_section(pdf_dir, section_id)
                for chunk in text_chunks:
                    cursor.execute("""
                        INSERT INTO text_chunks (section_id, chunk_order, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        section_id,
                        chunk["order"],
                        chunk["content"],
                        chunk["embedding"],
                        json.dumps(chunk["metadata"])
                    ))
                    results["text_chunks_inserted"] += 1
                    
                # Insert images for this section
                images = self.get_images_for_section(pdf_dir, section_id)
                for image in images:
                    cursor.execute("""
                        INSERT INTO images (section_id, file_path, caption, description, embedding, 
                                          tags, page_number, width, height, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        section_id,
                        image["file_path"],
                        image["caption"],
                        image["description"],
                        image["embedding"],
                        image["tags"],
                        image["page_number"],
                        image["width"],
                        image["height"],
                        json.dumps(image["metadata"])
                    ))
                    results["images_inserted"] += 1
                    
                # Insert tables for this section
                tables = self.get_tables_for_section(pdf_dir, section_id)
                for table in tables:
                    cursor.execute("""
                        INSERT INTO tables (section_id, csv_path, json_path, description, 
                                          embedding, tags, page_number, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        section_id,
                        table["csv_path"],
                        table["json_path"],
                        table["description"],
                        table["embedding"],
                        table["tags"],
                        table["page_number"],
                        json.dumps(table["metadata"])
                    ))
                    results["tables_inserted"] += 1
                    
            self.connection.commit()
            
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Failed to insert PDF data for {pdf_dir.name}: {str(e)}")
            results["errors"].append({
                "pdf_dir": pdf_dir.name,
                "error": str(e)
            })
        finally:
            cursor.close()
            
        return results
        
    def create_sections_from_structure(self, pdf_dir: Path, document_id: int) -> List[Dict]:
        """Create sections based on PDF structure"""
        sections = []
        
        # Look for section files in text directory
        text_dir = pdf_dir / "text"
        if text_dir.exists():
            section_files = list(text_dir.glob("section_*.md"))
            
            for i, section_file in enumerate(sorted(section_files)):
                # Extract section title from markdown
                with open(section_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    title = lines[0].replace('#', '').strip() if lines else f"Section {i+1}"
                    
                sections.append({
                    "title": title,
                    "order": i + 1,
                    "page_number": i + 1,  # Approximate
                    "notes": f"Extracted from {section_file.name}"
                })
                
        # If no sections found, create a default section
        if not sections:
            sections.append({
                "title": "Main Content",
                "order": 1,
                "page_number": 1,
                "notes": "Default section for unstructured content"
            })
            
        return sections
        
    def get_text_chunks_for_section(self, pdf_dir: Path, section_id: int) -> List[Dict]:
        """Get text chunks for a section"""
        chunks = []
        text_dir = pdf_dir / "text"
        
        if text_dir.exists():
            # Process page files
            page_files = list(text_dir.glob("page_*.txt"))
            
            for i, page_file in enumerate(sorted(page_files)):
                try:
                    # Read text content
                    with open(page_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Load enriched metadata if available
                    metadata_file = page_file.with_suffix('.json')
                    embedding = None
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            embedding = metadata.get("embedding")
                            
                    # Create chunk
                    chunks.append({
                        "order": i + 1,
                        "content": content,
                        "embedding": embedding,
                        "metadata": {
                            "file_path": str(page_file),
                            "content_length": len(content)
                        }
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process text chunk {page_file}: {str(e)}")
                    
        return chunks
        
    def get_images_for_section(self, pdf_dir: Path, section_id: int) -> List[Dict]:
        """Get images for a section"""
        images = []
        images_dir = pdf_dir / "images"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                try:
                    # Load enriched metadata if available
                    metadata_file = image_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                        # Extract page number from filename
                        page_number = 1
                        if "_img_" in image_file.name:
                            try:
                                page_part = image_file.name.split("_img_")[0]
                                page_number = int(page_part.split("_")[-1])
                            except:
                                pass
                                
                        images.append({
                            "file_path": str(image_file),
                            "caption": f"Image from {image_file.name}",
                            "description": metadata.get("description", "an image"),
                            "embedding": metadata.get("embedding"),
                            "tags": [],
                            "page_number": page_number,
                            "width": metadata.get("width", 0),
                            "height": metadata.get("height", 0),
                            "metadata": metadata
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process image {image_file}: {str(e)}")
                    
        return images
        
    def get_tables_for_section(self, pdf_dir: Path, section_id: int) -> List[Dict]:
        """Get tables for a section"""
        tables = []
        tables_dir = pdf_dir / "tables"
        
        if tables_dir.exists():
            csv_files = list(tables_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    # Find corresponding JSON file
                    json_file = csv_file.with_suffix('.json')
                    
                    # Load enriched metadata if available
                    metadata_file = csv_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                        tables.append({
                            "csv_path": str(csv_file),
                            "json_path": str(json_file) if json_file.exists() else None,
                            "description": metadata.get("description", "a data table"),
                            "embedding": metadata.get("embedding"),
                            "tags": [],
                            "page_number": 1,  # Approximate
                            "metadata": metadata
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process table {csv_file}: {str(e)}")
                    
        return tables
        
    def test_database(self):
        """Test database functionality with sample queries"""
        self.logger.info("Testing database functionality")
        
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Test 1: Count records
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()["count"]
            self.logger.info(f"Documents in database: {doc_count}")
            
            cursor.execute("SELECT COUNT(*) as count FROM text_chunks")
            text_count = cursor.fetchone()["count"]
            self.logger.info(f"Text chunks in database: {text_count}")
            
            cursor.execute("SELECT COUNT(*) as count FROM images")
            image_count = cursor.fetchone()["count"]
            self.logger.info(f"Images in database: {image_count}")
            
            cursor.execute("SELECT COUNT(*) as count FROM tables")
            table_count = cursor.fetchone()["count"]
            self.logger.info(f"Tables in database: {table_count}")
            
            # Test 2: Vector search (if we have embeddings)
            if text_count > 0:
                cursor.execute("""
                    SELECT content, embedding <-> '[0.1, 0.2, 0.3]'::vector as distance
                    FROM text_chunks 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
                    LIMIT 5
                """)
                vector_results = cursor.fetchall()
                self.logger.info(f"Vector search test returned {len(vector_results)} results")
                
            # Test 3: Join query
            cursor.execute("""
                SELECT d.doc_name, COUNT(t.id) as text_chunks, COUNT(i.id) as images, COUNT(tab.id) as tables
                FROM documents d
                LEFT JOIN sections s ON d.id = s.document_id
                LEFT JOIN text_chunks t ON s.id = t.section_id
                LEFT JOIN images i ON s.id = i.section_id
                LEFT JOIN tables tab ON s.id = tab.section_id
                GROUP BY d.id, d.doc_name
            """)
            join_results = cursor.fetchall()
            self.logger.info(f"Join query returned {len(join_results)} document summaries")
            
            self.connection.commit()
            self.logger.info("Database tests completed successfully")
            
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Database test failed: {str(e)}")
            raise
        finally:
            cursor.close()
            
    def save_database_summary(self, results: Dict):
        """Save database population summary"""
        summary = {
            "database_config": {
                "host": self.db_config["host"],
                "database": self.db_config["database"],
                "user": self.db_config["user"]
            },
            "population_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = Path("database_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Create human-readable report
        txt_report = Path("database_summary.txt")
        with open(txt_report, 'w') as f:
            f.write("Database Construction Summary Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Database: {self.db_config['database']} on {self.db_config['host']}\n")
            f.write(f"Population completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Records inserted:\n")
            f.write(f"  - Documents: {results['documents_inserted']}\n")
            f.write(f"  - Sections: {results['sections_inserted']}\n")
            f.write(f"  - Text chunks: {results['text_chunks_inserted']}\n")
            f.write(f"  - Images: {results['images_inserted']}\n")
            f.write(f"  - Tables: {results['tables_inserted']}\n\n")
            f.write(f"Errors: {len(results['errors'])}\n")
            
            if results['errors']:
                f.write("\nErrors:\n")
                for error in results['errors']:
                    f.write(f"  - {error}\n")

def main():
    parser = argparse.ArgumentParser(description="Build and populate PostgreSQL database with pgvector")
    parser.add_argument("--extracted_assets", default="extracted_assets", 
                       help="Directory containing extracted and enriched assets")
    parser.add_argument("--host", help="Database host")
    parser.add_argument("--port", help="Database port")
    parser.add_argument("--database", help="Database name")
    parser.add_argument("--user", help="Database user")
    parser.add_argument("--password", help="Database password")
    parser.add_argument("--skip_tests", action="store_true", help="Skip database tests")
    
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
        
    # Create database builder and run
    builder = DatabaseBuilder(db_config)
    
    try:
        builder.connect()
        builder.create_schema()
        results = builder.populate_database(args.extracted_assets)
        
        if not args.skip_tests:
            builder.test_database()
            
        builder.save_database_summary(results)
        
        print(f"\nDatabase construction completed!")
        print(f"Documents: {results['documents_inserted']}")
        print(f"Text chunks: {results['text_chunks_inserted']}")
        print(f"Images: {results['images_inserted']}")
        print(f"Tables: {results['tables_inserted']}")
        print(f"Errors: {len(results['errors'])}")
        print("Check database_summary.txt for details")
        
    finally:
        builder.disconnect()

if __name__ == "__main__":
    main() 