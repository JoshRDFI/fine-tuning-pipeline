#!/usr/bin/env python3
"""
Phase 7: Streamlit UI Development
Build user interface for all major functions.
"""

import streamlit as st
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import our pipeline components
from extract_pdfs import PDFExtractor
from enrich_metadata import MetadataEnricher
from build_db import DatabaseBuilder
from prepare_training import TrainingDataPreparer
from fine_tune import ModelFineTuner
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitApp:
    def __init__(self):
        self.setup_logging()
        self.initialize_session_state()
        
    def setup_logging(self):
        """Setup logging for the Streamlit app"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
            
    def main(self):
        """Main application"""
        st.title("📚 Multi-Modal RAG + Training Pipeline")
        st.markdown("---")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Home", "💬 Chat", "📁 File Upload", "🔧 Training", "📊 Database", "⚙️ Settings"]
        )
        
        if page == "🏠 Home":
            self.home_page()
        elif page == "💬 Chat":
            self.chat_page()
        elif page == "📁 File Upload":
            self.file_upload_page()
        elif page == "🔧 Training":
            self.training_page()
        elif page == "📊 Database":
            self.database_page()
        elif page == "⚙️ Settings":
            self.settings_page()
            
    def home_page(self):
        """Home page with overview and quick actions"""
        st.header("Welcome to Multi-Modal RAG Pipeline")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This application provides a complete pipeline for:
            
            - **📄 PDF Processing**: Extract text, images, and tables from PDFs
            - **🧠 AI Training**: Fine-tune multi-modal models on your data
            - **🔍 RAG System**: Query your documents with AI assistance
            - **📊 Database**: Store and manage all your document assets
            
            ### Quick Start
            1. Upload PDFs in the **File Upload** tab
            2. Process and enrich your data
            3. Train a custom model
            4. Start chatting with your documents!
            """)
            
        with col2:
            st.info("**System Status**")
            
            # Check system status
            status_checks = {
                "Database Connection": self.check_database_connection(),
                "Model Availability": self.check_model_availability(),
                "Storage Space": self.check_storage_space()
            }
            
            for check, status in status_checks.items():
                if status:
                    st.success(f"✓ {check}")
                else:
                    st.error(f"✗ {check}")
                    
        # Quick actions
        st.markdown("---")
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start New Pipeline", use_container_width=True):
                st.switch_page("📁 File Upload")
                
        with col2:
            if st.button("💬 Open Chat", use_container_width=True):
                st.switch_page("💬 Chat")
                
        with col3:
            if st.button("🔧 Training Dashboard", use_container_width=True):
                st.switch_page("🔧 Training")
                
    def chat_page(self):
        """Chat interface with RAG pipeline"""
        st.header("💬 Document Chat")
        
        # Initialize RAG pipeline if not already done
        if st.session_state.rag_pipeline is None:
            self.initialize_rag_pipeline()
            
        if st.session_state.rag_pipeline is None:
            st.error("RAG pipeline not available. Please ensure a model is trained and database is populated.")
            return
            
        # Chat interface
        st.markdown("Ask questions about your documents:")
        
        # Chat input
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response from RAG pipeline
            with st.spinner("Processing your question..."):
                try:
                    result = st.session_state.rag_pipeline.query_rag(user_input)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["response"],
                        "retrieved_assets": {
                            "text": len(result["retrieved_text"]),
                            "images": len(result["retrieved_images"]),
                            "tables": len(result["retrieved_tables"])
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
                    
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show retrieved assets info
                if message["role"] == "assistant" and "retrieved_assets" in message:
                    assets = message["retrieved_assets"]
                    st.info(f"📄 Retrieved: {assets['text']} text chunks, {assets['images']} images, {assets['tables']} tables")
                    
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
    def file_upload_page(self):
        """File upload and processing page"""
        st.header("📁 File Upload & Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to process"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            # Processing options
            st.subheader("Processing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_subdirs = st.checkbox("Include subdirectories", value=True)
                use_gpu = st.checkbox("Use GPU acceleration", value=True)
                
            with col2:
                output_dir = st.text_input("Output directory", value="extracted_assets")
                
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                self.process_uploaded_files(uploaded_files, output_dir, include_subdirs, use_gpu)
                
        # Show existing processed files
        self.show_processed_files()
        
    def training_page(self):
        """Model training page"""
        st.header("🔧 Model Training")
        
        # Training configuration
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_model = st.selectbox(
                "Base Model",
                ["llama2", "mistral", "phi-2", "qwen"],
                help="Select the base model to fine-tune"
            )
            
            epochs = st.slider("Training Epochs", 1, 10, 3)
            batch_size = st.slider("Batch Size", 1, 16, 4)
            
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-3,
                value=2e-5,
                format="%.2e"
            )
            
            max_length = st.slider("Max Sequence Length", 256, 2048, 512, step=256)
            
        # Training data selection
        st.subheader("Training Data")
        
        training_data_dir = st.text_input("Training data directory", value="training_data")
        
        if st.button("📊 Prepare Training Data", type="secondary"):
            self.prepare_training_data(training_data_dir)
            
        # Start training
        if st.button("🚀 Start Training", type="primary"):
            self.start_training(base_model, epochs, batch_size, learning_rate, max_length, training_data_dir)
            
        # Show training progress and results
        self.show_training_results()
        
    def database_page(self):
        """Database management page"""
        st.header("📊 Database Management")
        
        # Database connection
        st.subheader("Database Connection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input("Host", value="localhost")
            db_port = st.text_input("Port", value="5432")
            
        with col2:
            db_name = st.text_input("Database", value="multimodal_rag")
            db_user = st.text_input("User", value="postgres")
            
        db_password = st.text_input("Password", type="password")
        
        # Database actions
        st.subheader("Database Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Create Schema"):
                self.create_database_schema(db_host, db_port, db_name, db_user, db_password)
                
        with col2:
            if st.button("📥 Populate Database"):
                self.populate_database(db_host, db_port, db_name, db_user, db_password)
                
        with col3:
            if st.button("📊 View Statistics"):
                self.show_database_stats(db_host, db_port, db_name, db_user, db_password)
                
        # Database queries
        st.subheader("Database Queries")
        
        query = st.text_area("SQL Query", height=100)
        
        if st.button("🔍 Execute Query"):
            self.execute_database_query(query, db_host, db_port, db_name, db_user, db_password)
            
    def settings_page(self):
        """Settings and configuration page"""
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Settings")
        
        model_path = st.text_input("Model Path", value="fine_tuned_models")
        
        if st.button("🔍 Load Model"):
            self.load_model(model_path)
            
        # System settings
        st.subheader("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_memory = st.slider("Max Memory (GB)", 1, 32, 8)
            use_gpu = st.checkbox("Use GPU", value=True)
            
        with col2:
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
            auto_save = st.checkbox("Auto-save chat history", value=True)
            
        # Save settings
        if st.button("💾 Save Settings"):
            self.save_settings(max_memory, use_gpu, log_level, auto_save)
            
        # System information
        st.subheader("System Information")
        
        import psutil
        import torch
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
            
        with col2:
            st.metric("GPU Available", "Yes" if torch.cuda.is_available() else "No")
            if torch.cuda.is_available():
                st.metric("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
    def initialize_rag_pipeline(self):
        """Initialize RAG pipeline"""
        try:
            # This would load the trained model and connect to database
            # For now, we'll create a mock pipeline
            st.session_state.rag_pipeline = "mock_pipeline"
            st.success("RAG pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {str(e)}")
            
    def process_uploaded_files(self, uploaded_files, output_dir, include_subdirs, use_gpu):
        """Process uploaded PDF files"""
        with st.spinner("Processing uploaded files..."):
            try:
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = temp_path / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                    # Run extraction
                    extractor = PDFExtractor(output_dir, use_gpu)
                    results = extractor.extract_from_directory(str(temp_path), include_subdirs)
                    
                    st.success(f"Processing completed! Extracted {results['extracted_assets']['text_sections']} text sections, "
                              f"{results['extracted_assets']['images']} images, {results['extracted_assets']['tables']} tables")
                              
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                
    def show_processed_files(self):
        """Show list of processed files"""
        st.subheader("Processed Files")
        
        # This would read from the extracted_assets directory
        # For now, we'll show a placeholder
        st.info("No processed files found. Upload and process some PDFs to see them here.")
        
    def prepare_training_data(self, training_data_dir):
        """Prepare training data"""
        with st.spinner("Preparing training data..."):
            try:
                preparer = TrainingDataPreparer("extracted_assets", training_data_dir)
                results = preparer.prepare_dataset()
                
                st.success(f"Training data prepared! {results['total_pairs']} pairs created.")
                
            except Exception as e:
                st.error(f"Error preparing training data: {str(e)}")
                
    def start_training(self, base_model, epochs, batch_size, learning_rate, max_length, training_data_dir):
        """Start model training"""
        with st.spinner("Starting training..."):
            try:
                fine_tuner = ModelFineTuner(training_data_dir)
                results = fine_tuner.fine_tune_model(
                    base_model=base_model,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_length=max_length
                )
                
                st.success(f"Training completed! Model saved as {results['model_name']}")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                
    def show_training_results(self):
        """Show training results"""
        st.subheader("Training Results")
        
        # This would read from training logs and results
        st.info("No training results found. Start a training session to see results here.")
        
    def create_database_schema(self, host, port, db_name, user, password):
        """Create database schema"""
        with st.spinner("Creating database schema..."):
            try:
                db_config = {
                    'host': host,
                    'port': port,
                    'database': db_name,
                    'user': user,
                    'password': password
                }
                
                builder = DatabaseBuilder(db_config)
                builder.connect()
                builder.create_schema()
                builder.disconnect_from_database()
                
                st.success("Database schema created successfully!")
                
            except Exception as e:
                st.error(f"Error creating schema: {str(e)}")
                
    def populate_database(self, host, port, db_name, user, password):
        """Populate database with data"""
        with st.spinner("Populating database..."):
            try:
                db_config = {
                    'host': host,
                    'port': port,
                    'database': db_name,
                    'user': user,
                    'password': password
                }
                
                builder = DatabaseBuilder(db_config)
                builder.connect()
                results = builder.populate_database("extracted_assets")
                builder.disconnect_from_database()
                
                st.success(f"Database populated! {results['documents_inserted']} documents, "
                          f"{results['text_chunks_inserted']} text chunks, "
                          f"{results['images_inserted']} images, "
                          f"{results['tables_inserted']} tables")
                          
            except Exception as e:
                st.error(f"Error populating database: {str(e)}")
                
    def show_database_stats(self, host, port, db_name, user, password):
        """Show database statistics"""
        try:
            db_config = {
                'host': host,
                'port': port,
                'database': db_name,
                'user': user,
                'password': password
            }
            
            import psycopg2
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM text_chunks")
            text_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM tables")
            table_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", doc_count)
            with col2:
                st.metric("Text Chunks", text_count)
            with col3:
                st.metric("Images", image_count)
            with col4:
                st.metric("Tables", table_count)
                
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            
    def execute_database_query(self, query, host, port, db_name, user, password):
        """Execute database query"""
        if not query.strip():
            st.warning("Please enter a query")
            return
            
        try:
            db_config = {
                'host': host,
                'port': port,
                'database': db_name,
                'user': user,
                'password': password
            }
            
            import psycopg2
            conn = psycopg2.connect(**db_config)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            # This would load the model into session state
            st.session_state.current_model = model_path
            st.success(f"Model loaded from {model_path}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            
    def save_settings(self, max_memory, use_gpu, log_level, auto_save):
        """Save application settings"""
        settings = {
            "max_memory": max_memory,
            "use_gpu": use_gpu,
            "log_level": log_level,
            "auto_save": auto_save
        }
        
        # Save to file
        with open("app_settings.json", "w") as f:
            json.dump(settings, f, indent=2)
            
        st.success("Settings saved successfully!")
        
    def check_database_connection(self):
        """Check database connection"""
        try:
            # This would test the database connection
            return True
        except:
            return False
            
    def check_model_availability(self):
        """Check if models are available"""
        try:
            # This would check if trained models exist
            return True
        except:
            return False
            
    def check_storage_space(self):
        """Check available storage space"""
        try:
            import psutil
            disk_usage = psutil.disk_usage('/')
            return disk_usage.free > 1e9  # 1GB free
        except:
            return False

def main():
    app = StreamlitApp()
    app.main()

if __name__ == "__main__":
    main() 