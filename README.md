# Multi-Modal RAG + Training Pipeline

## Overview

This project provides an end-to-end, fully automated pipeline for extracting, organizing, and leveraging data from directories of PDF files—including text, images, and tables—for multi-modal LLM fine-tuning and retrieval-augmented generation (RAG). The system is designed to run efficiently on a local Ubuntu (WSL2) machine with an Nvidia 5080 GPU, and integrates with Ollama for LLM serving.

## Key Features

- **Automated PDF Extraction:** Extracts structured text, images, and tables (as both images and structured data) from PDFs, including OCR for scanned documents.
- **Metadata & Embeddings:** Enriches all assets with descriptions and vector embeddings using a vision model.
- **Database-Backed RAG:** Stores all assets and metadata in a PostgreSQL database with pgvector for fast multi-modal retrieval.
- **Multi-Modal Training:** Prepares and fine-tunes a multi-modal LLM (user-selectable base model) on the extracted dataset.
- **Streamlit UI:** Provides a user-friendly interface for chat, file upload, model training, and RAG-based querying with contextual memory and asset display.
- **RLHF Ready:** Supports collection of user feedback for future reinforcement learning from human feedback.
- **Extensible & Modular:** Each phase is modular, with clear completion checks and test coverage.

## Project Structure

```
fine-tuning-pipeline/
├── scripts/                    # Main pipeline scripts
│   ├── extract_pdfs.py        # Phase 1: PDF extraction
│   ├── enrich_metadata.py     # Phase 2: Metadata enrichment
│   ├── build_db.py           # Phase 3: Database construction
│   ├── prepare_training.py   # Phase 4: Training data preparation
│   ├── fine_tune.py          # Phase 5: Model fine-tuning
│   ├── rag_pipeline.py       # Phase 6: RAG integration
│   └── streamlit_app.py      # Phase 7: Streamlit UI
├── tests/                     # Test scripts
│   ├── test_phase1_extraction.py
│   └── test_all_phases.py
├── data/                      # Input PDF files
├── extracted_assets/          # Extracted content
├── training_data/             # Prepared training data
├── fine_tuned_models/         # Trained models
├── logs/                      # Log files
├── config.json               # Pipeline configuration
├── env_template.txt          # Environment variables template
├── run_pipeline.py           # Main orchestration script
├── requirements.txt          # Python dependencies
├── sample_schema.sql         # Database schema
├── pipeline_phases.md        # Detailed phase documentation
└── tests-README.md           # Testing documentation
```

## Pipeline Phases

1. **Data Ingestion & Extraction:**  
   Extracts all text, images, and tables from PDFs, preserving structure and references.

2. **Metadata Enrichment & Embedding Generation:**  
   Enriches assets with descriptions and generates vector embeddings.

3. **Database Construction:**  
   Builds and populates a PostgreSQL + pgvector database for RAG.

4. **Training Data Preparation:**  
   Prepares a dataset for multi-modal fine-tuning.

5. **Model Fine-Tuning:**  
   Fine-tunes the selected multi-modal model using the prepared dataset.

6. **RAG Pipeline Integration:**  
   Integrates the model with the RAG pipeline and database.

7. **Streamlit UI Development:**  
   Provides a user interface for chat, file upload, training, and asset display.

8. **Testing & Validation:**  
   Ensures all components work as expected with automated tests.

9. **(Optional) RLHF:**  
   Collects user feedback for further model alignment.

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd fine-tuning-pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install System Dependencies

**Ubuntu/WSL2:**
```bash
# Install Tesseract OCR
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install Poppler for PDF processing
sudo apt install poppler-utils

# Install PostgreSQL and pgvector
sudo apt install postgresql postgresql-contrib
# Follow pgvector installation: https://github.com/pgvector/pgvector#installation
```

**Windows:**
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler from: http://blog.alivate.com.au/poppler-windows/
- Install PostgreSQL from: https://www.postgresql.org/download/windows/

### 3. Configure Database

```bash
# Copy environment template
cp env_template.txt .env

# Edit .env with your database credentials
nano .env
```

### 4. Run the Pipeline

**Option 1: Complete Pipeline**
```bash
# Run all phases
python run_pipeline.py --input_dir data --output_dir pipeline_output
```

**Option 2: Individual Phases**
```bash
# Phase 1: Extract PDFs
python run_pipeline.py --phase 1 --input_dir data

# Phase 2: Enrich metadata
python run_pipeline.py --phase 2

# Phase 3: Build database
python run_pipeline.py --phase 3

# Phase 4: Prepare training data
python run_pipeline.py --phase 4

# Phase 5: Fine-tune model
python run_pipeline.py --phase 5

# Phase 6: Test RAG pipeline
python run_pipeline.py --phase 6
```

**Option 3: Streamlit UI**
```bash
# Launch the web interface
streamlit run scripts/streamlit_app.py
```

## Detailed Usage

### Phase 1: PDF Extraction
```bash
python scripts/extract_pdfs.py \
    --input_dir data \
    --output_dir extracted_assets \
    --include_subdirs \
    --use_gpu
```

### Phase 2: Metadata Enrichment
```bash
python scripts/enrich_metadata.py \
    --input_dir extracted_assets \
    --use_gpu
```

### Phase 3: Database Construction
```bash
python scripts/build_db.py \
    --extracted_assets extracted_assets \
    --host localhost \
    --database multimodal_rag \
    --user postgres
```

### Phase 4: Training Data Preparation
```bash
python scripts/prepare_training.py \
    --input_dir extracted_assets \
    --output_dir training_data \
    --format jsonl \
    --base_model llama2
```

### Phase 5: Model Fine-Tuning
```bash
python scripts/fine_tune.py \
    --training_data training_data \
    --output_dir fine_tuned_models \
    --base_model llama2 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

### Phase 6: RAG Pipeline
```bash
python scripts/rag_pipeline.py \
    --model_path fine_tuned_models/llama2_multimodal_20240101_120000 \
    --test
```

## Configuration

### Environment Variables
Create a `.env` file based on `env_template.txt`:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=multimodal_rag
DB_USER=postgres
DB_PASSWORD=your_password_here

# Model Configuration
BASE_MODEL=llama2
USE_GPU=true

# Training Configuration
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-5
MAX_LENGTH=512
```

### Pipeline Configuration
Edit `config.json` to customize pipeline settings:
```json
{
  "input_dir": "data",
  "extracted_assets_dir": "pipeline_output/extracted_assets",
  "training_data_dir": "pipeline_output/training_data",
  "fine_tuned_models_dir": "pipeline_output/fine_tuned_models",
  "database_config": {
    "host": "localhost",
    "port": "5432",
    "database": "multimodal_rag",
    "user": "postgres",
    "password": ""
  },
  "base_model": "llama2",
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 2e-5,
  "use_gpu": true
}
```

## Testing

### Run All Tests
```bash
python tests/test_all_phases.py --comprehensive
```

### Run Individual Phase Tests
```bash
python tests/test_all_phases.py --individual
```

### Run Specific Phase Test
```bash
python tests/test_all_phases.py --phase 1
```

### Run Phase 1 Tests Only
```bash
python tests/test_phase1_extraction.py
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Check database credentials in `.env`
   - Verify pgvector extension is installed

2. **GPU Memory Issues**
   - Reduce batch size in configuration
   - Use CPU-only mode with `--use_gpu false`

3. **PDF Extraction Errors**
   - Install Tesseract OCR
   - Ensure Poppler is installed
   - Check PDF file permissions

4. **Model Loading Issues**
   - Verify HuggingFace model access
   - Check available disk space
   - Ensure sufficient RAM

### Logs and Debugging

- Check `pipeline.log` for detailed execution logs
- Review phase-specific logs in output directories
- Use `--log_level DEBUG` for verbose output

## Performance Optimization

### GPU Optimization
- Use mixed precision training with `fp16=True`
- Optimize batch size for your GPU memory
- Use gradient accumulation for larger effective batch sizes

### Database Optimization
- Create indexes on frequently queried columns
- Use connection pooling for multiple queries
- Optimize vector similarity search with proper indexing

### Memory Management
- Process PDFs in batches for large datasets
- Use streaming for large file processing
- Implement proper cleanup in long-running processes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License

## Support

- Check the documentation in `pipeline_phases.md`
- Review test examples in `tests/`
- Open an issue for bugs or feature requests

---

**For detailed phase-by-phase documentation, see `pipeline_phases.md`**
**For testing documentation, see `tests-README.md`**