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

As it's own file: structure.txt

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

## Setup Instructions

1. **Clone the repository and create a virtual environment:**
    ```bash
    git clone <your-repo-url>
    cd project-root
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install system dependencies:**
    - Tesseract OCR: `sudo apt install tesseract-ocr`
    - Poppler: `sudo apt install poppler-utils`
    - PostgreSQL + pgvector:  
      See [pgvector installation](https://github.com/pgvector/pgvector#installation)

4. **Configure database connection:**  
   Set your PostgreSQL connection string in a `.env` file.

5. **Run the pipeline:**  
   Each phase can be run via its script in the `scripts/` directory, or orchestrated via the Streamlit UI.

## Usage

- **Extract and process PDFs:**  
  Run `python scripts/extract_pdfs.py --input_dir <path-to-pdfs>`

- **Enrich metadata and generate embeddings:**  
  Run `python scripts/enrich_metadata.py`

- **Build and populate the database:**  
  Run `python scripts/build_db.py`

- **Prepare training data and fine-tune model:**  
  Use `prepare_training.py` and `fine_tune.py` as needed.

- **Launch the Streamlit UI:**  
  Run `streamlit run scripts/streamlit_app.py`

## Testing

- All test scripts are in the `tests/` directory.
- Run tests with:
    ```bash
    pytest tests/
    ```

## Notes

- The pipeline is modular—each phase can be run independently or as part of the full workflow.
- The system is designed for local execution on Ubuntu (WSL2) with Nvidia 5080 GPU and Ollama for LLM serving.
- For RLHF, user feedback is stored in the database and can be used for further fine-tuning.

## License

MIT License

---

**For more details, see the phase-by-phase pipeline spec and test plan in the project root.**