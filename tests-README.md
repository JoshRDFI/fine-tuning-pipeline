# Test Plan: Multi-Modal RAG + Training Pipeline

## Overview

This document outlines the test strategy for each phase of the pipeline. Each phase should have automated or semi-automated tests in the `root/tests` directory. All critical functionality, edge cases, and error handling should be covered.

---

## Phase 1: Data Ingestion & Extraction

**Objectives:**
- Verify all PDFs in a directory (and subdirectories) are processed.
- Ensure text, images, and tables are extracted and saved with correct structure and references.
- Confirm OCR is used for scanned PDFs.
- Check logs for errors and "UNCERTAIN" markers.

**Example Test Cases:**
- Process a directory with mixed PDFs (native, scanned, encrypted, with/without images/tables).
- Validate output directory structure and file naming.
- Check that markdown references in text files point to actual image/table files.
- Confirm extraction log is generated and contains expected entries.

**Expected Outcomes:**
- All assets extracted and referenced.
- Errors/ambiguities logged and marked as "UNCERTAIN".
- No missing or misnamed files.

---

## Phase 2: Metadata Enrichment & Embedding Generation

**Objectives:**
- Ensure all images and tables have generated descriptions and embeddings.
- Validate that metadata is updated and complete.

**Example Test Cases:**
- Run enrichment on a sample set of extracted assets.
- Check that every image/table has a non-empty description and embedding vector.
- Validate metadata JSON for required fields (tags, source, etc.).

**Expected Outcomes:**
- All assets enriched with valid metadata and embeddings.
- No missing or malformed metadata.

---

## Phase 3: Database Construction

**Objectives:**
- Confirm database schema is created as specified.
- Ensure all assets and metadata are inserted correctly.
- Test vector search functionality.

**Example Test Cases:**
- Run schema migration and check all tables exist.
- Insert sample data and query for retrieval by ID, tag, and vector similarity.
- Validate referential integrity (e.g., images link to correct sections/documents).

**Expected Outcomes:**
- Database matches schema.
- All data retrievable as expected.
- Vector search returns relevant results.

---

## Phase 4: Training Data Preparation

**Objectives:**
- Verify training dataset is correctly formatted for the selected model.
- Ensure data splits (train/val/test) are correct if used.

**Example Test Cases:**
- Generate training data from sample assets.
- Load a batch into the training script and check for errors.
- Validate that text-image/table pairs are correctly linked.

**Expected Outcomes:**
- Training data loads without errors.
- Data pairs are correct and complete.

---

## Phase 5: Model Fine-Tuning

**Objectives:**
- Confirm fine-tuning runs to completion on the selected base model.
- Validate that the new model can be loaded and used in Ollama.

**Example Test Cases:**
- Fine-tune on a small dataset and check for successful completion.
- Run basic inference (text and image input) on the new model.
- Load the model in Ollama and verify it responds.

**Expected Outcomes:**
- Model fine-tunes and saves successfully.
- Inference produces reasonable outputs.
- Model is compatible with Ollama.

---

## Phase 6: RAG Pipeline Integration

**Objectives:**
- Ensure the model can retrieve and display relevant text, images, and tables from the database.
- Test adding new data to the RAG pipeline.

**Example Test Cases:**
- Query the RAG system with sample questions and check that correct assets are retrieved.
- Add a new PDF and verify it is processed and available for retrieval.
- Test retrieval of both text and non-text assets.

**Expected Outcomes:**
- RAG pipeline returns relevant, referenced assets.
- New data is integrated without errors.

---

## Phase 7: Streamlit UI Development

**Objectives:**
- Validate all UI features: chat, file upload, training tab, inference tab, asset display, etc.
- Ensure chat history and contextual memory work.

**Example Test Cases:**
- Upload a new PDF via UI and check it appears in the RAG pipeline.
- Run a chat session and verify history/context.
- Zoom and download images/tables from the UI.

**Expected Outcomes:**
- All UI features function as intended.
- No crashes or missing features.

---

## Phase 8: Testing & Validation

**Objectives:**
- Ensure all previous tests are automated and pass.
- Validate error handling and logging.

**Example Test Cases:**
- Run all test scripts in `root/tests`.
- Intentionally introduce errors (e.g., corrupt PDF) and check logs/outputs.
- Review summary reports for completeness.

**Expected Outcomes:**
- All tests pass or failures are logged and explained.
- System is robust to edge cases.

---

## Optional Phase: RLHF

**Objectives:**
- Test feedback collection and storage.
- Validate that feedback can be used for further fine-tuning or retrieval adjustment.

**Example Test Cases:**
- Submit feedback via UI and check it is stored in the database.
- Use feedback data to trigger a re-training or retrieval update.

**Expected Outcomes:**
- Feedback loop is functional.
- Model/retrieval improves with feedback.

---

## Instructions

- Place all test scripts in the `root/tests` directory.
- Use clear, explicit assertions and log all test results.
- Update this plan as the pipeline evolves.