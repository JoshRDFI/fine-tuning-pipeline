-- Enable pgvector extension (run once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: documents
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    date_added TIMESTAMP DEFAULT NOW(),
    notes TEXT
);

-- Table: sections (for logical document structure)
CREATE TABLE sections (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    section_title TEXT,
    section_order INTEGER,
    page_number INTEGER,
    notes TEXT
);

-- Table: text_chunks
CREATE TABLE text_chunks (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    chunk_order INTEGER,
    content TEXT,
    embedding vector(768), -- adjust dimension to match your model
    metadata JSONB,
    is_uncertain BOOLEAN DEFAULT FALSE
);

-- Table: images
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    caption TEXT,
    description TEXT,
    embedding vector(768), -- adjust dimension to match your model
    tags TEXT[],
    page_number INTEGER,
    width INTEGER,
    height INTEGER,
    ocr_text TEXT,
    metadata JSONB,
    is_uncertain BOOLEAN DEFAULT FALSE
);

-- Table: tables
CREATE TABLE tables (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    csv_path TEXT,
    json_path TEXT,
    image_path TEXT,
    description TEXT,
    embedding vector(768), -- adjust dimension to match your model
    tags TEXT[],
    page_number INTEGER,
    ocr_text TEXT,
    metadata JSONB,
    is_uncertain BOOLEAN DEFAULT FALSE
);

-- Table: feedback (for RLHF)
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    asset_type TEXT CHECK (asset_type IN ('text', 'image', 'table')),
    asset_id INTEGER,
    user_id TEXT,
    feedback_type TEXT, -- e.g., 'like', 'dislike', 'correction', etc.
    feedback_text TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Table: model_versions
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    base_model TEXT,
    date_trained TIMESTAMP DEFAULT NOW(),
    training_data_path TEXT,
    notes TEXT
);

-- Table: rag_queries (for logging RAG usage)
CREATE TABLE rag_queries (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    query_text TEXT,
    response_text TEXT,
    response_assets JSONB, -- e.g., {"images": [...], "tables": [...]}
    timestamp TIMESTAMP DEFAULT NOW()
);