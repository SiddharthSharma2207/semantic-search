# Climba - Semantic FAQ Search Engine

Climba is a semantic search engine designed for FAQ sections. It leverages Large Language Models (LLMs) to generate question variations and uses Vector Database (ChromaDB) for efficient semantic retrieval.

## Features

- **Semantic Search**: Understands the intent behind user queries, not just keyword matching.
- **Data Augmentation**: Uses LLMs to generate diverse phrasing for FAQ questions to improve retrieval accuracy.
- **Hybrid Search**: Combines vector similarity with keyword filtering using spaCy.
- **CLI Interface**: Simple command-line interface for testing queries.

## Tech Stack

- **Python**: Core programming language.
- **ChromaDB**: Vector database for storing and querying embeddings.
- **OpenAI API / LLM**: For generating question variations.
- **spaCy**: For keyword extraction and linguistic processing.
- **FastAPI / Pydantic**: For configuration and potential API extensions.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd climba
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Configuration

1.  Create a `.env` file in the root directory.
2.  Add the following configuration variables:

    ```env
    API_KEY=your_llm_api_key
    BASE_URL=your_llm_base_url
    LLM_MODEL_NAME=your_model_name
    COLLECTION_NAME=faq_collection
    CHROMA_DB_DIR=chroma_db
    ```

## Usage

### 1. Generate Data (Optional)

If you want to augment your `data/faq.json` with LLM-generated variations:

```bash
python scripts/generate_data.py
```

This will create `data/generated_faq_llm.json`.

### 2. Build Database

Index the FAQ data into ChromaDB:

```bash
python scripts/build_db.py
```

### 3. Run Search

Use the CLI to search the FAQ:

```bash
python main.py --query "How do I reset my password?"
```

## Project Structure

- `src/`: Core application logic (Semantic Engine).
- `scripts/`: Data processing and database building scripts.
- `data/`: FAQ data files (original and generated).
- `chroma_db/`: Persisted vector database.
- `config.py`: Configuration settings.
- `main.py`: Main entry point for CLI.
