# Research Paper Q&A

## Overview
A research-oriented RAG application for answering questions
using academic research papers.

## Project Structure
- src/: Core application logic
- tests/: Unit and integration tests
- data/: Raw research papers and datasets
- outputs/: Generated results and evaluation logs

## Installation

```bash
python -m venv .venv


## Environment Configuration

### 1. Create local environment file

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

### 2. Configure credentials

Open `.env` and provide your actual credentials:

- GROQ_API_KEY
- GROQ_MODEL
- LOCAL_ENDPOINT_URL

### 3. Install dependencies

```powershell
uv sync --locked
```

### 4. Run the application

```powershell
uv run python -m src
```

### Security Guidelines

- Never commit `.env`.
- Never hardcode API keys in Python source files.
- Never share credentials through GitHub.
- Commit `.env.example` with placeholder values.
- Rotate credentials immediately if exposed.
