# Term1 Glossary-Controlled Translator

Runnable Streamlit implementation of the JP-EN Plant Translator concept.

## What It Does

- Translates Japanese plant-floor text to English
- Applies approved glossary terms before AI translation
- Supports text input and CSV, TXT, AS, DOCX, XLSX, and XLSM uploads
- Skips non-Japanese Excel cells to reduce cost and runtime
- Saves translation progress by batch for large files
- Resumes after network or API interruption
- Runs controlled parallel batch translation for faster large-file processing
- Shows progress, batch count, elapsed time, and ETA
- Records local translation job history in SQLite
- Counts app sessions locally in the sidebar

## Safety

This folder intentionally excludes confidential runtime files:

- `app.env`
- `glossary.xlsx`
- `glossary.csv`
- uploaded documents
- translated customer/company files
- `.term1_progress/`
- `.term1_usage_count.json`
- `.term1_jobs.db`

Use only synthetic or sanitized sample files in this public repository.

## Local Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `app.env` locally:

```text
OPENAI_API_KEY=your_api_key_here
```

Add a local `glossary.xlsx` or `glossary.csv` with `JP` and `EN` columns.

Run the app:

```powershell
streamlit run upload-app.py
```
