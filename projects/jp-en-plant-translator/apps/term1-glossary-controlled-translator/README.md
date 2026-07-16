# Manufacturing AI Translation Platform — Public-Safe Prototype

Public-safe Streamlit application for Japanese-to-English manufacturing translation with glossary control, resilient document processing, and domain-specific translation modes. The folder retains its original prototype codename for repository continuity; public documentation uses **Manufacturing AI Translation Platform**.

This app demonstrates an engineering approach to AI-assisted translation for plant-floor and manufacturing documents. Instead of treating translation as generic text conversion, it combines approved terminology, manufacturing context, batch processing, checkpoint recovery, and local job tracking so technical meaning stays consistent across large files.

## Engineering Purpose

Manufacturing translation often fails when generic language tools do not understand equipment names, PLC comments, supplier wording, product descriptions, or robot-program context. This project shows how an engineer can build a governed translation workflow around those constraints:

- Preserve approved technical terminology through a local glossary
- Translate only Japanese content when processing mixed-language files
- Support large Word, Excel, text, CSV, and robot-program files
- Resume work after interruption instead of restarting long translations
- Track translation jobs locally without sending runtime history to the repository
- Separate public-safe source code from private runtime files and company data

## Key Capabilities

- Japanese-to-English manufacturing translation
- Glossary-controlled terminology enforcement
- Translation modes for:
  - General plant translation
  - PLC/SPLC comment standardization
  - Supplier email translation
  - Product catalog and specification translation
  - Kawasaki robot `.as` / `.ad` file comments and labels
- TXT, CSV, DOCX, XLSX, XLSM, AS, and AD upload support
- Large-file processing up to 100 MB
- Japanese-only cell filtering for Excel files
- Controlled parallel batch translation for faster long documents
- Per-document checkpoint files for resume support
- Local SQLite job history
- Local translation memory cache
- Optional email notification or manual email draft support
- Local app usage count in the sidebar

## Public Repository Safety

This repository intentionally excludes private runtime data and real operating documents. It is designed to show the application architecture and engineering implementation without exposing confidential information.

Organization logos and branded production UI assets are also intentionally excluded.

Do not commit:

- `app.env`
- `glossary.xlsx`
- `glossary.csv`
- `translation_memory.sqlite`
- uploaded source documents
- translated output documents
- company, customer, supplier, or production files
- `.term1_progress/`
- `.term1_job_storage/`
- `.term1_usage_count.json`
- `.term1_jobs.db`

Use synthetic, sanitized, or personally created sample files only.

## Local Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `app.env` locally:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional email notification settings:

```text
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_username
SMTP_PASSWORD=your_password
SMTP_FROM=your_sender@example.com
SMTP_TLS=true
```

Add a local glossary file with approved terminology:

```text
glossary.xlsx
```

or:

```text
glossary.csv
```

The glossary should include Japanese and English term columns such as `JP` and `EN`.

Optional PLC abbreviation rules can be added as:

```text
plc_abbreviation_rules.csv
```

Run the app:

```powershell
streamlit run upload-app.py
```

## Notes

This app is a software engineering artifact for demonstrating AI workflow design in manufacturing contexts. The public version is intentionally separated from any private deployment configuration, real glossary content, operational documents, or organization-specific data.
