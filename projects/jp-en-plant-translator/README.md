# Manufacturing AI Translation Platform

A public-safe engineering portfolio project for governed Japanese-to-English translation in manufacturing environments.

The platform is designed as a **translation shikumi**: a repeatable mechanism that combines controlled terminology, manufacturing standards, AI translation, engineering review, recovery controls, and continuous improvement. The objective is not generic translation or word collection. It is consistent plant-language governance that can be extended through Yokoten.

> Public-repository boundary: this project uses sanitized architecture descriptions, synthetic examples, and a public-safe prototype. It does not contain organization logos, real controlled terminology, production documents, credentials, private endpoints, or internal deployment configuration.

## Executive View

```text
Text | Documents | HMI / Images
                ↓
    Manufacturing AI Translation Platform
                ↕
Controlled Glossary + Abbreviation Standard
     + Comment Standard + Pattern Rules
                ↕
        Enterprise AI Translation
                ↓
Controlled English | Reviewable Files | HMI Review Map
                ↓
Engineering Review → Knowledge Improvement → Yokoten
```

## Engineering Problem

Generic translation tools often lose manufacturing meaning because plant language depends on equipment context, PLC/HMI conventions, abbreviations, product terminology, and locally approved wording. Large technical files add operational challenges such as interruption recovery, duplicate work, encoding preservation, and review traceability.

This project addresses those constraints as a system-design problem rather than a prompt-only problem.

## Platform Capabilities

The public-safe runnable prototype demonstrates:

- glossary-controlled Japanese-to-English translation
- manufacturing-specific translation modes
- Japanese-only filtering for mixed-language content
- TXT, CSV, DOCX, XLSX, XLSM, and robot-program processing
- large-file batching with controlled parallel execution
- checkpoint recovery and resumable document jobs
- local translation memory and job tracking
- retry handling, progress reporting, and usage counting
- protected handling for technical codes and structured content

The broader platform architecture also covers an HMI/image workflow:

1. detect Japanese regions and physical HMI cells
2. read each region with visual context
3. apply controlled terminology and translation standards
4. generate a numbered engineering review map
5. capture engineering corrections for governed knowledge improvement

The HMI workflow is documented as an architecture and case-study artifact. Organization-specific vision integration and production assets are intentionally excluded from the public prototype.

## Controlled Knowledge System

The architecture separates four knowledge controls:

| Control | Purpose |
|---|---|
| Controlled Glossary | Approved Japanese-English manufacturing terminology |
| Abbreviation Standard | Consistent plant and controls abbreviations |
| Comment Standard | Standard PLC, HMI, alarm, and manufacturing wording |
| Pattern Rules | Repeatable structures with variable equipment numbers or codes |

These controls surround the AI model. The model provides contextual reasoning; the governed knowledge system provides consistency, ownership, and repeatability.

## Architecture and Leadership Artifacts

- [High-level platform architecture](../../docs/architecture/manufacturing-ai-translation-platform.md)
- [Platform evolution case study](../../docs/case-studies/manufacturing-ai-translation-platform-evolution.md)
- [Production-readiness roadmap](../../docs/runbooks/manufacturing-ai-platform-production-readiness.md)
- [Evolution timeline and evidence framework](../../docs/portfolio/)
- [Architecture Decision Records](../../docs/architecture-decisions/)
- [Glossary update runbook](../../docs/runbooks/glossary-update-runbook.md)
- [Runnable public-safe prototype](apps/term1-glossary-controlled-translator/)

## Technical-Leadership Scope Demonstrated

- translated a plant-language problem into a governed AI system architecture
- defined boundaries between deterministic controls and generative reasoning
- designed resilient processing for large engineering documents
- introduced reviewability, recovery, job tracking, and usage measurement
- identified HMI OCR/segmentation as a separate quality layer from translation
- evaluated shared-server pilot constraints and defined a production-scaling path
- established a public/private separation model for responsible portfolio evidence

## Long-Term Direction

The target state is an IT-managed enterprise service with authentication, role-based access, queued workloads, observability, controlled knowledge ownership, capacity validation, backup, and phased rollout. This makes the platform suitable for daily engineering use and future plant-to-plant Yokoten without depending on one user account or one shared PC.

## Data and IP Policy

Only synthetic, sanitized, or personally created examples belong in this repository. Do not commit:

- organization logos or branded UI assets
- real glossary or standards files
- production screenshots or operating documents
- API keys, tokens, endpoints, or credentials
- customer, supplier, employee, or plant data
- local job databases, translation memory, checkpoints, or usage files
- internal server names, IP addresses, or deployment details
