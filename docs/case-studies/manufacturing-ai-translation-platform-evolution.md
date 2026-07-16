# Case Study: From Translation Prototype to Manufacturing AI Shikumi

## Challenge

Manufacturing translation quality depends on more than language fluency. Engineers must preserve equipment terminology, PLC/HMI conventions, abbreviations, codes, document structure, and locally approved wording. Long documents and HMI screenshots introduce additional reliability and review challenges.

The initial question was how to translate Japanese plant content. The engineering challenge became broader: how to create a governed mechanism that produces consistent, reviewable output and improves over time.

## Architecture Leadership

The platform direction was shaped around five responsibilities:

1. **Knowledge control** — separate glossary, abbreviation, comment, and pattern-rule layers.
2. **Contextual AI translation** — use enterprise AI reasoning while preserving controlled terms and protected codes.
3. **Multiformat processing** — support text, structured documents, spreadsheets, robot programs, and HMI/image review workflows.
4. **Operational resilience** — introduce batching, checkpoint resume, retry handling, job state, translation memory, and usage tracking.
5. **Governance and Yokoten** — capture approved engineering feedback so improvements can be reused across teams and plants.

## Key Engineering Lessons

### Glossary matching is necessary but not sufficient

A glossary controls known terminology, but it cannot correct incorrect OCR, merged HMI cells, missing context, or ambiguous source text. Quality must be managed across the complete pipeline.

### A visually attractive AI result is not automatically engineering-safe

Free-form image regeneration may look polished while duplicating rows, dropping labels, or changing layout. The safer pattern is controlled reconstruction: preserve the source layout and replace only reviewed text regions.

### Reliability features are product features

Checkpoint recovery, resumable jobs, structured error handling, and progress visibility are essential when engineers process large files. They turn a demonstration into an operable workflow.

### Adoption requires production ownership

A shared-server pilot can validate value, but sustained daily use requires IT-managed infrastructure, authentication, monitoring, backup, capacity testing, support ownership, and a phased rollout plan.

### Shared applications need an explicit ownership boundary

Background-job persistence introduced a multi-user failure mode: a globally selected "latest job" could expose another user's file name, progress, or result in the shared UI. The correction was architectural rather than cosmetic. Job creation, queries, stop controls, retry behavior, result retrieval, and checkpoint paths were scoped to one opaque browser-session owner.

This was validated with two simulated users, including same-name document submission and owner-scoped stop-all behavior. The design improves shared-pilot privacy while preserving an honest boundary: an anonymous browser token is not enterprise authentication.

### Bidirectional and presentation workflows require explicit product contracts

The platform added direction selection before text or document translation and extended document handling to PowerPoint presentations. Translation direction, slide-oriented wording, supported formats, and file-size limits are presented before execution so engineers can understand the output contract instead of discovering behavior after a long-running job.

## Current Public-Safe Evidence

This repository provides:

- a runnable sanitized Streamlit prototype
- source code for glossary-controlled document translation patterns
- architecture documentation for multimodal translation and governance
- a glossary maintenance runbook
- a production-readiness roadmap
- explicit data/IP boundaries for responsible public disclosure

No organization logos, proprietary terminology files, production documents, private endpoints, credentials, or internal screenshots are included.

## Impact Model

The platform is designed to create measurable value through:

- terminology consistency across repeated engineering content
- reduced manual retranslation through translation memory
- reduced restart cost through checkpoint recovery
- faster engineering review through source-to-output mapping
- controlled adoption measurement through usage and job metrics
- reusable standards that support Yokoten

Specific organizational metrics should be validated and documented privately before being used as evidence. This public case study intentionally avoids unsupported adoption or productivity claims.

## Leadership Narrative

The work demonstrates an industrial AI architecture role: converting an ambiguous plant problem into a governed software platform, defining the knowledge and system boundaries, implementing a resilient prototype, evaluating production constraints, and creating a path from local pilot to enterprise service.
