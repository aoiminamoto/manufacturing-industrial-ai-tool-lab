# Manufacturing AI Translation Platform Evolution Timeline

This timeline is based on the repository's public Git history. It documents the evolution of the engineering work without inventing historical dates or private adoption claims.

## Phase 1 - Governed Translation Foundation

**Public period:** May 29, 2026

**Engineering focus:** Move from generic translation toward glossary-controlled manufacturing translation.

**Contribution**

- introduced the public-safe glossary-controlled Streamlit translator
- supported text and CSV document translation
- added progress feedback and more tolerant CSV handling
- established the foundation for resumable document processing

**Public evidence**

- [`2072ca3`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/2072ca3) - Add Term1 glossary-controlled translator app
- [`08542c6`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/08542c6) - Support CSV document translation
- [`11471a3`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/11471a3) - Handle irregular CSV document rows
- [`8cdf8f0`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/8cdf8f0) - Clarify resumed translation progress display

## Phase 2 - Performance, Recovery, and Operability

**Public period:** May 29-June 1, 2026

**Engineering focus:** Make long-running engineering translation usable and recoverable.

**Contribution**

- introduced controlled parallel document batches
- added timing and token-usage visibility
- added local job history and a job-details dashboard
- added background document jobs and duplicate translation reuse
- expanded the workflow to robot-program and PLC-comment contexts

**Public evidence**

- [`c46b67c`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/c46b67c) - Run document translation batches in parallel
- [`6ef1514`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/6ef1514) - Add AS support and local job history
- [`a9fb177`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/a9fb177) - Add translation job details dashboard
- [`602ff02`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/602ff02) - Reuse duplicate document translations
- [`43cda39`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/43cda39) - Run document translations as background jobs
- [`d01f004`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/d01f004) - Add PLC comment translation mode

## Phase 3 - Resilience and Engineering User Experience

**Public period:** June 8-10, 2026

**Engineering focus:** Improve failure handling, progress clarity, and daily engineering usability.

**Contribution**

- added resilient batch-translation fallback
- stabilized background progress behavior
- simplified progress, ETA, upload, and knowledge-panel presentation
- clarified supported upload types and model visibility

**Public evidence**

- [`0bb9537`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/0bb9537) - Add resilient Term1 batch translation fallback
- [`b7d8271`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/b7d8271) - Stabilize Term1 document progress display
- [`5124499`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/5124499) - Show Term1 supported upload types

## Phase 4 - Public-Safe Portfolio Integration

**Public period:** June 29, 2026

**Engineering focus:** Separate a demonstrable public artifact from private runtime data and controlled knowledge.

**Contribution**

- integrated the sanitized runnable application into the industrial AI portfolio repository
- documented repository safety boundaries
- excluded credentials, runtime state, real terminology, and operational documents

**Public evidence**

- [`ef92c72`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/ef92c72) - Update Term1 glossary-controlled translator app
- [`f773fab`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/f773fab) - Update Term1 translator README for public portfolio
- [Pull request #9](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/pull/9)

## Phase 5 - Manufacturing AI Platform Architecture

**Public period:** July 15, 2026

**Engineering focus:** Document the transition from a translator prototype to a governed industrial AI platform.

**Contribution**

- defined the four-part controlled-knowledge architecture
- documented multimodal text, document, and HMI/image workflow boundaries
- separated OCR/segmentation quality from translation quality
- documented the pilot-to-production scaling path
- created a public evidence framework for architecture leadership and continuous improvement

**Public evidence**

- [`1197a66`](https://github.com/aoiminamoto/manufacturing-industrial-ai-tool-lab/commit/1197a66) - Document manufacturing AI platform evolution
- [High-level architecture](../architecture/manufacturing-ai-translation-platform.md)
- [Platform evolution case study](../case-studies/manufacturing-ai-translation-platform-evolution.md)
- [Production-readiness roadmap](../runbooks/manufacturing-ai-platform-production-readiness.md)

## Phase 6 - Bidirectional Documents and Shared-Session Isolation

**Work date:** July 16, 2026

**Public documentation date:** July 16, 2026

**Engineering focus:** Extend document coverage while correcting shared-state behavior for concurrent pilot users.

**Contribution**

- introduced explicit JP-to-EN and EN-to-JP direction selection for text and document workflows
- added PowerPoint processing and presentation-oriented translation guidance
- assigned each persisted document job to an opaque browser-session owner
- scoped job discovery, stop/retry actions, previews, results, and checkpoint files to that owner
- validated two-session isolation, same-name file separation, database migration, and UI startup behavior
- clarified in-product file-type and upload-limit guidance

**Evidence boundary**

- public: sanitized architecture, ADR, case study, timeline, and PPTX-capable public-safe prototype
- private: implementation diff, regression outputs, operational screenshots, and deployment record
- excluded: organization branding, internal URLs, controlled terminology, production documents, and runtime databases

**Public evidence**

- [Browser-session task-isolation decision](../architecture-decisions/ADR-006-browser-session-task-isolation.md)
- [High-level platform architecture](../architecture/manufacturing-ai-translation-platform.md)
- [Platform evolution case study](../case-studies/manufacturing-ai-translation-platform-evolution.md)

## Next Verifiable Milestones

Future milestones should be added only after evidence exists:

- controlled capacity-test report
- measured translation quality baseline
- documented engineering-review feedback cycle
- IT-managed deployment decision
- phased user rollout
- verified reuse by another team or plant
