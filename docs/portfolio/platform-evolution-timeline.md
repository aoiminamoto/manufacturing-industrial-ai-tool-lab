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

## Phase 7 - Production Hardening and Governed-Knowledge Transparency

**Work date:** July 17, 2026

**Public documentation date:** July 17, 2026

**Engineering focus:** Make shared-host operation measurable, diagnosable, and reviewable without exposing user content or private infrastructure.

**Contribution**

- added transactional aggregate workflow counters for text, document, and image/HMI translation starts
- preserved the existing overall-use metric while separating feature-level adoption signals
- avoided storing user identity, source text, translated text, or uploaded content in the aggregate metrics
- exposed complete validation, approval, and category metadata for controlled terms actually used in translation
- consolidated repeated term use into one knowledge-review row with an application count
- hardened API connectivity through operating-system certificate trust and proxy-aware startup
- classified connection failures into actionable certificate, timeout, authentication, API-status, and network-path categories
- identified interactive-session hosting and duplicate supervisors as operational risks requiring an IT-managed service boundary

**Private validation retained**

- concurrent counter-update test
- JP-to-EN and EN-to-JP complete-metadata regression tests
- syntax and application-start checks
- certificate, proxy-path, and authentication-layer connectivity tests
- supervised stop/start and port-listener verification

**Evidence boundary**

- public: sanitized decision record, architecture, case study, timeline, and validation summary
- private: production source, runtime logs, aggregate counts, controlled glossary, reviewer identities, host configuration, and operational screenshots
- not yet claimed: 24/7 availability, managed-cloud deployment, authenticated enterprise access, capacity limit, or cross-plant rollout

**Public evidence**

- [Controlled-terminology transparency decision](../architecture-decisions/ADR-007-controlled-terminology-transparency.md)
- [July 17 production-hardening evidence](2026-07-17-production-hardening.md)
- [High-level platform architecture](../architecture/manufacturing-ai-translation-platform.md)
- [Production-readiness roadmap](../runbooks/manufacturing-ai-platform-production-readiness.md)

## Phase 8 - PowerPoint Semantic Integrity and Reviewable AI Controls

**Work date:** August 10, 2026

**Public documentation date:** August 10, 2026

**Engineering focus:** Correct a document-segmentation failure discovered through realistic testing, then prevent fluent but semantically incomplete PowerPoint output from passing silently.

**Discovery and contribution**

- observed that one sentence split by PowerPoint paragraphs was translated as unrelated fragments
- tested an intermediate context-sharing design and identified that paragraph-by-paragraph output still imposed an invalid Japanese-to-English alignment
- redesigned extraction and reconstruction around the complete PowerPoint text box as the semantic unit
- preserved genuinely separate list items while allowing continuous prose to reorder naturally
- detected a later candidate that omitted an explicit actor despite being fluent and concise
- added deterministic semantic coverage checks and correction retries for selected actors, conditions, actions, negation, identifiers, numbers, units, glossary requirements, and list counts
- pinned the public prototype's default model snapshot, lowered temperature to zero, and versioned PowerPoint checkpoints to reduce avoidable inconsistency
- preserved explicit user-selected content profiles for PLC/SPLC, supplier email, PowerPoint, product catalog, robot program, and general plant content
- extended document preview so every matched term exposes every available public-safe glossary column

**Public validation**

- six synthetic regression tests covering text-box extraction, whole-sentence reconstruction, list preservation, actor omission, negation/glossary coverage, and complete glossary-column visibility
- syntax validation of the public-safe prototype and tests
- no real terminology, production presentation, identity, infrastructure detail, or credential committed

**Evidence boundary**

- publicly demonstrated: architecture decision, implementation pattern, synthetic tests, learning record, and traceability design
- privately retain when permitted: dated user-test notes, non-public test outputs, and operational change record
- not yet claimed: measured production-quality gain, zero omission rate, deterministic LLM output, independent adoption, or safety validation

**Public evidence**

- [ADR-008: PowerPoint semantic units and quality gates](../architecture-decisions/ADR-008-powerpoint-semantic-units-and-quality-gates.md)
- [August 10 engineering learning record](2026-08-10-powerpoint-semantic-quality-learning-record.md)
- [Public-safe prototype and synthetic tests](../../projects/manufacturing-ai-translation-platform/apps/term1-glossary-controlled-translator/)

## Next Verifiable Milestones

Future milestones should be added only after evidence exists:

- controlled capacity-test report
- measured translation quality baseline
- documented engineering-review feedback cycle
- IT-managed deployment decision
- phased user rollout
- verified reuse by another team or plant
