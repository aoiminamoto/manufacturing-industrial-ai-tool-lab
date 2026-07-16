# Engineering Contribution and Architecture Leadership Record

## Contribution thesis

Aoi Minamoto's contribution was not limited to connecting a language model or collecting translated terms. She converted floor-level manufacturing workflows into a governed translation mechanism with explicit input boundaries, output contracts, quality controls, and review evidence.

This public record demonstrates competencies associated with a manufacturing AI architect or engineering lead. It describes demonstrated work and design responsibility; it does not claim a formal job title or unsupported organizational impact.

## Demonstrated contribution areas

### 1. Floor-level requirement discovery

Recognized that engineers use different translated artifacts differently. Some outputs require Japanese and English side by side for technical review; others must retain their original structure for direct reuse.

**Evidence in this repository:** four public-safe requirement profiles and their documented floor workflows.

### 2. Profile-specific output design

Designed explicit output contracts rather than one generic translation layout:

- PLC and safety PLC tables preserve the source and place English in the adjacent right column.
- HMI text replaces eligible Japanese in place.
- Robot programs replace only eligible comments in place while instructions remain protected.
- UI wording communicates the expected output before processing.

**Evidence in this repository:** typed `OutputContract` models, reconstruction logic, ADR-004, and automated tests.

### 3. Manufacturing AI system architecture

Separated probabilistic translation from deterministic engineering controls: parsing, terminology injection, identifier protection, reconstruction, and validation remain outside the model boundary.

**Evidence in this repository:** pipeline architecture, adapter interface, and ADR-001.

### 4. Engineering-file integrity

Treated addresses, device IDs, instructions, variables, positions, rows, columns, and encodings as protected engineering data rather than ordinary text.

**Evidence in this repository:** identifier checks, robot comment-only reconstruction, same-encoding tests, and ADR-003.

### 5. Knowledge governance

Defined approved terminology as controlled knowledge and rejected blind reuse of unreviewed translation memory.

**Evidence in this repository:** terminology status filtering, glossary-hit reporting, unknown-Japanese review gates, and ADR-002.

### 6. Quality and user trust

Made quality conditions visible through checks and warnings. The system does not silently approve untranslated Japanese or a violated HMI length constraint.

**Evidence in this repository:** deterministic `QualityCheck` results and failure-path unit tests.

### 7. Product and operational thinking

Connected technical behavior to user expectation, reviewability, recovery, evidence collection, and future measurement rather than treating model output as the end of the workflow.

**Evidence in this repository:** the evolution timeline, evidence-status section, and output-contract documentation.

### 8. Secure knowledge transfer

Created a clean-room public reference that communicates reusable architecture without publishing company identities, logos, production files, real terminology sources, credentials, endpoints, or proprietary code.

**Evidence in this repository:** synthetic fixtures and the security/IP boundary.

## Lead-level competency map

| Competency | Demonstrated behavior | Public evidence |
|---|---|---|
| Requirements leadership | Converted floor workflows into testable software contracts | Output-contract table and ADR-004 |
| Architecture ownership | Defined model/control boundaries and reusable stages | Architecture and ADR-001 |
| Cross-domain engineering | Connected AI behavior with PLC, HMI, robot, and file constraints | Requirement profiles and tests |
| Quality governance | Defined approval, warning, and validation mechanisms | Terminology controller and quality checks |
| User-centered design | Disclosed expected output before execution | User-visible output-contract design |
| Risk management | Protected syntax, identifiers, encoding, and confidential material | Reconstruction tests and IP boundary |
| Scalability of knowledge | Abstracted one project into reusable public-safe patterns | Clean-room package and ADR set |

## Evidence discipline

The repository supports the existence and design of the public reference implementation, its automated tests, its architecture decisions, and the documented contribution narrative. Adoption, time savings, business impact, organizational criticality, and reuse across sites require separately retained and independently supportable evidence. Until available, those metrics remain `Not yet measured` or `Not yet attached`.
