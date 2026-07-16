# Manufacturing Engineering Translation Quality Engine

**Controlled Translation for Requirement-Specific Engineering Outputs**

This project is a clean-room, public-safe reference implementation of an engineering translation quality layer. It demonstrates how Japanese-to-English AI translation can be made more dependable for PLC comments, robot-program comments, HMI labels, and structured engineering files.

It is intentionally separate from the repository's [Manufacturing AI Translation Platform](../jp-en-plant-translator/):

- The **platform** demonstrates end-user workflows, governed knowledge, document and image processing, review, and deployment thinking.
- The **quality engine** demonstrates specialized parsing, terminology control, protected technical identifiers, requirement profiles, structure-preserving reconstruction, and deterministic validation.

## Why this project exists

Generic translation can produce fluent English while still damaging engineering meaning or file integrity. Manufacturing outputs have different constraints: a PLC address must not change, a robot instruction must not be translated, an HMI label must fit, and previously approved terminology must be applied consistently.

The engine treats these requirements as software controls around an interchangeable translation model:

```text
Engineering Input
        |
Encoding and Structure Parsing
        |
Japanese Fragment Detection
        |
Controlled Terminology + Identifier Protection
        |
Translation Adapter
        |
Format-Specific Reconstruction
        |
Deterministic Quality Validation
        |
Requirement-Specific Output + Review Evidence
```

## Demonstrated capabilities

- Requirement profiles for PLC, safety PLC, robot, HMI, and structured-file outputs
- User-visible output contracts designed around floor-level engineering workflows
- Translation of Japanese fragments without rewriting surrounding English
- Approved-term injection before model translation
- Preservation checks for addresses, identifiers, model codes, and robot instructions
- Same-encoding reconstruction for robot-program bytes
- Human-readable glossary-hit and quality-check evidence
- A translator adapter boundary: the public demo requires no external API
- Unit tests using only synthetic examples

## Plant-floor output contracts

The output document is part of the engineering requirement, not an afterthought. The UI should tell users what to expect before processing, and reconstruction should enforce the same promise.

| Public-safe profile | Output behavior | Floor-level need addressed |
|---|---|---|
| PLC comment table | Preserve Japanese; place English in the adjacent right column | Side-by-side technical review and correction |
| Safety PLC comment table | Preserve Japanese; place English in the adjacent right column | Traceable review of safety-related wording |
| HMI text | Replace eligible Japanese in place | Reuse the translated file in its original display structure |
| Robot program | Replace only eligible Japanese comments in place | Reuse the program while protecting instructions and syntax |

This profile-specific design demonstrates the conversion of manufacturing-user needs into explicit software behavior: review-oriented formats remain bilingual, while reuse-oriented formats receive structure-preserving in-place output.

## Quick start

Python 3.10 or later is sufficient; the demo has no third-party dependencies.

```bash
cd projects/manufacturing-engineering-translation-quality-engine
python -m unittest discover -s tests -v
python demo.py
```

Expected demo behavior:

- Existing English and technical IDs remain unchanged.
- Approved terminology is inserted deterministically.
- Unknown Japanese is reported for review instead of silently treated as correct.
- Robot instructions remain byte-for-byte equivalent after decoding while only comment text changes.

## Architecture and engineering record

- [High-level architecture](docs/architecture.md)
- [Retrospective evolution timeline](docs/evolution-timeline.md)
- [Public-safe engineering case study](docs/case-study.md)
- [Security and intellectual-property boundary](docs/security-and-ip-boundary.md)
- [Engineering contribution and lead-level competencies](docs/contribution-and-leadership.md)
- [ADR-001: Separate deterministic controls from probabilistic translation](docs/architecture-decisions/ADR-001-deterministic-controls.md)
- [ADR-002: Only approved terminology may act as memory](docs/architecture-decisions/ADR-002-approved-memory.md)
- [ADR-003: Reconstruct the original engineering structure](docs/architecture-decisions/ADR-003-structure-preservation.md)
- [ADR-004: Make the output document a user-visible contract](docs/architecture-decisions/ADR-004-floor-output-contracts.md)

## Evidence status

This repository provides dated source history after publication, tested reference code, architecture decisions, and a retrospective engineering timeline. Production adoption, quality improvement, time savings, and business impact are **not claimed here unless supported by separately retained evidence**. Current public metrics are marked `Not yet measured`.

## Public-data policy

All examples are synthetic. This project contains no company names, logos, internal terminology files, production files, credentials, internal endpoints, server details, or proprietary source code.

## Authorship

Architecture, manufacturing workflow requirements, profile-specific output design, implementation direction, validation strategy, and public portfolio documentation: **Aoi Minamoto**.
