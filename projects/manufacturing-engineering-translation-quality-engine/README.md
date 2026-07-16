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

- Requirement profiles for PLC, robot, HMI, and structured-file outputs
- Translation of Japanese fragments without rewriting surrounding English
- Approved-term injection before model translation
- Preservation checks for addresses, identifiers, model codes, and robot instructions
- Same-encoding reconstruction for robot-program bytes
- Human-readable glossary-hit and quality-check evidence
- A translator adapter boundary: the public demo requires no external API
- Unit tests using only synthetic examples

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
- [ADR-001: Separate deterministic controls from probabilistic translation](docs/architecture-decisions/ADR-001-deterministic-controls.md)
- [ADR-002: Only approved terminology may act as memory](docs/architecture-decisions/ADR-002-approved-memory.md)
- [ADR-003: Reconstruct the original engineering structure](docs/architecture-decisions/ADR-003-structure-preservation.md)

## Evidence status

This repository provides dated source history after publication, tested reference code, architecture decisions, and a retrospective engineering timeline. Production adoption, quality improvement, time savings, and business impact are **not claimed here unless supported by separately retained evidence**. Current public metrics are marked `Not yet measured`.

## Public-data policy

All examples are synthetic. This project contains no company names, logos, internal terminology files, production files, credentials, internal endpoints, server details, or proprietary source code.

## Authorship

Architecture, engineering requirements, implementation direction, validation strategy, and public portfolio documentation: **Aoi Minamoto**.
