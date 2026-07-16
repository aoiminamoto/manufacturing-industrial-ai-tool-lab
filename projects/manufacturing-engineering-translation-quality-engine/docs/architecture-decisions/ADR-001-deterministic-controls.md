# ADR-001: Separate Deterministic Controls from Probabilistic Translation

- Status: Accepted
- Public record date: 2026-07-16

## Context

Language models can generate fluent text but cannot alone guarantee preservation of engineering identifiers, syntax, file structure, or display constraints.

## Decision

Place parsing, terminology injection, identifier protection, reconstruction, and validation outside the translation adapter. The model translates only eligible text fragments.

## Consequences

- Translation providers can be changed without rewriting the quality controls.
- Deterministic requirements can be tested independently.
- The pipeline adds implementation complexity, but failures become observable and reviewable.
