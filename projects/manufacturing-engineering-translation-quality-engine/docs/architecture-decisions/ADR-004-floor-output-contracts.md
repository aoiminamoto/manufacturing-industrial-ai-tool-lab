# ADR-004: Make the Output Document a User-Visible Contract

- Status: Accepted
- Public record date: 2026-07-16

## Context

A correct sentence can still produce an unusable engineering file. Floor-level users may need bilingual columns for review or an in-place translated artifact for direct reuse. If placement behavior is hidden until download, users cannot select the appropriate workflow with confidence.

## Decision

Define output placement by engineering profile and disclose it in the UI before processing:

- PLC and safety PLC comment tables preserve the source and write English to the adjacent right column.
- HMI text replaces eligible Japanese in place while preserving the display-file structure.
- Robot programs replace only eligible Japanese comments in place while protecting instructions and syntax.

Represent the same behavior as a typed `OutputContract` in code and verify it with automated tests.

## Consequences

- Output documents match review or reuse needs instead of applying one generic layout.
- Users understand the expected result before starting a job.
- UI descriptions, reconstruction logic, and tests must remain synchronized.
- Each future file profile must explicitly define its output placement and floor workflow.
