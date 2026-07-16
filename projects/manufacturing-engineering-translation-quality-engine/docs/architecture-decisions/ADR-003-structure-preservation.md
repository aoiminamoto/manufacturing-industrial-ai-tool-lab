# ADR-003: Reconstruct the Original Engineering Structure

- Status: Accepted
- Public record date: 2026-07-16

## Context

Program files, PLC exports, HMI labels, and spreadsheets contain both translatable text and non-translatable engineering structure. Rewriting the full input can corrupt a valid artifact.

## Decision

Parse eligible regions, translate them independently, and reconstruct the original format. For the public robot-program example, only semicolon comments are translated and the selected source encoding is preserved.

## Consequences

- Engineering syntax remains outside the model boundary.
- Each file category requires a specific parser and validation profile.
- Format support must be added deliberately rather than treated as generic text processing.
