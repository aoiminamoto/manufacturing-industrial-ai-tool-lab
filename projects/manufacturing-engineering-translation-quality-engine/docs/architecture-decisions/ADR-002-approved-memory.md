# ADR-002: Only Approved Terminology May Act as Memory

- Status: Accepted
- Public record date: 2026-07-16

## Context

Automatically reusing every previous translation can propagate an incorrect result. A prior match is evidence, not necessarily truth.

## Decision

Only entries with an explicit `approved` status may be injected deterministically. Draft or rejected mappings remain outside the automatic path. Unknown Japanese fails the review gate instead of being silently accepted.

## Consequences

- Human approval becomes part of terminology governance.
- Corrections can be traced to controlled entries.
- Coverage grows more slowly than automatic memory, but quality risk is lower and easier to audit.
