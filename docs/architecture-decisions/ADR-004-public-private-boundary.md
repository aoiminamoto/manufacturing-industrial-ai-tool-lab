# ADR-004: Maintain a Public-Safe Portfolio Boundary

- **Date:** 2026-06-29; expanded 2026-07-15
- **Status:** Accepted
- **Evidence:** Public repository safety documentation and PR #9

## Problem

Industrial AI work can demonstrate valuable architecture and engineering skills while also involving controlled terminology, production documents, infrastructure details, credentials, or branded assets that do not belong in a public repository.

## Decision

Maintain a sanitized public prototype and documentation layer separate from private runtime data, organization-specific integrations, and operational evidence.

## Public Content

- generic architecture
- sanitized source patterns
- synthetic examples
- public Git history
- evidence and measurement templates

## Private Content

- real glossary and standards files
- production documents and screenshots
- organization logos and branded UI
- credentials, endpoints, server details, and user identities
- signed confirmations and internal deployment records unless publication is approved

## Why

This preserves evidence of individual engineering contribution without exposing confidential information or creating unsupported public claims.
