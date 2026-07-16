# ADR-002: Design Long Document Translation as a Recoverable Job

- **Date:** 2026-05-29 to 2026-06-01
- **Status:** Accepted
- **Evidence:** Commits `c46b67c`, `8cdf8f0`, `43cda39`, and `602ff02`

## Problem

Large engineering documents can take a long time to translate. Network interruption, API failure, browser refresh, or process restart can otherwise cause complete rework.

## Options Considered

1. Process the entire document synchronously in one request.
2. Split content into batches without persistent state.
3. Use batched jobs with checkpoints, retry handling, translation reuse, and progress state.

## Decision

Treat document translation as a recoverable job. Use controlled batches, checkpoint state, resilient fallback, local job history, and duplicate translation reuse.

## Why

Recovery and progress visibility are necessary product behaviors for real engineering documents, not optional infrastructure details.

## Tradeoffs

- persistent state requires lifecycle and privacy controls
- concurrency must be limited according to server and API capacity
- checkpoints and source fingerprints require careful versioning

## Validation

The public Git history shows the incremental addition of parallel batches, resume display, background jobs, job tracking, and duplicate reuse.
