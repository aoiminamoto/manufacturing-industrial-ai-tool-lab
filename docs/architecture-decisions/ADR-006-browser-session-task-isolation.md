# ADR-006: Scope Shared-Pilot Document Jobs to a Browser Session

- **Date documented:** 2026-07-16
- **Status:** Implemented pilot safeguard; enterprise identity remains required

## Problem

Persisted background jobs were stored in one shared database. If the UI selected the globally latest job, concurrent users could see or operate on another user's task through the application interface. Identical file names could also share checkpoint state.

## Options Considered

1. Keep global job visibility and rely on users not to overlap.
2. Store only the current job in framework session state.
3. Assign persisted jobs and checkpoints to an opaque browser-session owner.
4. Require enterprise authentication before any shared pilot use.

## Decision

For the controlled pilot, assign each browser an opaque session identifier and persist that owner with every document job. Apply the owner condition consistently to job discovery, details, counts, stop actions, retry/continue behavior, previews, downloads, and checkpoint paths.

Do not describe this identifier as a user identity. It separates anonymous browser workspaces but does not authenticate a person or authorize access at an enterprise level.

## Validation

- two independently initialized browser sessions receive different identifiers
- one session cannot retrieve another session's job detail
- stop-all affects only the requesting session's active jobs
- identical file names do not cause cross-session cancellation
- identical document content produces separate checkpoint paths per owner
- existing job databases migrate without manual recreation

## Consequences

### Positive

- concurrent pilot users no longer see each other's document tasks in the UI
- task controls cannot stop another browser session's work
- refresh continuity is preserved for the same session
- the ownership rule is explicit and testable across the job lifecycle

### Limitations

- possession of a complete session URL can reproduce that anonymous session
- server administrators can still access server-side runtime files
- browser-session ownership does not provide employee identity, roles, audit, or policy enforcement
- legacy global batch artifacts must not be exposed in the multi-user UI

## Production Follow-Up

Replace anonymous session ownership with enterprise identity and authorization. Add retention policy, audit logging, encrypted storage, managed worker queues, and security review before broad daily use.
