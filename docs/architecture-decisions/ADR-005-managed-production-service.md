# ADR-005: Move from Shared Pilot Hosting to a Managed Production Service

- **Date documented:** 2026-07-15
- **Status:** Proposed roadmap decision

## Problem

A shared pilot server can validate technical value, but it does not establish capacity, security, availability, support ownership, or scalability for sustained daily engineering use.

## Options Considered

1. Continue indefinitely on one shared host and one application process.
2. Increase the shared server size without changing the operating model.
3. Move to an IT-managed service with identity, queued work, observability, backup, and staged scaling.

## Decision

Use the shared environment only as a controlled pilot. Before broad rollout, complete capacity and security validation and establish a managed production operating model.

## Target Capabilities

- enterprise authentication and role-based access
- separate interactive sessions and queued translation workers
- managed secrets, logs, metrics, backup, and recovery
- defined support and release ownership
- multiple instances when measured demand requires them
- governed terminology releases and approved local extensions

## Validation Required

- realistic staged load test
- API-rate-limit and failure testing
- data-retention and access review
- recovery exercise
- phased user acceptance
- documented rollout or Yokoten approval
