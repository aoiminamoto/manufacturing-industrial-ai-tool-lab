# ADR-001: Apply Controlled Manufacturing Knowledge Around AI Translation

- **Date:** 2026-05-29
- **Status:** Accepted
- **Evidence:** Public prototype commit `2072ca3`

## Problem

Generic translation can produce fluent English while changing approved manufacturing terminology or using inconsistent wording across repeated plant content.

## Options Considered

1. Use a generic translation prompt without terminology control.
2. Replace all glossary terms mechanically before translation.
3. Combine controlled terminology with contextual AI reasoning.

## Decision

Use the controlled glossary as a governed constraint while retaining contextual AI translation for surrounding language. Protect technical codes and preserve reviewable source-to-output relationships.

## Why

This balances deterministic consistency with contextual language quality. It also creates a foundation for adding abbreviation, comment, and pattern-rule controls.

## Tradeoffs

- glossary quality directly affects output quality
- exact matches may require context-specific exceptions
- terminology ownership and approval become operational responsibilities

## Validation

The public prototype and subsequent translation-mode commits demonstrate the implementation pattern. Quality metrics and reviewer acceptance should be recorded separately.
