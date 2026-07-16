# ADR-003: Separate HMI Detection Quality from Translation Quality

- **Date documented:** 2026-07-15
- **Status:** Accepted at architecture level
- **Public implementation boundary:** Organization-specific vision integration is excluded

## Problem

An HMI translation can fail before translation begins. Incorrect physical-cell detection, merged labels, missed text, or OCR errors produce incorrect source blocks that a glossary cannot reliably repair.

## Options Considered

1. Ask a generative model to redraw the complete English HMI in one step.
2. Detect arbitrary text regions and translate each region without structural validation.
3. Separate physical-cell detection, OCR, terminology control, contextual translation, engineering review, and controlled reconstruction.

## Decision

Use a staged HMI workflow and measure each quality layer independently. Preserve the original layout and replace only reviewed text regions when generating an English preview.

## Why

Free-form regeneration may look attractive while duplicating, omitting, or moving engineering labels. Controlled reconstruction preserves traceability.

## Tradeoffs

- staged processing is more complex
- cell detection requires specialized validation
- long English labels require layout and font-fitting rules

## Validation

The public architecture and case study record the decision. Future public evidence should use synthetic HMI examples and measured detection/acceptance metrics.
