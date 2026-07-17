# ADR-007: Expose Complete Controlled-Term Records at the Review Surface

- **Date documented:** 2026-07-17
- **Status:** Implemented in the private shared-host application; documented publicly with synthetic-safe boundaries

## Problem

The translation pipeline applied governed terminology internally, but a summary count did not allow an engineer to see which controlled knowledge influenced the result. Showing only a source term, required target term, and count also omitted the validation, approval, category, and governance context maintained with the controlled record.

This created an explainability gap. The model output was visible, but the governed engineering decision behind it was not.

## Options Considered

1. Keep controlled terminology hidden and display only a match count.
2. Display only source term, target term, and count.
3. Display every source-region match, including repeated rows.
4. Display one complete governed record per controlled term used, plus an aggregate application count.

## Decision

Use option 4.

For text and image/HMI review:

- identify controlled terms actually applied to the translation input
- retrieve the complete governed record available to the application
- preserve the source glossary's review metadata columns
- display one row per unique controlled term
- aggregate repeated use into a `Used Count`
- support the same evidence shape for both translation directions

The display is a review aid, not a public release of the glossary. Production-controlled records remain inside the authorized application environment.

## Rationale

This design gives engineering users the information needed to understand and reuse approved terminology without creating a noisy row for every repeated image region. It also makes knowledge governance visible as a product capability rather than leaving it as undocumented prompt context.

## Validation

- complete glossary metadata columns remain available after normalization
- a term used in multiple regions produces one display row
- repeated use increments the aggregate application count
- JP-to-EN and EN-to-JP reporting use the same controlled record
- no-match input produces an explicit empty-state message

## Consequences

### Positive

- engineers can verify which governed knowledge shaped a translation
- validation and approval context is available at the point of review
- repeated terminology becomes a knowledge-reuse signal
- the application supports team learning without requiring a separate glossary lookup

### Limitations

- terminology visibility cannot compensate for OCR text that was missed or recognized incorrectly
- authorization is still required because validation and approval metadata may identify internal reviewers
- a visible controlled record does not prove that every free-form phrase is correct
- production screenshots and controlled records remain private evidence

## Production Follow-Up

Add authenticated role-based access, glossary-release versioning, correction feedback, audit logging, and quality metrics that distinguish OCR recall from terminology compliance.
