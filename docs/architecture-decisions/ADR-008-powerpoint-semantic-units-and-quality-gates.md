# ADR-008: Translate PowerPoint Text Boxes as Semantic Units and Gate Semantic Loss

- **Status:** Accepted
- **Decision date:** August 10, 2026
- **Maintainer and product owner:** Aoi Minamoto
- **Scope:** Public-safe Manufacturing AI Translation Platform prototype

## Context

PowerPoint stores text according to presentation structure, not necessarily linguistic structure. A sentence can be split into several paragraphs because an author pressed Enter for visual layout. Automatic wrapping can also create multiple visible lines without creating separate paragraphs. Treating each stored paragraph as an independent translation request can therefore turn one sentence into several unrelated fragments.

A realistic synthetic test deck exposed this failure mode. The first implementation gave every paragraph the complete text-box context but still required one output per source paragraph. This improved context availability, yet the output could remain grammatically fragmented because Japanese and English do not share the same word order. A later test also showed that a fluent, concise translation could omit an explicit actor such as `作業者` even after the text box was translated as a whole.

The product requirement is stronger than fluency: concise slide wording must not remove an explicit actor, condition, action, object, negation, number, unit, protected identifier, or governed term.

## Decision Drivers

- preserve complete meaning across presentation-driven line breaks
- support natural Japanese-to-English word-order changes
- retain real list structure without treating prose lines as separate meanings
- make glossary influence visible and reviewable
- reduce avoidable output variation
- prevent known semantic omissions from being silently delivered
- keep content-type behavior explicit rather than applying one translation style to every artifact

## Options Considered

### 1. Translate every paragraph independently

This preserves paragraph mapping but loses cross-paragraph meaning. It was rejected because layout structure is not a reliable sentence boundary.

### 2. Give every paragraph full text-box context, then translate each paragraph separately

This was the first corrective design. It preserved layout more closely, but testing showed that particles, conditions, and word order could still produce fragmented English. It was rejected as the final architecture.

### 3. Translate the complete slide as one unit

This maximizes context but risks mixing unrelated text boxes, titles, labels, and diagrams. It also weakens traceability between source objects and translated objects. It was rejected.

### 4. Translate each complete text box as one semantic unit

This aligns the translation boundary with the smallest presentation object that normally carries a coherent meaning. Continuous prose is translated once and written back as one result. Clearly identified list items remain separate. This option was selected.

## Decision

1. Extract one translation block for each PowerPoint text body rather than for each paragraph.
2. Join all populated paragraphs inside the text box before translation.
3. In PowerPoint mode, instruct the translator to read the complete block before producing one meaning-preserving result.
4. Reconstruct continuous prose as one translated paragraph and preserve separate items when the source is clearly a list.
5. Version PowerPoint checkpoints so prior paragraph-level results cannot be reused accidentally.
6. Run a deterministic semantic coverage gate on PowerPoint output.
7. If the gate detects a high-confidence omission, return the missing requirements to the translator and retry.
8. If repeated attempts still fail, fail the job explicitly instead of knowingly delivering the deficient translation.
9. Use a pinned model snapshot and zero temperature as consistency controls; use translation memory for exact reuse of previously accepted source text.
10. Preserve user-selectable content profiles. PLC/SPLC comments, supplier email, PowerPoint, product catalog, robot program, and general plant content retain different output contracts.

## Semantic Quality Gate

The public-safe gate demonstrates checks for:

- explicit actor coverage, such as `作業者`
- conditional relationships, such as `場合`, `前に`, `後`, and `まで`
- common manufacturing actions, including confirm, start, stop, contact, press, open, close, and translate
- negation and prohibition
- protected identifiers
- numbers and units
- every glossary target term detected for the source block
- list-item count preservation

These checks intentionally cover high-confidence patterns rather than claiming complete Japanese semantic parsing.

## Glossary Traceability Decision

For every term matched in a translated block, the preview exposes the complete corresponding public-safe glossary record. Columns are discovered dynamically instead of being hard-coded. This means validation, approval, category, provenance, or future governance fields remain visible when present. Unmatched content remains blank, and missing metadata is never invented.

## Consequences

### Positive

- presentation line breaks no longer define translation meaning
- Japanese-to-English reordering can occur naturally within the complete sentence
- known omissions can trigger automatic correction
- glossary use is visible at the point of review
- repeated approved source text can remain stable through translation memory
- content-type intent remains a product-level user choice

### Tradeoffs

- one text box may contain genuinely unrelated paragraphs; list detection mitigates only part of this case
- rule-based semantic checks can produce false positives or miss concepts outside their coverage
- automatic retry adds latency and API cost when a candidate fails
- temperature zero and a pinned snapshot reduce variation but do not establish mathematical determinism
- engineering review remains necessary for safety-relevant or production-critical content

## Public Validation

The repository includes synthetic regression tests for:

- complete text-box extraction
- whole-sentence write-back
- list preservation
- detection of an omitted actor
- detection of lost negation and a missing required term
- dynamic display of all available glossary columns

## Evidence Boundary

Public evidence includes sanitized architecture, code, synthetic Japanese examples, tests, and this decision record. It excludes real company terminology, operational documents, user identities, infrastructure details, credentials, and production outcomes. No claim of clinical, safety, deployment, adoption, or translation-quality improvement is made without separate evidence.
