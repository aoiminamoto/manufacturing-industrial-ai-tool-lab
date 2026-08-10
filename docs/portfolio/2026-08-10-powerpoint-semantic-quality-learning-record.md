# Engineering Learning Record: From Visible Lines to Meaning-Preserving PowerPoint Translation

- **Work date:** August 10, 2026
- **Public documentation date:** August 10, 2026
- **Product owner, system builder, and design maintainer:** Aoi Minamoto
- **Evidence type:** Sanitized implementation and validation record

## Why This Record Exists

This record documents how product observation, user-centered reasoning, engineering diagnosis, implementation, and testing changed the Manufacturing AI Translation Platform. It is not a claim that one prompt solved technical translation. It shows how a product owner used concrete failures to improve the system boundary and add verifiable controls.

## Initial Observation

A PowerPoint text box contained one Japanese sentence distributed across multiple lines. The translation pipeline treated stored paragraphs as separate meanings because the code followed PowerPoint XML paragraph boundaries. From a user perspective, this was wrong: a person first reads the entire box and then understands the sentence.

The product insight was simple but consequential:

> Presentation structure is not linguistic structure. The user-visible text box, not each stored line, is the appropriate semantic boundary for this workflow.

## First Engineering Response

The first change supplied each paragraph with the complete text-box context while preserving separate paragraph write-back. This was a reasonable intermediate design because it attempted to improve meaning without disturbing layout.

Testing showed why it was insufficient. Japanese particles and English word order can cross paragraph boundaries. Asking the model to return one fragment for each source paragraph still forced an artificial alignment. The result could be fluent at the fragment level while becoming awkward or incorrect when read as one sentence.

## Second Engineering Response

The translation boundary was redesigned:

- one PowerPoint text body became one translation block
- all internal prose was read before translation
- one natural English result replaced continuous prose
- clearly identifiable list items retained their separate structure
- prior paragraph-level checkpoints were invalidated through a versioned cache key

A second synthetic test showed a substantial improvement: previously fragmented sentences became coherent text-box translations, and bullet items remained distinct.

## New Failure Discovered Through Review

The improved translation was concise and understandable, but one candidate omitted the explicit actor `作業者` (`operator`). This exposed a deeper product problem. Fluency, concision, and full semantic coverage are different objectives. A language model may satisfy the first two while failing the third.

The product owner therefore rejected a design that treated model output as automatically complete.

## Quality-Gate Design

The system now performs a high-confidence semantic coverage review for PowerPoint output. It checks selected actors, conditions, manufacturing actions, negation, protected codes, numbers, units, required glossary terms, and list-item counts.

When a candidate fails:

1. the system creates a structured correction message naming the missing requirement;
2. the complete text box is translated again with that feedback;
3. the result is checked again;
4. after the retry limit, the job fails explicitly rather than delivering an output already known to be deficient.

The implementation also lowers generation temperature to zero and pins a model snapshot to reduce avoidable variability. These controls improve repeatability but are not described as perfect determinism. Translation memory remains the strongest mechanism for returning the exact same approved translation for the exact same normalized source.

## Terminology Transparency Improvement

The same review session raised a second user-experience requirement: engineers should not merely see that a glossary was used. They should see which term matched and the complete governed record behind it.

The preview now:

- shows every term matched in the translated block;
- displays match count;
- exposes every available column from the corresponding glossary row;
- automatically includes future columns;
- leaves missing information blank rather than inventing provenance.

The public test uses synthetic reviewer and category values. Real governed terminology and identities remain private.

## Why Content-Type Selection Remains Essential

The product continues to ask users what kind of content they are translating. This is an interaction-design decision as well as a prompt decision:

- PLC/SPLC comments prioritize stable, compact control terminology;
- supplier email preserves professional business meaning and tone;
- PowerPoint balances complete meaning with slide fit;
- product catalogs preserve technical marketing structure;
- robot programs translate comments without changing executable syntax;
- general plant translation favors clear engineering English.

Treating these as one generic translation problem would hide important user intent and create inconsistent output contracts.

## Contribution Demonstrated

This milestone records Aoi Minamoto's role across four connected responsibilities:

- **Product ownership:** defined the acceptance criterion that a fluent but incomplete translation is not acceptable;
- **Human-centered design:** interpreted the document as a user reads it rather than as XML happens to store it;
- **AI system engineering:** changed extraction, prompting, reconstruction, caching, retry, and validation behavior;
- **Governance design:** made controlled-term provenance visible without publishing private knowledge.

The contribution is evidenced by the dated Git commit, public-safe implementation, synthetic regression tests, ADR-008, the platform timeline, and the evidence index. Independent adoption, operational impact, and quality improvement remain separate future evidence requirements.

## What Was Learned

1. Input segmentation is part of model quality.
2. More context does not solve a workflow that still demands the wrong output structure.
3. Concision must be subordinate to semantic completeness.
4. Probabilistic generation requires deterministic controls around it.
5. A glossary becomes governed knowledge only when its provenance is reviewable.
6. Content type is an explicit user intent, not an implementation detail.
7. Repeated realistic testing is more valuable than assuming a fluent output is correct.

## Next Measurable Work

- expand the public synthetic evaluation set beyond the initial high-confidence patterns;
- measure omission, terminology, and reviewer-acceptance rates before and after the quality gate;
- record false-positive and false-negative rates for deterministic checks;
- introduce a review state for accepted translations before they enter translation memory;
- obtain independent engineering review using non-sensitive test material;
- report only measured outcomes and clearly separate them from design intent.
