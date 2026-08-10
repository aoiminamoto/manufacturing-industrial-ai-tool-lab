# ADR-001: Separate Pixel Localization from Language Translation

- **Date:** 2026-08-10
- **Status:** Implemented; field validation continuing
- **Decision owner:** Aoi Minamoto

## Context

Mobile field testing showed that a multimodal language model could often read and
translate small Japanese PLC comments while returning geometrically unstable
bounding boxes. In some frames, multiple translated rows appeared near the top of
the screen or expanded across neighboring comments even though the reading order
was reasonable. Earlier successful frames demonstrated feasibility but did not
establish deterministic localization.

This distinction matters: semantic translation quality and pixel localization are
different engineering problems. A language model is useful for manufacturing
context and controlled terminology, but approximate vision coordinates are not a
reliable replacement for a dedicated OCR detector.

## Decision

The default **ACCURATE** path separates responsibilities:

1. A local PaddleOCR sidecar detects Japanese text and returns `rec_boxes` derived
   from image pixels.
2. The sidecar binds only to `127.0.0.1:8506` and does not persist frames or text.
3. The Lens normalizes pixel boxes to its 0–1000 overlay coordinate system.
4. OpenAI receives extracted text for glossary-controlled contextual translation,
   not the camera frame.
5. Canvas renders English near the source box, with horizontal expansion capped to
   reduce collisions with neighboring PLC comments.
6. **FAST** retains the lower-cost vision path and labels its boxes as approximate.

PaddleOCR runs in `.ocr-venv`, separate from the web application's `.venv`, because
its numerical/runtime dependencies are large and version-sensitive.

## Evidence used in the decision

- Repeated iPhone tests reproduced line displacement and overlapping overlays.
- The dedicated OCR probe returned 74 total text boxes on a 2532 × 1170 test frame;
  after Japanese/confidence filtering, the local endpoint returned 22 Japanese
  regions distributed across the actual upper, middle, and lower image areas.
- With models kept resident, the local OCR request completed in approximately
  12 seconds on the tested Windows CPU. This is an environment-specific observation,
  not a general performance guarantee.
- A server recognition model was rejected for the interactive path after consuming
  excessive CPU time; the Japanese-capable mobile recognition model was selected.

No internal screenshot, production terminology, credential, tunnel URL, or raw
recognized text is included in this repository as evidence.

## Consequences

### Benefits

- Pixel-derived boxes are more reproducible than free-form vision estimates.
- Accurate-mode images remain in the local Lens/OCR processes.
- OCR and translation can be evaluated and upgraded independently.
- Dependency isolation limits the blast radius of Paddle runtime changes.

### Costs and limitations

- Initial setup downloads sizable model/runtime artifacts.
- Local CPU inference adds latency and memory use.
- Extracted text still crosses the configured OpenAI boundary in Accurate mode.
- OCR coordinates do not prove recognition or translation correctness.
- Glare, perspective, moiré, very small fonts, and dense ladder logic remain risks.
- Continuous iPhone tests, synthetic fixtures, collision metrics, and independent
  engineering review are still required before production use.

## Rejected alternatives

- **Fixed screen offsets:** worked only for one framing/orientation and was removed.
- **Treating previous successful frames as validation:** insufficient because later
  frames reproduced displacement.
- **Sending the full frame to a second language-model localization pass:** adds cost
  without creating deterministic pixel geometry.
- **Using the server OCR recognition model for every scan:** too slow for the tested
  interactive CPU environment.
