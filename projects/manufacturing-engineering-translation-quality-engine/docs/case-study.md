# Case Study: From Translation Script to Engineering Quality Engine

## Problem

Manufacturing translation outputs must satisfy requirements that a generic text translator does not understand. Fluency alone cannot prove that PLC addresses were preserved, robot instructions were untouched, an HMI label will fit, or terminology came from an approved source.

## Engineering contribution

Aoi Minamoto defined and implemented a quality-engine approach that places deterministic controls around AI translation:

- identified format-specific translation boundaries;
- separated Japanese fragments from existing English and technical syntax;
- introduced approved terminology with visible match evidence;
- protected identifiers before translation;
- reconstructed engineering outputs in their original structure and encoding;
- converted hidden failure modes into explicit checks and warnings;
- separated public architecture evidence from private organizational artifacts.

## Representative public scenario

The synthetic source `Ready signal M500 (非常停止)` contains English, a technical address, and an approved Japanese term. The engine translates only the controlled Japanese span, preserves `M500`, and records the terminology match. An unknown Japanese term remains visible and fails the review gate instead of being silently accepted.

For the synthetic robot example, only text following the comment delimiter is eligible for translation. Program instructions and the selected source encoding remain unchanged.

## Architectural significance

The core contribution is not a list of translated words. It is a reusable mechanism that converts manufacturing requirements into enforceable software stages and review evidence. Translation providers can change without discarding the engineering controls.

## Evidence and claim discipline

This case study supports authorship, software implementation, architecture decisions, and a dated evolution narrative. It does not by itself establish organizational impact, broad adoption, commercial success, or field-wide significance. Those claims require independent and verifiable evidence outside this public-safe codebase.

## Next evidence to collect

- benchmark accuracy before and after controlled terminology;
- identifier-preservation pass rate by requirement profile;
- engineering-review acceptance and correction rates;
- representative processing time and throughput;
- dated reviewer or manager confirmation of the contribution and its importance;
- sanitized evidence of reuse across teams or sites, when authorized.
