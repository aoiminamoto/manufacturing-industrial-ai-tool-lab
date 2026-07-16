# Usage and Quality Metrics Template

Use one record per reporting period. Publish only sanitized aggregates that have a reliable data source.

## Reporting Context

| Field | Value |
|---|---|
| Reporting period | `YYYY-MM-DD to YYYY-MM-DD` |
| Environment | Pilot / Test / Production |
| Scope | Sanitized team or engineering-group description |
| Data source | Usage counter / job database / review log / test report |
| Prepared by | Name and role |
| Verified by | Name and role, if applicable |

## Adoption Metrics

| Metric | Value | Calculation/source | Public-safe? |
|---|---:|---|---|
| Unique active users | Not yet measured | Defined unique-user source required | Review before publication |
| Translation actions | Not yet measured | Application action counter | Aggregate only |
| Documents processed | Not yet measured | Completed job records | Aggregate only |
| HMI/images reviewed | Not yet measured | Completed image-review jobs | Aggregate only |
| Reporting-period growth | Not yet measured | Current versus previous period | Aggregate only |

## Reliability Metrics

| Metric | Value | Calculation/source |
|---|---:|---|
| Successful completion rate | Not yet measured | Completed / total jobs |
| Failed-job rate | Not yet measured | Failed / total jobs |
| Checkpoint recoveries | Not yet measured | Resumed job events |
| Median completion time | Not yet measured | Job duration records |
| P95 completion time | Not yet measured | Job duration records |

## Translation Quality Metrics

| Metric | Value | Calculation/source |
|---|---:|---|
| Glossary match count | Not yet measured | Terminology report |
| Engineering acceptance rate | Not yet measured | Accepted without material correction / reviewed items |
| Correction rate | Not yet measured | Corrected / reviewed items |
| Japanese text missed | Not yet measured | Review findings |
| Terminology consistency | Not yet measured | Repeated-term audit |

## Impact Metrics

Time-saved or productivity claims require a documented baseline. Use:

```text
Measured impact = Baseline manual effort - Measured assisted effort
```

Record sample size, workflow, reviewer, and limitations. Do not publish estimated savings as measured results.

## Monthly Summary

- What improved:
- What degraded:
- Main user feedback:
- Corrective action:
- Next measurement:
