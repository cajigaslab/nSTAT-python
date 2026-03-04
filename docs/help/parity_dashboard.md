# Parity Dashboard

This dashboard summarizes current MATLAB-to-Python parity status from generated
artifacts in the `parity/` directory.

## Structural parity
| Metric | Value |
|---|---:|
| High gaps | 0 |
| Medium gaps | 0 |
| Low gaps | 0 |
| Total gaps | 0 |

## Functional parity (methods)
| Metric | Value |
|---|---:|
| Total methods | 501 |
| Contract-verified | 480 |
| Contract-explicit verified | 480 |
| Probe-verified | 0 |
| Excluded methods | 21 |
| Missing symbols | 0 |
| Unverified behavior | 0 |

## Example parity
| Metric | Value |
|---|---:|
| Total topics | 30 |
| Validated topics | 26 |
| MATLAB doc-only topics | 4 |
| Pending manual review topics | 0 |
| Missing executable topics | 0 |

### Out-of-scope example topics
- `DocumentationSetup2025b`
- `FitResSummaryExamples`
- `FitResultExamples`
- `FitResultReference`

## Numeric drift
| Metric | Value |
|---|---:|
| Topics checked | 31 |
| Required notebook topics | 30 |
| Required topics checked | 30 |
| Topics passed | 31 |
| Topics failed | 0 |
| Metrics checked | 180 |
| Metrics failed | 0 |

## Frozen MATLAB data snapshot
| Metric | Value |
|---|---|
| Snapshot file | `matlab_gold_snapshot_20260302.yml` |
| Snapshot id | `matlab_gold_20260302` |
| Snapshot date | `2026-03-02` |
| Mirror file count | `42` |
| Source manifest SHA256 | `578a20db6433efed11466eb64ab23d77a1f106f3a2da8937c28849511e9385e7` |
| Mirror manifest SHA256 | `b980cb16f5872b53e841f360e9a81a34c9e6b4a5149a9ef3b7fd96314899bfe4` |

## Artifact links
- [parity_gap_report.json](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/parity_gap_report.json)
- [function_example_alignment_report.json](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/function_example_alignment_report.json)
- [numeric_drift_report.json](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/numeric_drift_report.json)
- [example_output_spec.yml](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/example_output_spec.yml)
- [method_closure_sprint.md](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/method_closure_sprint.md)
- [Full validation report PDF](../assets/reports/nstat_python_validation_report_full_latest.pdf)
