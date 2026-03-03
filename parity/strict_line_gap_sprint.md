# Strict Line-Port Gap Sprint

- Source report: `/tmp/nstat_python_exec_next/parity/function_example_alignment_report.json`
- Total topics: `30`
- Strict summary: verified=1, partial=6, gap=19

## Priority Queue
| Priority | Topic | Coverage | Function recall | Code-line ratio (Py/MATLAB) | MATLAB lines | Python lines |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SignalObjExamples | 0.0000 | 0.0833 | 0.5432 | 81 | 44 |
| 2 | NetworkTutorial | 0.0000 | 0.1081 | 1.2500 | 88 | 110 |
| 3 | DecodingExampleWithHist | 0.0000 | 0.1429 | 1.2545 | 55 | 69 |
| 4 | StimulusDecode2D | 0.0000 | 0.1489 | 0.6848 | 92 | 63 |
| 5 | AnalysisExamples | 0.0000 | 0.1795 | 1.0678 | 59 | 63 |
| 6 | DecodingExample | 0.0000 | 0.1842 | 1.2105 | 57 | 69 |
| 7 | PSTHEstimation | 0.0000 | 0.2143 | 1.6071 | 28 | 45 |
| 8 | HybridFilterExample | 0.0069 | 0.1324 | 0.4896 | 288 | 141 |
| 9 | PPSimExample | 0.0488 | 0.1111 | 1.8293 | 41 | 75 |
| 10 | PPThinning | 0.0750 | 0.3000 | 2.3500 | 40 | 94 |
| 11 | EventsExamples | 0.1250 | 0.2500 | 3.8750 | 8 | 31 |
| 12 | CovariateExamples | 0.1579 | 0.7143 | 2.9474 | 19 | 56 |
| 13 | TrialExamples | 0.1600 | 0.9091 | 3.1200 | 25 | 78 |
| 14 | nSpikeTrainExamples | 0.3000 | 0.8333 | 4.2000 | 10 | 42 |
| 15 | nstCollExamples | 0.3125 | 0.6364 | 3.5000 | 16 | 56 |
| 16 | ConfigCollExamples | 0.3333 | 1.0000 | 11.0000 | 3 | 33 |
| 17 | TrialConfigExamples | 0.3333 | 1.0000 | 9.3333 | 3 | 28 |
| 18 | CovCollExamples | 0.7000 | 1.0000 | 5.6000 | 10 | 56 |
| 19 | HistoryExamples | 1.0000 | 1.0000 | 4.0556 | 18 | 73 |

## Execution Checklist
- Export executable-line snapshots for each gap topic.
- Regenerate notebooks with snapshot anchors.
- Re-run `tools/parity/sync_parity_artifacts.py`.
- Target strict status: `line_port_partial` or `line_port_verified`.

