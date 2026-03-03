# Line Review Sprint Backlog

- Source report: `parity/line_by_line_review_report.json`
- Generated at: `2026-03-03T00:23:45.940093+00:00`
- Total topics: `30`
- Needs review: `24`
- Average line alignment ratio: `0.089`

## Priority Queue
| Priority | Topic | Status | Line ratio | Step recall | Step precision | Missing MATLAB steps |
|---:|---|---|---:|---:|---:|---:|
| 1 | publish_all_helpfiles | needs_review | 0.000 | 0.000 | 0.000 | 48 |
| 2 | nSTATPaperExamples | needs_review | 0.015 | 0.015 | 0.323 | 1386 |
| 3 | HippocampalPlaceCellExample | needs_review | 0.035 | 0.043 | 0.106 | 121 |
| 4 | AnalysisExamples | needs_review | 0.036 | 0.222 | 0.214 | 52 |
| 5 | HistoryExamples | needs_review | 0.036 | 0.062 | 0.028 | 16 |
| 6 | AnalysisExamples2 | needs_review | 0.037 | 0.057 | 0.056 | 52 |
| 7 | mEPSCAnalysis | needs_review | 0.039 | 0.038 | 0.043 | 51 |
| 8 | ValidationDataSet | needs_review | 0.040 | 0.034 | 0.050 | 58 |
| 9 | PPThinning | needs_review | 0.050 | 0.314 | 0.133 | 32 |
| 10 | DecodingExampleWithHist | needs_review | 0.051 | 0.102 | 0.113 | 57 |
| 11 | DecodingExample | needs_review | 0.054 | 0.185 | 0.189 | 52 |
| 12 | ConfigCollExamples | needs_review | 0.059 | 0.333 | 0.032 | 2 |

## Execution Notes
- Address topics in queue order unless a dependency forces reordering.
- For each topic, update notebook logic first, then rerun `review_line_by_line_equivalence.py`.
- Keep MATLAB/Python operation ordering aligned before adjusting numeric thresholds.
- After each topic fix, regenerate and commit: `parity/line_by_line_review_report.json` and this backlog.

