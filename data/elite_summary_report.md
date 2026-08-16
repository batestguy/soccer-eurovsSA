# ELITE COMPOSITION — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 12:56 UTC
- Metric A: share of the top-N monthly Elo ranking per confederation (thresholds [5, 10, 20]), window 1992-01..2026-07.
- Model: SES per confederation x threshold (alpha fitted by rolling-origin +12-month forecast SSE); forecast anchored at the fitted SES level.
- Fit uses observed months only; gaps (Covid 2020 etc.) carried forward for continuity.
- 2026-07 is the latest observed ranking month in ranking_chronology.csv.
- Membership overrides applied (Australia, Israel, Kazakhstan).

## Running leader (% of window led, final share/rank)
| Threshold | Confederation | alpha | % window led | final share | final rank | forecast (+12m) |
|---|---|---|---|---|---|---|
| 5 | UEFA | 0.590 | 68.3 | 0.60 | 1 | 0.56 |
| 5 | CONMEBOL | 0.040 | 15.3 | 0.20 | 2 | 0.20 |
| 5 | CONCACAF | 0.030 | 13.6 | 0.20 | 2 | 0.23 |
| 5 | CAF | 0.060 | 6.9 | 0.00 | 4 | 0.10 |
| 5 | OFC | 0.010 | 0.7 | 0.00 | 4 | 0.01 |
| 5 | AFC | 0.020 | 9.4 | 0.00 | 4 | 0.13 |
| 10 | UEFA | 0.740 | 69.6 | 0.60 | 1 | 0.57 |
| 10 | CONMEBOL | 0.060 | 6.2 | 0.20 | 2 | 0.17 |
| 10 | CAF | 0.060 | 10.4 | 0.10 | 3 | 0.12 |
| 10 | CONCACAF | 0.030 | 12.4 | 0.10 | 3 | 0.24 |
| 10 | OFC | 0.010 | 0.7 | 0.00 | 5 | 0.01 |
| 10 | AFC | 0.390 | 10.1 | 0.00 | 5 | 0.02 |
| 20 | UEFA | 0.790 | 64.9 | 0.45 | 1 | 0.48 |
| 20 | CAF | 0.090 | 15.6 | 0.20 | 2 | 0.15 |
| 20 | CONMEBOL | 0.340 | 2.7 | 0.15 | 3 | 0.14 |
| 20 | CONCACAF | 0.030 | 10.4 | 0.15 | 3 | 0.25 |
| 20 | AFC | 0.480 | 9.9 | 0.05 | 5 | 0.07 |
| 20 | OFC | 0.030 | 0.5 | 0.00 | 6 | 0.02 |
