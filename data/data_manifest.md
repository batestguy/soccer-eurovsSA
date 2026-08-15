# DATA MANIFEST — stage 00

- Generated: 2026-08-15 02:44 UTC
- Match source: martj42/international_results @ 65d212a
- Matches: 49,520  (1872-11-30 .. 2026-07-19)
- Historical canon rows: 23 (WCs 1930-2026; 2026 = prediction target)
- Continental finals rows: 146
- Ranking chronology rows: 33,165  teams=276  months=404
- Latest Elo rows: 337
- Confederation map rows: 337  Others=102
- Alignment rows: 138  (WC x confederation)

## Ranking chronology note
Rankings are **self-computed Elo** from all international matches since 1872, NOT
official FIFA points. The briefing requested 'FIFA rankings from 1992'; official FIFA
points history is not bulk-downloadable for free, so Elo is the reproducible proxy.
Elo params (documented in stage00_data.py): K = 60/50/40/30 (WC/continental/
qualifier-competitive/friendly), home advantage +100 when not neutral, goal-margin
factor G = 1.0 (draw) else 1.5 + 0.125*(min(gd,5)-1). Monthly = last rating of month.
USSR matches are recorded as 'Russia', West Germany as 'Germany' (dataset convention),
so Elo series are continuous across the political transitions.

## Alignment engine note
For each WC, each confederation's 'continental champion' is the winner of the most
recent continental final within 18 months before the WC start. If none qualifies,
the engine falls back to the most recent final and flags used_fallback=True.
For 2026, UEFA/CONMEBOL/AFC/OFC use the fallback (their trophies are ~23-28 months out).

## Known gaps
- continental finals flagged UNKNOWN need results filled in before modeling:
  - African Cup of Nations 2025
  - Gold Cup 2025
- Oceania Nations Cup 2020 was cancelled (COVID); no row included.
- WC 2026 outcome is intentionally absent (prediction target).
