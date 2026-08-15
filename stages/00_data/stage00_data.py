# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 00 — DATA FOUNDATION
# Bayesian World Cup Prediction (soccer-deeplearning)
#
# Produces the 3-layer data foundation:
#   Layer 1  historical_canon.csv      — every World Cup final 1930-2026
#   Layer 1  continental_finals.csv    — every continental final (>=1930)
#   Layer 2  ranking_chronology.csv    — monthly Elo per team since 1992-01
#           latest_elo.csv             — most recent Elo per team
#   Layer 3  alignment_engine.csv      — WC x confederation aligned tournament
#   helper   team_confederations.csv   — team -> confederation map
#
# Reproducibility: pinned dataset commit 65d212a of
#   martj42/international_results (49,521 international matches, 1872-2026).
# Runtime: Google Colab (free). Sessions are ephemeral — state lives on
# Google Drive + a public GitHub mirror, never in the runtime.
# =====================================================================

import os
import io
import sys
import json
import base64
import shutil
import subprocess
import datetime as dt
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0. CONFIG — edit before first run in Colab
# ---------------------------------------------------------------------
DRIVE_DIR   = "/content/drive/MyDrive/SoccerDL"   # Google Drive root
GITHUB_REPO = "batestguy/soccer-eurovsSA"  # PUBLIC repo
DATA_SHA    = "65d212aac5deec5157071dcf6e9b05fce0223c84"
RAW_URL     = f"https://raw.githubusercontent.com/martj42/international_results/{DATA_SHA}/results.csv"

README_MD = """# Soccer Deep Learning — Bayesian World Cup Prediction

Bayesian decision-support app for football (PyMC + Monte Carlo + Gradio).
Stage 00 = data foundation. Artifacts:
- `data/historical_canon.csv`      World Cup finals 1930-2026 (2026 = prediction target)
- `data/continental_finals.csv`    all continental finals (>= 1930)
- `data/ranking_chronology.csv`    monthly Elo per team since 1992-01 (self-computed proxy)
- `data/latest_elo.csv`            most recent Elo per team
- `data/alignment_engine.csv`      WC x confederation aligned continental tournament
- `data/team_confederations.csv`   team -> confederation map
- `data/source/`                   pinned raw match data (reproducibility)

See `data/data_manifest.md` for provenance and known gaps.
"""

# Tournament -> Elo K factor (standard Elo formulation, documented in data_manifest.md)
CONT_K       = {"FIFA World Cup": 60.0}
CONF_CHAMP   = {"UEFA Euro", "Copa América", "African Cup of Nations",
                "Gold Cup", "Oceania Nations Cup", "AFC Asian Cup",
                "CONCACAF Championship"}
COMPETITIVE  = {"UEFA Nations League", "Confederations Cup", "FIFA Series",
                "CONMEBOL-UEFA Cup of Champions", "CONCACAF Nations League",
                "Olympic Games"}

def k_factor(tournament):
    t = str(tournament or "")
    if t in CONT_K:            return 60.0
    if t in CONF_CHAMP:        return 50.0
    if "qualification" in t or t in COMPETITIVE: return 40.0
    return 30.0   # friendlies + everything else

def goal_factor(goal_diff):
    if goal_diff <= 0:
        return 1.0
    return 1.5 + 0.125 * (min(goal_diff, 5) - 1)   # 1-goal -> 1.50, 3-goal -> 1.75

# ---------------------------------------------------------------------
# 1. HISTORICAL CANON — World Cup finals 1930-2026 (hand-curated)
# ---------------------------------------------------------------------
# (winner / runner-up use dataset team names; West Germany == "Germany")
WORLD_CUPS = [
    dict(wc_year=1930, host="Uruguay",     winner="Uruguay", runner_up="Argentina",
         start="1930-07-13", end="1930-07-30"),
    dict(wc_year=1934, host="Italy",       winner="Italy",   runner_up="Czechoslovakia",
         start="1934-05-27", end="1934-06-10"),
    dict(wc_year=1938, host="France",      winner="Italy",   runner_up="Hungary",
         start="1938-06-04", end="1938-06-19"),
    dict(wc_year=1950, host="Brazil",      winner="Uruguay", runner_up="Brazil",
         start="1950-06-24", end="1950-07-16"),
    dict(wc_year=1954, host="Switzerland", winner="Germany", runner_up="Hungary",
         start="1954-06-16", end="1954-07-04"),
    dict(wc_year=1958, host="Sweden",      winner="Brazil",  runner_up="Sweden",
         start="1958-06-08", end="1958-06-29"),
    dict(wc_year=1962, host="Chile",       winner="Brazil",  runner_up="Czechoslovakia",
         start="1962-05-30", end="1962-06-17"),
    dict(wc_year=1966, host="England",     winner="England", runner_up="Germany",
         start="1966-07-11", end="1966-07-30"),
    dict(wc_year=1970, host="Mexico",      winner="Brazil",  runner_up="Italy",
         start="1970-05-31", end="1970-06-21"),
    dict(wc_year=1974, host="Germany",     winner="Germany", runner_up="Netherlands",
         start="1974-06-13", end="1974-07-07"),
    dict(wc_year=1978, host="Argentina",   winner="Argentina", runner_up="Netherlands",
         start="1978-06-01", end="1978-06-25"),
    dict(wc_year=1982, host="Spain",       winner="Italy",   runner_up="Germany",
         start="1982-06-13", end="1982-07-11"),
    dict(wc_year=1986, host="Mexico",      winner="Argentina", runner_up="Germany",
         start="1986-05-31", end="1986-06-29"),
    dict(wc_year=1990, host="Italy",       winner="Germany", runner_up="Argentina",
         start="1990-06-08", end="1990-07-08"),
    dict(wc_year=1994, host="United States", winner="Brazil", runner_up="Italy",
         start="1994-06-17", end="1994-07-17"),
    dict(wc_year=1998, host="France",      winner="France",  runner_up="Brazil",
         start="1998-06-10", end="1998-07-12"),
    dict(wc_year=2002, host="South Korea/Japan", winner="Brazil", runner_up="Germany",
         start="2002-05-31", end="2002-06-30"),
    dict(wc_year=2006, host="Germany",     winner="Italy",   runner_up="France",
         start="2006-06-09", end="2006-07-09"),
    dict(wc_year=2010, host="South Africa", winner="Spain",  runner_up="Netherlands",
         start="2010-06-11", end="2010-07-11"),
    dict(wc_year=2014, host="Brazil",      winner="Germany", runner_up="Argentina",
         start="2014-06-12", end="2014-07-13"),
    dict(wc_year=2018, host="Russia",      winner="France",  runner_up="Croatia",
         start="2018-06-14", end="2018-07-15"),
    dict(wc_year=2022, host="Qatar",       winner="Argentina", runner_up="France",
         start="2022-11-20", end="2022-12-18"),
    # 2026 = PREDICTION TARGET, not training data
    dict(wc_year=2026, host="United States/Canada/Mexico", winner=None, runner_up=None,
         start="2026-06-11", end="2026-07-19"),
]

# ---------------------------------------------------------------------
# 2. CONTINENTAL FINALS — every edition that can feed the Alignment Engine
#    (champion names = dataset team names; UNKNOWN = result not curated yet)
# ---------------------------------------------------------------------
CONTINENTAL_FINALS = [
    # ---- UEFA European Championship ---------------------------------
    dict(conf="UEFA", tournament="UEFA Euro", year=1960, champion="Russia",      runner_up="Yugoslavia", final_date="1960-07-10"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1964, champion="Spain",       runner_up="Russia",      final_date="1964-06-21"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1968, champion="Italy",       runner_up="Yugoslavia", final_date="1968-06-10"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1972, champion="Germany",     runner_up="Russia",      final_date="1972-06-18"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1976, champion="Czechoslovakia", runner_up="Germany",  final_date="1976-06-20"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1980, champion="Germany",     runner_up="Belgium",     final_date="1980-06-22"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1984, champion="France",      runner_up="Spain",       final_date="1984-06-27"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1988, champion="Netherlands", runner_up="Russia",      final_date="1988-06-25"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1992, champion="Denmark",     runner_up="Germany",     final_date="1992-06-26"),
    dict(conf="UEFA", tournament="UEFA Euro", year=1996, champion="Germany",     runner_up="Czech Republic", final_date="1996-06-30"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2000, champion="France",      runner_up="Italy",       final_date="2000-07-02"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2004, champion="Greece",      runner_up="Portugal",    final_date="2004-07-04"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2008, champion="Spain",       runner_up="Germany",     final_date="2008-06-29"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2012, champion="Spain",       runner_up="Italy",       final_date="2012-07-01"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2016, champion="Portugal",    runner_up="France",      final_date="2016-07-10"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2020, champion="Italy",       runner_up="England",     final_date="2021-07-11"),
    dict(conf="UEFA", tournament="UEFA Euro", year=2024, champion="Spain",       runner_up="England",     final_date="2024-07-14"),
    # ---- Copa América (CONMEBOL) ------------------------------------
    dict(conf="CONMEBOL", tournament="Copa América", year=1929, champion="Argentina", runner_up="Paraguay", final_date="1929-11-17"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1935, champion="Uruguay",   runner_up="Argentina", final_date="1935-01-27"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1937, champion="Argentina", runner_up="Brazil",    final_date="1937-02-01"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1939, champion="Peru",      runner_up="Uruguay",   final_date="1939-02-12"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1941, champion="Argentina", runner_up="Uruguay",   final_date="1941-03-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1942, champion="Uruguay",   runner_up="Argentina", final_date="1942-02-07"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1945, champion="Argentina", runner_up="Brazil",    final_date="1945-03-25"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1946, champion="Argentina", runner_up="Brazil",    final_date="1946-02-12"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1947, champion="Argentina", runner_up="Brazil",    final_date="1947-12-31"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1949, champion="Brazil",    runner_up="Paraguay",  final_date="1949-05-11"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1953, champion="Paraguay",  runner_up="Brazil",    final_date="1953-04-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1955, champion="Argentina", runner_up="Chile",     final_date="1955-03-30"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1956, champion="Uruguay",   runner_up="Argentina", final_date="1956-03-15"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1957, champion="Argentina", runner_up="Brazil",    final_date="1957-04-10"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1959, champion="Argentina", runner_up="Brazil",    final_date="1959-04-10"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1959, champion="Uruguay",   runner_up="Argentina", final_date="1959-12-30"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1963, champion="Bolivia",   runner_up="Paraguay",  final_date="1963-03-31"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1967, champion="Uruguay",   runner_up="Argentina", final_date="1967-11-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1975, champion="Peru",      runner_up="Colombia",  final_date="1975-10-28"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1979, champion="Paraguay",  runner_up="Chile",     final_date="1979-12-11"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1983, champion="Uruguay",   runner_up="Brazil",    final_date="1983-11-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1987, champion="Uruguay",   runner_up="Chile",     final_date="1987-07-12"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1989, champion="Brazil",    runner_up="Uruguay",   final_date="1989-07-16"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1991, champion="Argentina", runner_up="Brazil",    final_date="1991-07-21"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1993, champion="Argentina", runner_up="Mexico",    final_date="1993-07-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1995, champion="Uruguay",   runner_up="Brazil",    final_date="1995-07-23"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1997, champion="Brazil",    runner_up="Bolivia",   final_date="1997-06-29"),
    dict(conf="CONMEBOL", tournament="Copa América", year=1999, champion="Brazil",    runner_up="Uruguay",   final_date="1999-07-18"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2001, champion="Colombia",  runner_up="Mexico",    final_date="2001-07-29"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2004, champion="Brazil",    runner_up="Argentina", final_date="2004-07-25"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2007, champion="Brazil",    runner_up="Argentina", final_date="2007-07-15"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2011, champion="Uruguay",   runner_up="Paraguay",  final_date="2011-07-24"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2015, champion="Chile",     runner_up="Argentina", final_date="2015-07-04"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2016, champion="Chile",     runner_up="Argentina", final_date="2016-06-26"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2019, champion="Brazil",    runner_up="Peru",      final_date="2019-07-07"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2021, champion="Argentina", runner_up="Brazil",    final_date="2021-07-10"),
    dict(conf="CONMEBOL", tournament="Copa América", year=2024, champion="Argentina", runner_up="Colombia",  final_date="2024-07-14"),
    # ---- African Cup of Nations (CAF) -------------------------------
    dict(conf="CAF", tournament="African Cup of Nations", year=1957, champion="Egypt",       runner_up="Ethiopia",     final_date="1957-02-16"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1959, champion="Egypt",       runner_up="Sudan",        final_date="1959-05-29"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1962, champion="Ethiopia",    runner_up="Egypt",        final_date="1962-01-21"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1963, champion="Ghana",       runner_up="Sudan",        final_date="1963-11-24"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1965, champion="Ghana",       runner_up="Tunisia",      final_date="1965-11-21"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1968, champion="DR Congo",    runner_up="Ghana",        final_date="1968-01-21"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1970, champion="Sudan",       runner_up="Ghana",        final_date="1970-03-06"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1972, champion="Congo",       runner_up="Mali",         final_date="1972-03-05"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1974, champion="DR Congo",    runner_up="Zambia",       final_date="1974-03-14"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1976, champion="Morocco",     runner_up="Guinea",       final_date="1976-03-14"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1978, champion="Ghana",       runner_up="Uganda",       final_date="1978-03-18"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1980, champion="Nigeria",     runner_up="Algeria",      final_date="1980-03-22"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1982, champion="Ghana",       runner_up="Libya",        final_date="1982-03-19"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1984, champion="Cameroon",    runner_up="Nigeria",      final_date="1984-03-18"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1986, champion="Egypt",       runner_up="Cameroon",     final_date="1986-03-28"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1988, champion="Cameroon",    runner_up="Nigeria",      final_date="1988-03-27"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1990, champion="Algeria",     runner_up="Nigeria",      final_date="1990-03-16"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1992, champion="Ivory Coast", runner_up="Ghana",        final_date="1992-01-26"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1994, champion="Nigeria",     runner_up="Zambia",       final_date="1994-04-10"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1996, champion="South Africa", runner_up="Tunisia",     final_date="1996-02-03"),
    dict(conf="CAF", tournament="African Cup of Nations", year=1998, champion="Egypt",       runner_up="South Africa", final_date="1998-02-28"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2000, champion="Cameroon",    runner_up="Nigeria",      final_date="2000-02-13"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2002, champion="Cameroon",    runner_up="Senegal",      final_date="2002-02-10"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2004, champion="Tunisia",     runner_up="Morocco",      final_date="2004-02-14"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2006, champion="Egypt",       runner_up="Ivory Coast",  final_date="2006-02-10"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2008, champion="Egypt",       runner_up="Cameroon",     final_date="2008-02-10"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2010, champion="Egypt",       runner_up="Ghana",        final_date="2010-01-31"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2012, champion="Zambia",      runner_up="Ivory Coast",  final_date="2012-02-12"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2013, champion="Nigeria",     runner_up="Burkina Faso", final_date="2013-02-10"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2015, champion="Ivory Coast", runner_up="Ghana",        final_date="2015-02-08"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2017, champion="Cameroon",    runner_up="Egypt",        final_date="2017-02-05"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2019, champion="Algeria",     runner_up="Senegal",      final_date="2019-07-19"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2021, champion="Senegal",     runner_up="Egypt",        final_date="2022-02-06"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2023, champion="Ivory Coast", runner_up="Nigeria",      final_date="2024-02-11"),
    dict(conf="CAF", tournament="African Cup of Nations", year=2025, champion="Morocco",      runner_up="Senegal",       final_date="2026-01-18"),
    # ---- CONCACAF Championship (1963-1989) then Gold Cup (1991+) ----
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1963, champion="Costa Rica",  runner_up="El Salvador",     final_date="1963-04-04"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1965, champion="Mexico",      runner_up="Guatemala",       final_date="1965-04-11"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1967, champion="Guatemala",   runner_up="Mexico",          final_date="1967-03-19"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1969, champion="Costa Rica",  runner_up="Guatemala",       final_date="1969-12-04"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1971, champion="Mexico",      runner_up="Haiti",           final_date="1971-09-04"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1973, champion="Haiti",       runner_up="Trinidad and Tobago", final_date="1973-12-14"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1977, champion="Mexico",      runner_up="Haiti",           final_date="1977-10-23"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1981, champion="Honduras",    runner_up="El Salvador",     final_date="1981-11-22"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1985, champion="Canada",      runner_up="Honduras",        final_date="1985-09-14"),
    dict(conf="CONCACAF", tournament="CONCACAF Championship", year=1989, champion="Costa Rica",  runner_up="United States",   final_date="1989-11-19"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=1991, champion="United States", runner_up="Honduras",     final_date="1991-07-07"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=1993, champion="Mexico",       runner_up="United States", final_date="1993-07-25"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=1996, champion="Mexico",       runner_up="Brazil",       final_date="1996-01-21"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=1998, champion="Mexico",       runner_up="United States", final_date="1998-02-15"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2000, champion="Canada",       runner_up="Colombia",     final_date="2000-02-27"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2002, champion="United States", runner_up="Costa Rica",  final_date="2002-02-02"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2003, champion="Mexico",       runner_up="Brazil",       final_date="2003-07-27"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2005, champion="United States", runner_up="Panama",      final_date="2005-07-24"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2007, champion="United States", runner_up="Mexico",      final_date="2007-06-24"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2009, champion="Mexico",       runner_up="United States", final_date="2009-07-26"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2011, champion="Mexico",       runner_up="United States", final_date="2011-06-25"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2013, champion="United States", runner_up="Panama",      final_date="2013-07-28"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2015, champion="Mexico",       runner_up="Jamaica",      final_date="2015-07-26"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2017, champion="United States", runner_up="Jamaica",     final_date="2017-07-26"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2019, champion="Mexico",       runner_up="United States", final_date="2019-07-07"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2021, champion="United States", runner_up="Mexico",      final_date="2021-08-01"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2023, champion="Mexico",       runner_up="Panama",       final_date="2023-07-16"),
    dict(conf="CONCACAF", tournament="Gold Cup", year=2025, champion="Mexico",       runner_up="United States", final_date="2025-07-06"),
    # ---- Oceania Nations Cup (OFC) ----------------------------------
    dict(conf="OFC", tournament="Oceania Nations Cup", year=1973, champion="New Zealand",    runner_up="Tahiti",           final_date="1973-02-24"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=1980, champion="Australia",      runner_up="Tahiti",           final_date="1980-03-01"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=1996, champion="Australia",      runner_up="Tahiti",           final_date="1996-11-01"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=1998, champion="New Zealand",    runner_up="Australia",        final_date="1998-09-04"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2000, champion="Australia",      runner_up="New Zealand",      final_date="2000-06-28"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2002, champion="New Zealand",    runner_up="Australia",        final_date="2002-07-14"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2004, champion="Australia",      runner_up="Solomon Islands",  final_date="2004-10-10"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2008, champion="New Zealand",    runner_up="New Caledonia",    final_date="2008-11-19"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2012, champion="Tahiti",         runner_up="New Caledonia",    final_date="2012-06-10"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2016, champion="New Zealand",    runner_up="Papua New Guinea", final_date="2016-06-11"),
    dict(conf="OFC", tournament="Oceania Nations Cup", year=2024, champion="New Zealand",    runner_up="Vanuatu",          final_date="2024-06-30"),
    # ---- AFC Asian Cup ----------------------------------------------
    dict(conf="AFC", tournament="AFC Asian Cup", year=1956, champion="South Korea", runner_up="Israel",           final_date="1956-09-15"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1960, champion="South Korea", runner_up="Israel",           final_date="1960-10-23"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1964, champion="Israel",      runner_up="India",            final_date="1964-06-03"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1968, champion="Iran",        runner_up="Burma",            final_date="1968-05-19"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1972, champion="Iran",        runner_up="South Korea",      final_date="1972-05-19"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1976, champion="Iran",        runner_up="Kuwait",           final_date="1976-06-13"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1980, champion="Kuwait",      runner_up="South Korea",      final_date="1980-09-30"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1984, champion="Saudi Arabia", runner_up="China",           final_date="1984-12-16"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1988, champion="Saudi Arabia", runner_up="South Korea",     final_date="1988-12-18"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1992, champion="Japan",       runner_up="Saudi Arabia",     final_date="1992-11-08"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=1996, champion="Saudi Arabia", runner_up="United Arab Emirates", final_date="1996-12-21"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2000, champion="Japan",       runner_up="Saudi Arabia",     final_date="2000-10-29"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2004, champion="Japan",       runner_up="China",            final_date="2004-08-07"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2007, champion="Iraq",        runner_up="Saudi Arabia",     final_date="2007-07-29"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2011, champion="Japan",       runner_up="Australia",        final_date="2011-01-29"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2015, champion="Australia",   runner_up="South Korea",      final_date="2015-01-31"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2019, champion="Qatar",       runner_up="Japan",            final_date="2019-02-01"),
    dict(conf="AFC", tournament="AFC Asian Cup", year=2023, champion="Qatar",       runner_up="Jordan",           final_date="2024-02-10"),
]

# Manual confederation overrides (rare cross-confederation movers / safe pins).
# Everything else is inferred from continental-tournament participation.
MANUAL_CONF = {
    "Australia": "AFC", "New Zealand": "OFC", "Israel": "UEFA", "Kazakhstan": "UEFA",
    "Turkey": "UEFA", "Guyana": "CONCACAF", "Suriname": "CONCACAF", "Belize": "CONCACAF",
    "French Guiana": "CONCACAF", "Palau": "OFC", "North Vietnam": "AFC",
    "Greenland": "Other", "West Papua": "Other",
    "East Turkestan": "Other", "Tibet": "Other", "Yorkshire": "Other",
    "Cascadia": "Other", "Chagos Islands": "Other", "Sápmi": "Other",
    "Kiribati": "OFC", "Tuvalu": "OFC", "Nauru": "OFC", "Federated States of Micronesia": "OFC",
}

# Continental tournaments that identify a team's confederation (inference).
CONF_BY_TOURNAMENT = {
    "UEFA":   {"UEFA Euro", "UEFA Euro qualification", "UEFA Nations League"},
    "CONMEBOL": {"Copa América", "Copa América qualification", "CONMEBOL-UEFA Cup of Champions"},
    "CAF":    {"African Cup of Nations", "African Cup of Nations qualification",
               "African Friendship Games", "All-African Games", "COSAFA Cup",
               "Amílcar Cabral Cup", "CECAFA Cup", "CEMAC Cup", "UDEAC Cup",
               "UNIFFAC Cup", "Nile Basin Tournament", "Indian Ocean Island Games"},
    "CONCACAF": {"Gold Cup", "Gold Cup qualification", "CONCACAF Championship",
                 "CONCACAF Championship qualification", "CONCACAF Nations League",
                 "CONCACAF Nations League qualification", "CONCACAF Series",
                 "CCCF Championship", "NAFC Championship", "NAFU Championship",
                 "CFU Caribbean Cup", "CFU Caribbean Cup qualification",
                 "UNCAF Cup", "Central American and Caribbean Games"},
    "OFC":    {"Oceania Nations Cup", "Oceania Nations Cup qualification",
               "Melanesia Cup", "Pacific Games", "Pacific Mini Games",
               "South Pacific Games", "South Pacific Mini Games",
               "MSG Prime Minister's Cup", "Outrigger Challenge Cup", "Trans-Tasman Cup"},
    "AFC":    {"AFC Asian Cup", "AFC Asian Cup qualification", "AFF Championship",
               "AFF Championship qualification", "EAFF Championship", "AFC Challenge Cup",
               "AFC Solidarity Cup", "WAFF Championship", "SAFF Cup", "CAFA Nations Cup",
               "ASEAN Championship", "ASEAN Championship qualification",
               "Asian Games", "South Asian Games", "Southeast Asian Games", "Gulf Cup",
               "AFC Challenge Cup qualification"},
}


# ---------------------------------------------------------------------
# 3. Elo ENGINE — full history from match results, monthly since 1992
# ---------------------------------------------------------------------
def compute_elo(matches, home_adv=100.0):
    ratings = {}
    rows = []
    def get(t):
        return ratings.setdefault(t, 1500.0)
    for r in matches.itertuples(index=False):
        ha = 0.0 if (r.neutral is True) else home_adv
        ra = get(r.home_team) + ha
        rb = get(r.away_team)
        expa = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        sa = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        gd = abs(r.home_score - r.away_score)
        g = goal_factor(gd)
        k = k_factor(r.tournament)
        d = k * g * (sa - expa)
        ratings[r.home_team] += d
        ratings[r.away_team] -= d
        rows.append((r.date, r.home_team, ratings[r.home_team]))
        rows.append((r.date, r.away_team, ratings[r.away_team]))
    elo = pd.DataFrame(rows, columns=["date", "team", "elo"])
    elo["date"] = pd.to_datetime(elo["date"])
    return elo


def monthly_chronology(elo, start="1992-01", min_matches=1):
    counts = elo["team"].value_counts()
    keep = set(counts[counts >= min_matches].index)
    e = elo[elo["team"].isin(keep)].copy()
    e["month"] = e["date"].dt.strftime("%Y-%m")
    e = e.sort_values("date")
    monthly = e.groupby(["team", "month"], sort=False).tail(1).reset_index(drop=True)
    monthly = monthly[monthly["month"] >= start]
    return monthly[["month", "team", "elo"]]


# ---------------------------------------------------------------------
# 4. ALIGNMENT ENGINE — closest continental tournament within 18 months
# ---------------------------------------------------------------------
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]

def months_between(earlier, later):
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)

def align_continental_champion(wc_start, finals_df, window_months=18):
    out = []
    for conf in CONFEDERATIONS:
        prior = finals_df[(finals_df["confederation"] == conf) &
                          (finals_df["final_date"] < wc_start)].sort_values("final_date")
        if prior.empty:
            out.append(dict(confederation=conf, tournament=None, edition=None,
                            champion=None, runner_up=None, final_date=None,
                            gap_months=None, within_18mo=False, used_fallback=True,
                            note="no prior continental final"))
            continue
        best = prior.iloc[-1]
        gap = months_between(best["final_date"], wc_start)
        within = gap <= window_months
        out.append(dict(confederation=conf,
                        tournament=best["tournament"], edition=int(best["year"]),
                        champion=best["champion"], runner_up=best["runner_up"],
                        final_date=str(best["final_date"].date()),
                        gap_months=int(gap), within_18mo=bool(within),
                        used_fallback=bool(not within),
                        note=("" if within else "fallback: no continental final within 18 months")))
    return out


# ---------------------------------------------------------------------
# 5. TEAM -> CONFEDERATION (manual overrides + tournament inference)
# ---------------------------------------------------------------------
def build_confederation_map(matches):
    teams = set(matches["home_team"]) | set(matches["away_team"])
    conf = {t: c for t, c in MANUAL_CONF.items() if t in teams}
    by_tourn = {}
    for tourn, confs in CONF_BY_TOURNAMENT.items():
        for t in confs:
            by_tourn.setdefault(t, set()).add(tourn)
    for team in sorted(teams):
        if team in conf:
            continue
        hit = None
        for row in matches[matches["home_team"].eq(team) | matches["away_team"].eq(team)].itertuples(index=False):
            cand = by_tourn.get(str(row.tournament or ""))
            if cand:
                hit = next(iter(cand))
                break
        conf[team] = hit or "Other"
    return pd.DataFrame(sorted(conf.items()), columns=["team", "confederation"])


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------
def run(colab=True, repo=GITHUB_REPO, out_root=None, results_path=None):
    log = []
    def rec(msg):
        print(msg)
        log.append(msg)

    if colab:
        import urllib.request
        rec("[1/7] Downloading pinned match dataset  ...")
        req = urllib.request.Request(RAW_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=120).read()
        matches = pd.read_csv(io.BytesIO(raw))
    else:
        rec(f"[1/7] Loading local match dataset        ... {results_path}")
        matches = pd.read_csv(results_path)

    matches.columns = [c.strip().lower() for c in matches.columns]
    matches = matches.rename(columns={"home_team": "home_team", "away_team": "away_team",
                                      "home_score": "home_score", "away_score": "away_score",
                                      "tournament": "tournament", "neutral": "neutral",
                                      "date": "date"})
    matches["date"] = pd.to_datetime(matches["date"])
    matches["neutral"] = matches["neutral"].astype(str).str.upper().eq("TRUE")
    matches["tournament"] = matches["tournament"].fillna("").astype(str)
    rec(f"      matches={len(matches):,}  teams={matches['home_team'].nunique():,}  "
        f"range={matches['date'].min().date()}..{matches['date'].max().date()}")

    if out_root is None:
        out_root = DRIVE_DIR if colab else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "00_data")
    for d in (data_dir, stage_dir):
        os.makedirs(d, exist_ok=True)

    rec("[2/7] Confederation map                   ...")
    conf_map = build_confederation_map(matches)
    conf_map.to_csv(os.path.join(data_dir, "team_confederations.csv"), index=False)
    conf_lookup = conf_map.set_index("team")["confederation"]
    conf_counts = conf_map["confederation"].value_counts()
    rec("      coverage: " + ", ".join(f"{c}={n}" for c, n in conf_counts.items()))

    rec("[3/7] Historical canon                    ...")
    canon = pd.DataFrame(WORLD_CUPS)
    canon["winner_confederation"] = canon["winner"].map(conf_lookup).fillna("Other")
    canon["runner_up_confederation"] = canon["runner_up"].map(conf_lookup).fillna("Other")
    canon.to_csv(os.path.join(data_dir, "historical_canon.csv"), index=False)

    rec("[4/7] Continental finals                  ...")
    finals = pd.DataFrame(CONTINENTAL_FINALS).rename(columns={"conf": "confederation"})
    finals["final_date"] = pd.to_datetime(finals["final_date"])
    finals.to_csv(os.path.join(data_dir, "continental_finals.csv"), index=False)
    for u in finals[finals["champion"] == "UNKNOWN"].itertuples():
        rec(f"      ! UNKNOWN champion: {u.tournament} {u.year} (fill in before modeling)")

    rec("[5/7] Elo chronology (full history)       ...")
    elo = compute_elo(matches)
    top5 = elo.sort_values("date").groupby("team")["elo"].last().sort_values(ascending=False).head(5)
    rec("      latest top-5: " + ", ".join(f"{t} {r:.0f}" for t, r in top5.items()))

    rec("[6/7] Monthly ranking chronology 1992+    ...")
    monthly = monthly_chronology(elo, start="1992-01", min_matches=10)
    monthly.to_csv(os.path.join(data_dir, "ranking_chronology.csv"), index=False)
    latest = elo.sort_values("date").groupby("team")["elo"].last().reset_index()
    latest.columns = ["team", "elo"]
    latest.to_csv(os.path.join(data_dir, "latest_elo.csv"), index=False)
    rec(f"      ranking_chronology rows={len(monthly):,}  teams={monthly['team'].nunique():,}  "
        f"months={monthly['month'].nunique():,}")

    rec("[7/7] Alignment engine                    ...")
    canon2 = pd.DataFrame(WORLD_CUPS)
    canon2["start"] = pd.to_datetime(canon2["start"])
    align_rows = []
    for wc in canon2.itertuples():
        for row in align_continental_champion(wc.start, finals, window_months=18):
            align_rows.append(dict(wc_year=int(wc.wc_year), wc_start=str(wc.start.date()), **row))
    alignment = pd.DataFrame(align_rows)
    alignment.to_csv(os.path.join(data_dir, "alignment_engine.csv"), index=False)
    rec("      alignment for 2026 (target cycle):")
    for r in alignment[alignment["wc_year"] == 2026].itertuples():
        rec(f"        {r.confederation:>9} <- {r.tournament} {r.edition} ({r.champion}) "
            f"gap={r.gap_months}mo within18mo={r.within_18mo}")

    rec("[8/8] Manifest + write                    ...")
    manifest = data_manifest(matches, canon, finals, monthly, latest, conf_map, alignment)
    with open(os.path.join(data_dir, "data_manifest.md"), "w", encoding="utf-8") as f:
        f.write(manifest)
    source_file = "/content/stage00_data.py"
    if os.path.exists(source_file):
        shutil.copyfile(source_file, os.path.join(stage_dir, "stage00_data.py"))
    else:
        try:
            shutil.copyfile(__file__, os.path.join(stage_dir, "stage00_data.py"))
        except Exception:
            pass
    rec("      artifacts written to: " + data_dir)
    rec("      files: " + ", ".join(sorted(os.listdir(data_dir))))

    if colab:
        os.makedirs(os.path.join(data_dir, "source"), exist_ok=True)
        rec("      storing pinned results.csv (reproducibility) ...")
        with open(os.path.join(data_dir, "source", "results_" + DATA_SHA[:7] + ".csv"), "wb") as f:
            f.write(raw)
    return log, alignment, conf_map


def data_manifest(matches, canon, finals, monthly, latest, conf_map, alignment):
    w = []
    w.append("# DATA MANIFEST — stage 00")
    w.append("")
    w.append(f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w.append(f"- Match source: martj42/international_results @ {DATA_SHA[:7]}")
    w.append(f"- Matches: {len(matches):,}  ({matches['date'].min().date()} .. {matches['date'].max().date()})")
    w.append(f"- Historical canon rows: {len(canon)} (WCs 1930-2026; 2026 = prediction target)")
    w.append(f"- Continental finals rows: {len(finals)}")
    w.append(f"- Ranking chronology rows: {len(monthly):,}  teams={monthly['team'].nunique():,}  months={monthly['month'].nunique():,}")
    w.append(f"- Latest Elo rows: {len(latest):,}")
    w.append(f"- Confederation map rows: {len(conf_map):,}  Others={int((conf_map['confederation']=='Other').sum())}")
    w.append(f"- Alignment rows: {len(alignment):,}  (WC x confederation)")
    w.append("")
    w.append("## Ranking chronology note")
    w.append("Rankings are **self-computed Elo** from all international matches since 1872, NOT")
    w.append("official FIFA points. The briefing requested 'FIFA rankings from 1992'; official FIFA")
    w.append("points history is not bulk-downloadable for free, so Elo is the reproducible proxy.")
    w.append("Elo params (documented in stage00_data.py): K = 60/50/40/30 (WC/continental/")
    w.append("qualifier-competitive/friendly), home advantage +100 when not neutral, goal-margin")
    w.append("factor G = 1.0 (draw) else 1.5 + 0.125*(min(gd,5)-1). Monthly = last rating of month.")
    w.append("USSR matches are recorded as 'Russia', West Germany as 'Germany' (dataset convention),")
    w.append("so Elo series are continuous across the political transitions.")
    w.append("")
    w.append("## Alignment engine note")
    w.append("For each WC, each confederation's 'continental champion' is the winner of the most")
    w.append("recent continental final within 18 months before the WC start. If none qualifies,")
    w.append("the engine falls back to the most recent final and flags used_fallback=True.")
    w.append("For 2026, UEFA/CONMEBOL/AFC/OFC use the fallback (their trophies are ~23-28 months out).")
    w.append("")
    w.append("## Known gaps")
    w.append("- continental finals flagged UNKNOWN need results filled in before modeling:")
    for r in finals[finals["champion"] == "UNKNOWN"].itertuples():
        w.append(f"  - {r.tournament} {r.year}")
    w.append("- Oceania Nations Cup 2020 was cancelled (COVID); no row included.")
    w.append("- WC 2026 outcome is intentionally absent (prediction target).")
    return "\n".join(w) + "\n"


# ---------------------------------------------------------------------
# 7. GITHUB PUSH (Colab only) — token from Colab secrets, never in code
# ---------------------------------------------------------------------
def push_to_github(repo, token, files_root, commit_msg):
    if not repo or repo.startswith("your_github_"):
        return False, "GITHUB_REPO not set in config cell"
    if not token:
        return False, "GITHUB_TOKEN missing in Colab secrets (key icon -> add secret)"
    import tempfile
    clone_dir = "/content/soccerdl_repo"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_CONFIG_KEY_1": "user.name",
        "GIT_CONFIG_VALUE_1": "soccerdl-bot",
        "GIT_CONFIG_KEY_2": "user.email",
        "GIT_CONFIG_VALUE_2": "soccerdl-bot@users.noreply.github.com",
    })
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{repo}.git", clone_dir],
                       env=env, check=True, capture_output=True, text=True)
        shutil.copytree(files_root, clone_dir, dirs_exist_ok=True)
        subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=clone_dir, env=env,
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env,
                       check=True, capture_output=True, text=True)
        return True, "pushed to " + repo
    except subprocess.CalledProcessError as e:
        return False, "git error: " + (e.stderr or e.stdout or str(e))[:400]


# ---------------------------------------------------------------------
# 8. COLAB ENTRYPOINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Colab-safe argument parsing. Notebook kernels inject args like
    #   -f /root/.local/share/jupyter/runtime/kernel-xxx.json
    # which argparse rejects (SystemExit: 2). Parse manually, ignore the rest.
    argv = sys.argv[1:]
    IS_LOCAL = "--local" in argv
    repo = GITHUB_REPO
    results_path = None
    out_root = None
    for a in argv:
        if a.startswith("--repo="):
            repo = a.split("=", 1)[1]
        elif a.startswith("--results="):
            results_path = a.split("=", 1)[1]
        elif a.startswith("--out="):
            out_root = a.split("=", 1)[1]

    if IS_LOCAL:
        log, alignment, conf_map = run(colab=False, repo=repo,
                                       out_root=out_root, results_path=results_path)
        print("\n=== LOCAL SMOKE TEST OK ===")
        sys.exit(0)

    print("Python:", sys.version.split()[0], "| pandas:", pd.__version__, "| numpy:", np.__version__)

    # Auto-detect CLI-exec (colab-cli): marker = uploaded /tmp/github_token.
    # Interactive Colab needs the notebook Drive mount + Secrets; CLI mode
    # mounts Drive via `colab drivemount` and reads the token from the file.
    IS_CLI = os.path.exists("/tmp/github_token")

    # ---- installs (pinned; never fatal) ------------------------------------
    try:
        import subprocess as _sp
        _sp.run([sys.executable, "-m", "pip", "install", "-q",
                 "pandas==2.2.2", "numpy==2.1.1"],
                check=False, capture_output=True)
    except Exception as e:
        print("WARN: pip pin install skipped:", e)

    # ---- Drive mount + token ------------------------------------------------
    if IS_CLI:
        if not os.path.exists("/content/drive"):
            DRIVE_DIR = "/content/soccerdl_out"   # Drive not mounted: keep local
        os.makedirs(DRIVE_DIR, exist_ok=True)
        token = None
        try:
            with open("/tmp/github_token") as f:
                token = f.read().splitlines()[0].strip()   # first line only (file may carry notes after)
        except Exception as e:
            print("WARN: no /tmp/github_token:", e)
    else:
        from google.colab import drive, userdata
        drive.mount("/content/drive", force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        token = None
        try:
            token = userdata.get("GITHUB_TOKEN")
        except Exception as e:
            print("WARN: could not read GITHUB_TOKEN secret:", e)

    try:
        log, alignment, conf_map = run(colab=True, repo=repo)
    except Exception:
        import traceback
        print("\n[ERROR] run() failed. Paste the traceback below to the agent:")
        traceback.print_exc()
        sys.exit(1)

    # Assemble repo content: data/ + stages/ + README (mirror the online source of truth)
    assets = "/content/soccerdl_assets"
    if os.path.exists(assets):
        shutil.rmtree(assets)
    shutil.copytree(os.path.join(DRIVE_DIR, "data"), os.path.join(assets, "data"))
    shutil.copytree(os.path.join(DRIVE_DIR, "stages"), os.path.join(assets, "stages"))
    with open(os.path.join(assets, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_MD)

    print("\n=== AUTO-COMMIT TO GITHUB ===")
    ok, msg = push_to_github(repo, token, assets,
                             "stage 00: data foundation (canon, chronology, alignment)")
    if repo == "your_github_username/soccer-deeplearning":
        ok = False
        msg = "GITHUB_REPO not set - create your public repo and set the name in the CONFIG cell"
    print("commit:", ok, "|", msg)
