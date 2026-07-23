# AI Money Flow

A glanceable, auto-refreshing dashboard of money flow across the AI-infrastructure
buildout — market weather, sector flow bars, a rotation map, and a plain-English
game plan. Served by GitHub Pages; a GitHub Action recomputes `data/flow.json`
every 15 minutes during US market hours (Finnhub daily bars, Yahoo fallback).

The scoring rules are a self-contained port of a private research dashboard
(`pipeline.py` — sector flow score, setup labels, regime components). All math
is transparent constants in that one file.

Research & education only. Nothing here is investment advice.
