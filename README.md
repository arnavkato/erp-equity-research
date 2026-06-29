# Entropy Risk Premium in Equities

Companion code for [paper.md](paper.md) — a replication and null result.

**TL;DR:** The entropy risk premium (Chabi-Yo, Doshi & Zurita 2020) earns 18%/yr in commodity futures. In Dow 30 equities, cross-sectional corr(ERP, VRP) = 0.98 — entropy is variance in disguise. No cross-sectional equity edge.

## Replicate

```bash
pip install -r requirements.txt
# set THETA_EMAIL and THETA_PASSWORD in .env
python -m erp.vix_check              # validate L^Q against published VIX first
python -m erp.run_backtest --start 2025-01-01
python -m erp.run_paper_tests
```

## Key files

| File | What it does |
|---|---|
| `erp/core/entropy.py` | L^Q (eq 2.26), L^P cumulant estimator, BKM moments |
| `erp/core/bkm.py` | Bakshi-Kapadia-Madan risk-neutral var/skew/kurt |
| `erp/core/american.py` | Barone-Adesi-Whaley de-Americanization |
| `erp/vix_check.py` | VIX reconstruction gate — run before trusting results |
| `erp/run_paper_tests.py` | Tables 2-4 of the original paper, on equities |
23 unit tests: `python -m pytest erp/tests/`
