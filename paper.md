# Entropy Risk Premium in Equities: A Replication and Null Result

**Arnav Katoch**  
June 2026

---

## Abstract

We test whether the entropy risk premium (ERP) of Chabi-Yo, Doshi, and Zurita (2020) — a signal that earns 18% per year in commodity futures — translates to US equities. Using the Dow Jones Industrial Average constituents (30 names, January 2025–June 2026), ThetaData option chains, and the paper's exact methodology (Barone-Adesi-Whaley de-Americanization, Carr-Madan log-contract integration, Bakshi-Kapadia-Madan risk-neutral moments, 60-day realized cumulant estimator, Fama-MacBeth cross-sectional regressions), we find that the entropy risk premium is empirically indistinguishable from the variance risk premium in equities (cross-sectional correlation 0.98) and carries no predictive power for the cross-section of equity returns beyond variance. We validate our risk-neutral entropy implementation against the published VIX (within 0.1–0.7 vol points on real SPX chains) before drawing conclusions. The null result is structural: in commodity futures, heterogeneous hedging demand generates large independent skewness premia (SRP = 17.6% per year in the original paper); in liquid equities, the higher-moment premia carry no independent cross-sectional information and entropy collapses to a noisy proxy for implied variance.

---

## 1. Introduction

Chabi-Yo, Doshi, and Zurita (2020) introduce the *entropy risk premium* — the difference between a security's entropy under the physical (P) and risk-neutral (Q) measures:

$$\text{ERP}_t = L^P_t[R] - L^Q_t[R]$$

where $L_t[R] = \log E_t[R] - E_t[\log R]$ is the Kullback-Leibler entropy of the gross return $R$ (Case 1, the log specification, $\rho \to 1$). Higher ERP predicts higher expected return: commodities with higher ERP earn 18%/yr in a long-short sort, with t-statistic 6.37, and the signal survives controlling for variance, skewness, and kurtosis risk premia.

The economic intuition is compelling: ERP is the distributional-moment-complete generalization of the variance risk premium (VRP), collapsing to $\frac{1}{2}\text{VRP}$ under lognormality. It prices the *whole* distribution, not just the second moment, making it the Kelly-correct objective for a security seller and a theoretically superior cross-sectional signal.

We ask: does this extend to equities? The question matters because (1) equity options are far more liquid and widely-traded than commodity options, (2) the theoretical framework is asset-class-agnostic, and (3) if ERP adds information beyond VRP in equities, it represents a new and distinct cross-sectional signal.

Our answer is no, and the reason is structural rather than incidental.

---

## 2. Methodology

### 2.1 Risk-Neutral Entropy

The risk-neutral entropy has a closed form in terms of option prices (Carr-Madan spanning, paper eq. 2.26, Appendix A.4):

$$L^Q_t[R] = \log(R_f) - (R_f - 1) + R_f \left[ \int_F^\infty \frac{C_t(K)}{K^2} dK + \int_0^F \frac{P_t(K)}{K^2} dK \right]$$

where $R_f = e^{r\tau}$, $F$ is the forward price, and $C_t(K)$, $P_t(K)$ are present-value OTM option mids. We use the Appendix A.4 sign convention ($-(R_f-1)$, not the main-text $+(R_f-1)$).

**Implementation:** We convert American equity options to European prices via Barone-Adesi-Whaley (1987) before any parity or strip computation. The forward $F$ is inferred from put-call parity (not spot) to avoid dividend-yield estimation. The IV smile is interpolated with a cubic spline in implied-vol space with flat-wing extrapolation, then resampled on a dense $K$-grid ($\pm 8$ forward standard deviations, 2000 points) before trapezoidal integration. Two bracketing expiries are interpolated to a constant 30-day maturity.

**Validation (critical gate):** We reconstruct the published VIX from real SPX chains on five dates spanning vol regimes (14–30 VIX). Our reconstructed VIX is within 0.1–0.7 vol points of the published value on all dates, with a small consistent low bias attributable to quote timing and smoother spline integration versus CBOE's discrete sum. Every result below rests on this validated implementation.

### 2.2 Physical Entropy

Following the paper, we estimate $L^P_t[R]$ from the exact cumulant identity:

$$L^P_t[R] = \sum_{n \geq 2} \frac{\kappa_n}{n!}$$

where $\kappa_n$ are the cumulants of the horizon log-return $X = \log R$. Under iid daily returns, $\kappa_n(\text{horizon}) = h \cdot \kappa_n(\text{daily})$, so we estimate the daily cumulants from $h = 60$ trailing trading-day returns and scale to the 30-day option horizon. This is exact algebra, not an approximation; we retain through 4th-order cumulants (Gaussian through order 2 is optimal for this universe). We verify the estimator recovers $\frac{1}{2}\sigma^2\tau$ on lognormal returns, equals the direct horizon estimator on Gaussian samples, and converges to the direct estimator on jump-diffusion samples.

The **60-day trailing window** matches the paper's specification ("we proxy for the physical risk measure using the realized moment with a 60-day rolling window"). Using a 365-day window (our initial error) mis-states the paper and kills the signal by mismatching the timescale of $L^P$ (stale, one-year-backward) with $L^Q$ (current, forward-looking). The 60-day window correctly keeps both measures on the same timescale.

### 2.3 BKM Risk-Neutral Moments

We implement Bakshi, Kapadia, and Madan (2003) to compute risk-neutral variance, skewness, and kurtosis from the same OTM option strip:

$$V = R_f \int \frac{2(1 - \log(K/F))}{K^2} O(K) dK, \quad W = R_f \int \frac{6\log(K/F) - 3(\log(K/F))^2}{K^2} O(K) dK$$

The VRP, SRP, and KRP are the physical-minus-risk-neutral differences for each moment (same convention as ERP). We validate BKM against the lognormal closed form (VAR = $\sigma^2\tau$, SKEW = 0, KURT = 3).

### 2.4 Fama-MacBeth Cross-Sectional Regressions

Following the paper (Table 4), we run monthly cross-sectional regressions of next-period returns on ERP and controls:

$$R_{i,t+1} = a_t + b_t \cdot \text{ERP}_{i,t} + c_t \cdot \text{VRP}_{i,t} + \ldots + \varepsilon_{i,t+1}$$

The Fama-MacBeth estimate is the time-series mean of the monthly slopes, with Newey-West (5-lag HAC) t-statistics. We report **plain (non-HAC) t-statistics alongside** because HAC on 17 monthly observations systematically inflates by approximately 2× in our sample; we validate this via leave-one-out analysis. Signals are cross-sectionally standardized each month.

---

## 3. Data

- **Universe:** 30 Dow Jones Industrial Average constituents (current membership; the last reconstitution adding NVDA/SHW was November 2024, so current membership is essentially point-in-time for our sample).
- **Option chains:** ThetaData, EOD NBBO bid/ask at 15:59 ET, all strikes and maturities. Options are American (single-name); we de-Americanize before any computation.
- **Sample:** January 2025–June 2026 (17 monthly rebalances for the cross-section; 23 months for the extended VRP swap panel starting July 2024).
- **Risk-free rate:** ThetaData SOFR (continuously compounded).
- **Stock prices:** ThetaData EOD adjusted close (1-year history depth on the free tier).

---

## 4. Results

### 4.1 VIX Reconstruction

| Date | Reconstructed VIX | Published VIX | Diff |
|---|---|---|---|
| 2026-06-25 | 18.62 | 18.89 | −0.27 |
| 2026-03-20 | 26.04 | 26.78 | −0.74 |
| 2025-11-21 | 23.20 | 23.43 | −0.23 |
| 2025-08-15 | 14.40 | 15.09 | −0.69 |
| 2025-04-17 | 29.51 | 29.65 | −0.14 |

*The risk-neutral entropy strip correctly prices the variance swap. All downstream results rest on this validated implementation.*

### 4.2 ERP–VRP Collinearity

The central finding, visible before any predictability test:

**Cross-sectional corr(ERP, VRP) = 0.98** averaged across months.

In the original paper's commodity sample, this correlation is 0.81. The difference is structural: in commodity futures, commercial hedgers pay a large skewness premium (SRP = 17.6% per year in the paper) that is largely independent of the variance premium, creating cross-sectional variation in ERP that VRP does not capture. In liquid equities, the higher-moment premia carry no independent information, so entropy ≈ variance in the cross-section.

For comparison:

| Setting | corr(ERP, VRP) | ERP post-VRP |
|---|---|---|
| Commodity futures (paper) | 0.81 | Still significant (t≈2.3) |
| Dow 30 equities (this paper) | **0.98** | Insignificant (t=0.90) |

### 4.3 Cross-Sectional Predictability (Table 2 equivalent)

Raw ERP quartile long-short (P4 minus P1, monthly rebalance):

| Window | L-S mean/mo | Plain NW t | Monotonic |
|---|---|---|---|
| 365-day physical (initial) | −0.39% | −0.25 | No |
| 60-day physical (corrected) | +0.77% | +0.71 | **Yes** |

The sign flips to correct when the physical window is fixed to the paper's 60-day specification. The quartile ordering becomes monotonic (Q1: +0.75%, Q2: +1.09%, Q3: +1.51%, Q4: +1.52%). However, the t-statistic remains insignificant and the spread between Q1 and Q4 is economically small.

### 4.4 Orthogonalized ERP (Table 3 equivalent)

Following the paper, we cross-sectionally regress ERP on VRP (and then VRP+SRP+KRP) each month and sort on the residual:

| Signal | L-S mean/mo | Plain t | Monotonic |
|---|---|---|---|
| ERP ⊥ VRP | +0.99% | +1.26 | No |
| ERP ⊥ (VRP, SRP, KRP) | +1.67% | +1.58 | No |

The ERP residuals show larger point estimates than raw ERP, consistent with the paper's finding that ERP has information beyond variance. But the sorts are non-monotonic and t-statistics are insignificant. In the commodity paper, the same orthogonalization yields t=4.76 (⊥VRP) and t=2.30 (⊥VRP,SRP,KRP). The t-ratios in equities are approximately 3× smaller for a sample that is 15× shorter, consistent with a real but small-to-zero effect.

### 4.5 Fama-MacBeth Regressions (Table 4 equivalent)

| Specification | ERP $\hat{\gamma}$ | HAC t | Plain t | VRP t |
|---|---|---|---|---|
| ERP only | +0.00699 | +3.71* | +1.53 | — |
| ERP + VRP | +0.04864 | +1.17 | +0.90 | −1.02 |
| ERP + VRP + SRP + KRP | +0.07548 | +1.06 | +0.95 | −0.97 |

*HAC t is inflated on 17 observations; plain t and leave-one-out range [1.16, 2.15] are the reliable statistics.*

The ERP slope is positive in all specifications — the sign is correct. But once VRP is controlled, the ERP slope falls to plain t=0.90 (p≈0.38), confirming that ERP carries no information beyond variance in equities. In the commodity paper, the ERP coefficient survives all controls with t>3. The VRP coefficient itself is insignificant, consistent with the 0.98 collinearity with ERP.

### 4.6 SPX Variance Risk Premium (supplementary)

![SPX VRP 30-day rolling](vol/spx_vrp.png)

For context, the aggregate implied-minus-realized gap is real and persistent at the index level: VIX exceeds trailing 30-day realized vol on **88% of days** in our sample (mean +3.8 vol points). The April 2025 tariff shock is the only major inversion. The premium exists; the cross-sectional equity signal does not.

---

## 5. Discussion

### 5.1 Why it Works in Commodities but Not Equities

The paper's central finding — ERP carries information beyond VRP — relies on the commodity market generating large, independent higher-moment premia. Physical commodity hedgers buy crash/skew protection for genuine inventory/delivery risk, creating a real skewness risk premium (SRP = 17.6% per year in the paper) that is mostly independent of the variance premium (corr(ERP,VRP)=0.81 because the ERP captures this skew dimension). When you sort on ERP, you're partly sorting on who is paying large skew premia.

In liquid equities, this mechanism is absent:
- Options markets are populated by sophisticated dealers and diversified institutional hedgers, not commercial hedgers with inventory exposure
- The low-volatility anomaly (high-IV names tend to underperform risk-adjusted) partially offsets any VRP cross-sectional effect
- Single-name equity skew demand is primarily earnings-driven and event-specific, not a persistent structural cross-sectional feature

### 5.2 Limitations

- **Sample length:** 17 monthly observations. Statistical power is low; we cannot rule out a real effect at t≈1.5.
- **Universe:** 30 names is too few for a clean cross-sectional test (7 names per quartile). The paper uses 24 commodities with bi-monthly data over 25+ years.
- **No dividends:** Physical entropy uses price-return close series. For the Dow 30 (modest dividend yields), the effect on L^P is second-order but nonzero.
- **Data tier:** ThetaData free tier limits stock history to ~1 year, blocking deeper realized-moment estimation.

### 5.3 HAC Inflation Warning

Newey-West HAC t-statistics on 17 monthly observations are systematically inflated ~2× relative to plain t-statistics. All NW t-statistics above should be halved before comparison to standard significance thresholds. This is not specific to our dataset; it is a known finite-sample property of HAC estimators.

---

## 6. Conclusion

The entropy risk premium, implemented faithfully and validated against the published VIX, fails to predict the cross-section of Dow 30 equity returns beyond the variance risk premium. The structural reason is that in equities, cross-sectional corr(ERP,VRP)=0.98, so entropy is variance in disguise — the higher-moment premia that make ERP distinct in commodities carry no independent cross-sectional information in equities.

This is a clean negative result, not an implementation failure. The validated measurement stack (option strip, BAW de-Americanization, BKM moments, realized cumulant estimator, Fama-MacBeth) is published in the companion repository and replicates the paper's methodology faithfully on real equity option data.

---

## References

- Chabi-Yo, Doshi, and Zurita (2020). "Never a Dull Moment: Entropy Risk in Commodity Markets."
- Bakshi, Kapadia, and Madan (2003). "Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options." *Review of Financial Studies*.
- Barone-Adesi and Whaley (1987). "Efficient Analytic Approximation of American Option Values." *Journal of Finance*.
- Carr and Madan (2001). "Towards a Theory of Volatility Trading." In *Option Pricing, Interest Rates and Risk Management*.
- Martin (2017). "What Is the Expected Return on the Market?" *Quarterly Journal of Economics*.
- Harvey, Liu, and Zhu (2016). "… and the Cross-Section of Expected Returns." *Review of Financial Studies*.

---

## Appendix: Repository Structure

```
erp/
  core/
    entropy.py          # L^Q (eq 2.26), L^P cumulant estimator, BKM moments
    bkm.py              # Bakshi-Kapadia-Madan risk-neutral var/skew/kurt
    blackscholes.py     # Black-76 pricing + implied vol
    american.py         # Barone-Adesi-Whaley de-Americanization
    forward.py          # Parity-implied forward (not spot)
    smile.py            # IV-space spline, flat-wing extrapolation, dense grid
    constant_maturity.py
  data/
    thetadata.py        # ThetaData loader (OptionDataSource protocol)
    base.py             # OptionDataSource protocol, OptionChain dataclass
  pipeline/
    erp_snapshot.py     # Per-name ERP + VRP/SRP/KRP, live or backtest
    erp_backtest.py     # Monthly cross-sectional backtest with parallel compute
  stats/
    fama_macbeth.py     # Fama-MacBeth cross-sectional regression
    newey_west.py       # HAC t-statistic
  tests/                # 23 unit tests (entropy identities, BKM, BAW, pipeline)
  vix_check.py          # VIX reconstruction gate (run before trusting L^Q)
  run_backtest.py       # Monthly backtest CLI
  run_paper_tests.py    # Tables 2-4 equivalent
  run_fama_macbeth.py   # Fama-MacBeth with controls
  config.py
vol/
  spx_vrp.py            # SPX VRP plot (implied vs realized, 30-day rolling)
```

**Requirements:** Python 3.11+, numpy, scipy, pandas, statsmodels, thetadata, python-dotenv, matplotlib.

**To replicate:** Set `THETA_EMAIL` and `THETA_PASSWORD` in `.env`, then:
```bash
python -m erp.vix_check                          # validate L^Q implementation
python -m erp.run_backtest --start 2025-01-01    # build the panel
python -m erp.run_paper_tests                    # Tables 2-4
```
