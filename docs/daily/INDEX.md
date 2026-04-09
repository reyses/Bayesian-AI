# Daily Journal Index

| Date | Summary |
|------|---------|
| 2026-04-09 | March data filled (74 OOS days). 9 ExNMP tiers. 3-class CNN flip reverted to 2-class (cost $50K). FADE_AGAINST is poison. 5-CNN architecture defined. Tree pipeline deleted. |
| 2026-04-08 | Live hardening + pipeline + exit physics. SFE OHLCV mismatch fixed, dashboard cleaned, 8 live bugs. Oscillation discovery (100% long trades oscillate). FADE/RIDE exit modes. 3-bar confirmation. 4 ExNMP sub-tiers (fade/ride x calm/momentum). Full pipeline: $462/day IS, $340/day OOS (94% win days). |
| 2026-04-07 | CNN flip 70.6% → blended $367 IS, $397 OOS, 85% win. BASE_NMP $0.30→$7.90/trade. Entry-only CNN > path CNN. Stage 2 architecture: regret discovers new entry physics → CNN clusters → validate → expand ExNMP roster. |
| 2026-04-06 | Kill shot 96% WR. Blended engine (cascade/killshot/base). Pipeline $295 IS, $386 OOS. Tree exhausted: direction 55%, duration 64%, 5-point path 54% — can't predict BASE_NMP flips. Wick is the ONLY direction signal. Regime flip = poison (-$19/trade). Next: CNN for multi-feature patterns. |
| 2026-04-05 | Bayesian pipeline built. Root cause: counter-flip killed edge. Fix: corrected trades (regret → right dir + exit). IS $62/day (63%), OOS $51/day (52%). First real positive OOS. 17 commits. Next: DOE on 79D entry conditions. |
| 2026-04-04 | nn_v2 full pipeline. NMP→regret→tree→AI. Bulk OOS $711/day. Honest sequential built. Per-day iteration to $175/day IS. |
| 2026-04-03 | Clean data = -$2,427 (phantom spikes were fake edge). Template autopsy: 0 template matches, 100% peak reversal coin flip. Designed 79D unified feature vector (10 features x 6 TFs). NN architecture: direction + hold duration (signal half-life). 5s = atomic unit, TFs = aggregation windows. Cord length ceiling: 5% capture at 5s = $2,350/day. |
| 2026-03-29 | Levels + EDA (z_se=levels, 1m leads direction, 1h leads speed) + shapes (4 types distinct) + archive cleanup |
| 2026-03-28/29 | Oscillation research, calibration, level drawing tool, hand-drawn levels validated across month |
| 2026-03-27 | 29D pipeline built, 3-layer CNN ($602/day IS), pivoted to oscillation-aware proposal |
| 2026-03-26 | 18 live bug fixes, TradeCNN StatePredictor $1,609/day OOS, trailing stop |
| 2026-03-25 | Base measurements framework, CNN $736/day OOS, DMI flipper $208-400/day |
