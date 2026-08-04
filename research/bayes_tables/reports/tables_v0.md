# BAYESIAN TABLE ACTUARY v0

Counted, not trained. Every cell is a Beta posterior whose prior is the event's own global rate with 20 pseudo-counts (hierarchical shrinkage), so a thin cell is pulled toward the base rate rather than shouting from 3 samples. A cell is ACTIONABLE only when it survives BOTH a day-clustered bootstrap (4,000 draws resampling DAYS, because events inside one session are correlated and the iid Beta interval is too narrow) AND Benjamini-Hochberg FDR control at q=0.05 across every cell in the table, with n >= 15. Live sim day excluded.

## fakeout_poke — P(exceed_ref_first == True)

N = 153,029 events, global rate 0.782, 198 cells, 155 ACTIONABLE

    kind dir_s depth_b age_b clock_b    n  days    raw  day_lo  day_hi   post    lift  actionable
BREAKOUT    up     2-5 5-30m    1200 3842   519 0.9258  0.9168  0.9347 0.9251  0.1429        True
BREAKOUT    up     2-5 5-30m    1030 3563   523 0.8984  0.8868  0.9095 0.8978  0.1155        True
BREAKOUT    up     2-5  30m+    1030 3477   494 0.9177  0.9073  0.9271 0.9170  0.1348        True
BREAKOUT    up     2-5  30m+    1200 3432   507 0.9175  0.9069  0.9280 0.9168  0.1345        True
BREAKOUT    dn     2-5 5-30m    1200 3215   510 0.9061  0.8961  0.9161 0.9053  0.1231        True
BREAKOUT    dn     2-5  30m+    1200 2847   484 0.9031  0.8917  0.9141 0.9022  0.1200        True
BREAKOUT    dn     2-5 5-30m    1030 2844   515 0.8822  0.8698  0.8942 0.8815  0.0993        True
BREAKOUT    dn     2-5  30m+    1030 2662   481 0.8775  0.8638  0.8909 0.8768  0.0946        True
BREAKOUT    up     2-5 5-30m    1400 2586   501 0.9138  0.9018  0.9253 0.9128  0.1305        True
  RETURN    up     1-2 5-30m    1200 2253   504 0.7230  0.7008  0.7443 0.7236 -0.0587        True
BREAKOUT    dn     2-5 5-30m    1400 2216   490 0.9057  0.8943  0.9169 0.9046  0.1224        True
  RETURN    up     1-2 5-30m    1030 2204   507 0.6842  0.6630  0.7052 0.6851 -0.0971        True

**Strongest actionable cells (|lift| vs global):**

  kind dir_s depth_b age_b clock_b   n  days    raw  day_lo  day_hi   post    lift       p
RETURN    dn     1-2  30m+    0930 475   243 0.5621  0.5162  0.6061 0.5710 -0.2112 0.00025
RETURN    up   0.5-1 5-30m    0930 428   275 0.5771  0.5271  0.6244 0.5863 -0.1960 0.00025
RETURN    dn   <=0.5 5-30m    0930 414   259 0.5845  0.5386  0.6321 0.5937 -0.1886 0.00025
RETURN    up   <=0.5  30m+    0930 289   177 0.5813  0.5188  0.6418 0.5943 -0.1879 0.00025
RETURN    up   0.5-1  30m+    1000 265   170 0.5962  0.5316  0.6603 0.6093 -0.1729 0.00025
RETURN    dn   <=0.5  30m+    0930 248   169 0.5968  0.5357  0.6579 0.6106 -0.1716 0.00025

## fakeout_poke — P(sym_race == CONT)

N = 153,029 events, global rate 0.498, 70 cells, 2 ACTIONABLE

    kind dir_s depth_b clock_b    n  days    raw  day_lo  day_hi   post    lift  actionable
BREAKOUT    up     2-5    1200 9133   529 0.5071  0.4941  0.5209 0.5070  0.0094       False
BREAKOUT    up     2-5    1030 9086   534 0.4997  0.4846  0.5139 0.4997  0.0020       False
BREAKOUT    dn     2-5    1200 7624   524 0.4934  0.4788  0.5076 0.4935 -0.0042       False
BREAKOUT    dn     2-5    1030 7171   532 0.4910  0.4757  0.5057 0.4910 -0.0067       False
BREAKOUT    up     2-5    1400 5999   514 0.4679  0.4512  0.4840 0.4680 -0.0297        True
  RETURN    up     1-2    1030 5695   531 0.5243  0.5083  0.5397 0.5242  0.0265        True
  RETURN    up     1-2    1200 5507   528 0.5119  0.4957  0.5277 0.5118  0.0142       False
  RETURN    dn     1-2    1030 5305   523 0.5084  0.4921  0.5251 0.5083  0.0107       False
BREAKOUT    dn     2-5    1400 5144   506 0.4868  0.4705  0.5030 0.4868 -0.0109       False
  RETURN    dn     1-2    1200 5122   523 0.5158  0.4976  0.5336 0.5157  0.0181       False
  RETURN    up     1-2    1400 3860   502 0.4736  0.4560  0.4915 0.4737 -0.0240       False
  RETURN    dn     1-2    1400 3640   489 0.4926  0.4741  0.5110 0.4926 -0.0051       False

**Strongest actionable cells (|lift| vs global):**

    kind dir_s depth_b clock_b    n  days    raw  day_lo  day_hi   post    lift       p
BREAKOUT    up     2-5    1400 5999   514 0.4679  0.4512  0.4840 0.4680 -0.0297 0.00025
  RETURN    up     1-2    1030 5695   531 0.5243  0.5083  0.5397 0.5242  0.0265 0.00100

## leg_descent — P(race == NEW_LOW)

N = 58,480 events, global rate 0.690, 15 cells, 15 ACTIONABLE

defense_b chain_b    n  days    raw  day_lo  day_hi   post    lift  actionable
       d1       1 7052   538 0.8049  0.7958  0.8139 0.8046  0.1147        True
       d1      3+ 5296   535 0.8021  0.7918  0.8128 0.8017  0.1119        True
       d3       1 5212   527 0.7064  0.6936  0.7191 0.7064  0.0166        True
       d5       1 5152   485 0.4899  0.4725  0.5062 0.4907 -0.1991        True
       d4       1 4564   517 0.6295  0.6148  0.6444 0.6298 -0.0601        True
       d1       2 4096   537 0.8071  0.7952  0.8191 0.8066  0.1168        True
       d3      3+ 3851   525 0.7138  0.6983  0.7290 0.7137  0.0239        True
       d2       1 3605   533 0.7678  0.7537  0.7818 0.7674  0.0776        True
       d5      3+ 3332   466 0.4964  0.4773  0.5155 0.4976 -0.1923        True
       d4      3+ 3284   496 0.6389  0.6239  0.6538 0.6392 -0.0506        True
       d3       2 2911   516 0.7104  0.6936  0.7275 0.7103  0.0205        True
       d5       2 2783   443 0.5131  0.4921  0.5335 0.5144 -0.1754        True

**Strongest actionable cells (|lift| vs global):**

defense_b chain_b    n  days    raw  day_lo  day_hi   post    lift       p
       d5       1 5152   485 0.4899  0.4725  0.5062 0.4907 -0.1991 0.00025
       d5      3+ 3332   466 0.4964  0.4773  0.5155 0.4976 -0.1923 0.00025
       d5       2 2783   443 0.5131  0.4921  0.5335 0.5144 -0.1754 0.00025
       d1       2 4096   537 0.8071  0.7952  0.8191 0.8066  0.1168 0.00025
       d1       1 7052   538 0.8049  0.7958  0.8139 0.8046  0.1147 0.00025
       d1      3+ 5296   535 0.8021  0.7918  0.8128 0.8017  0.1119 0.00025

## stall — P(race == NEW_EXTREME)

N = 41,180 events, global rate 0.103, 10 cells, 9 ACTIONABLE

give_b dir_s    n  days    raw  day_lo  day_hi   post    lift  actionable
    g5    up 4506   455 0.0082  0.0055  0.0111 0.0086 -0.0947        True
    g2    dn 4419   531 0.0860  0.0777  0.0951 0.0861 -0.0172        True
    g3    dn 4337   529 0.0433  0.0373  0.0492 0.0436 -0.0597        True
    g1    up 4289   536 0.3917  0.3773  0.4063 0.3904  0.2871        True
    g4    dn 4234   510 0.0163  0.0122  0.0204 0.0167 -0.0866        True
    g4    up 4010   508 0.0244  0.0199  0.0292 0.0248 -0.0785        True
    g1    dn 3957   535 0.2990  0.2837  0.3137 0.2980  0.1947        True
    g3    up 3899   530 0.0559  0.0491  0.0628 0.0562 -0.0471        True
    g2    up 3807   534 0.1011  0.0910  0.1115 0.1011 -0.0022       False
    g5    dn 3722   442 0.0043  0.0022  0.0065 0.0048 -0.0985        True

**Strongest actionable cells (|lift| vs global):**

give_b dir_s    n  days    raw  day_lo  day_hi   post    lift       p
    g1    up 4289   536 0.3917  0.3773  0.4063 0.3904  0.2871 0.00025
    g1    dn 3957   535 0.2990  0.2837  0.3137 0.2980  0.1947 0.00025
    g5    dn 3722   442 0.0043  0.0022  0.0065 0.0048 -0.0985 0.00025
    g5    up 4506   455 0.0082  0.0055  0.0111 0.0086 -0.0947 0.00025
    g4    dn 4234   510 0.0163  0.0122  0.0204 0.0167 -0.0866 0.00025
    g4    up 4010   508 0.0244  0.0199  0.0292 0.0248 -0.0785 0.00025

## ultra_chop — P(escape_dir == 1)

N = 18,601 events, global rate 0.509, 15 cells, 10 ACTIONABLE

midbox_b ratio_b    n  days    raw  day_lo  day_hi   post    lift  actionable
      q3   loose 3277   516 0.4968  0.4794  0.5134 0.4969 -0.0117       False
  q1_low   loose 3239   519 0.3652  0.3480  0.3818 0.3661 -0.1424        True
      q2   loose 3126   510 0.4546  0.4370  0.4717 0.4549 -0.0536        True
 q5_high   loose 3107   515 0.6579  0.6419  0.6752 0.6569  0.1484        True
      q4   loose 3076   515 0.5793  0.5632  0.5960 0.5789  0.0703        True
      q3     mid  518   316 0.5444  0.5028  0.5871 0.5431  0.0345       False
      q2     mid  495   297 0.4141  0.3724  0.4536 0.4178 -0.0907        True
  q1_low     mid  494   311 0.3583  0.3169  0.3992 0.3641 -0.1444        True
 q5_high     mid  494   302 0.6478  0.6000  0.6913 0.6424  0.1338        True
      q4     mid  482   310 0.5871  0.5435  0.6303 0.5840  0.0755        True
      q3   tight   71    67 0.4648  0.3478  0.5857 0.4744 -0.0341       False
      q2   tight   66    60 0.4242  0.3030  0.5455 0.4438 -0.0647       False

**Strongest actionable cells (|lift| vs global):**

midbox_b ratio_b    n  days    raw  day_lo  day_hi   post    lift       p
 q5_high   loose 3107   515 0.6579  0.6419  0.6752 0.6569  0.1484 0.00025
  q1_low     mid  494   311 0.3583  0.3169  0.3992 0.3641 -0.1444 0.00025
  q1_low   loose 3239   519 0.3652  0.3480  0.3818 0.3661 -0.1424 0.00025
  q1_low   tight   53    50 0.3208  0.1961  0.4528 0.3722 -0.1363 0.00800
 q5_high     mid  494   302 0.6478  0.6000  0.6913 0.6424  0.1338 0.00025
 q5_high   tight   38    38 0.6842  0.5263  0.8158 0.6236  0.1151 0.03150

## defended_poke_shelf — P(outcome == CRACK)

N = 1,585 events, global rate 0.374, 8 cells, 5 ACTIONABLE

bounce_b day_class   n  days    raw  day_lo  day_hi   post    lift  actionable
      b4     other 358   261 0.1369  0.1035  0.1728 0.1494 -0.2247        True
      b1     other 346   196 0.5665  0.5108  0.6213 0.5560  0.1818        True
      b2     other 328   217 0.4909  0.4394  0.5429 0.4841  0.1100        True
      b3     other 328   214 0.3018  0.2507  0.3547 0.3060 -0.0681        True
      b1    flushV  71    40 0.5775  0.4923  0.6620 0.5328  0.1586        True
      b2    flushV  71    46 0.4507  0.3544  0.5570 0.4339  0.0597       False
      b3    flushV  56    46 0.2500  0.1321  0.3800 0.2827 -0.0915       False
      b4    flushV  27    24 0.0370  0.0000  0.1111 0.1805 -0.1937       False

**Strongest actionable cells (|lift| vs global):**

bounce_b day_class   n  days    raw  day_lo  day_hi   post    lift       p
      b4     other 358   261 0.1369  0.1035  0.1728 0.1494 -0.2247 0.00025
      b1     other 346   196 0.5665  0.5108  0.6213 0.5560  0.1818 0.00025
      b1    flushV  71    40 0.5775  0.4923  0.6620 0.5328  0.1586 0.00025
      b2     other 328   217 0.4909  0.4394  0.5429 0.4841  0.1100 0.00025
      b3     other 328   214 0.3018  0.2507  0.3547 0.3060 -0.0681 0.00900


---

# THE GEOMETRY CONTROL — what these tables actually measure

An adversarial audit (`audit_v0.md`) overturned the v1 claim that "direction
carries nothing": position-in-box predicts ultra-chop escape direction at
AUC 0.620, P(up) running 0.364 -> 0.657 across quintiles — a bigger spread
than anything in the v1 tables. Rebuilt on that variable (and on give_frac,
defense_pt, bounce_pt, which the audit showed likewise dominate the clock
dims I had shipped), the tables sharpened everywhere: chop 0.36->0.66,
stall 39%->0.4%, leg_descent 80%->49%.

Then the obvious question: is any of it PREDICTION, or is it distance to a
barrier?

## Test 1 — escape direction vs a driftless random walk

For a driftless walk between two absorbing barriers, P(hit upper first) =
d_down / (d_up + d_down). Computed per event from the box edges and the
escape buffer:

| | value |
|---|---|
| observed P(up) | 0.5085 |
| geometric null | 0.5094 |
| **mean excess over geometry** | **-0.0009, day-clustered 95% CI [-0.0078, +0.0063]** |

Decile by decile the observed rate tracks the geometric prediction (excess
+0.03 to -0.03, no systematic sign). **The entire AUC-0.620 "edge" is the
distance ratio.** Nothing is left once geometry is priced in.

## Test 2 — symmetric races, where geometry cancels by construction

The event library also stamps a SYMMETRIC race (+-10pt both directions) for
every event. Barrier distance is equal by definition, so any surviving
spread is real information:

| dim | cells | result |
|---|---|---|
| stall x give_frac quintiles | 5 | 0.4885 - 0.5085, ONE cell separates (g4, by 1.2pt — 1-in-5 at 95% is chance) |
| leg_descent x defense_pt quintiles | 5 | 0.4887 - 0.5047, NONE separate |

Global symmetric rates: stall 0.5002 [0.4949, 0.5053], leg_descent 0.4962
[0.4910, 0.5014]. **Every spread vanishes.**

## What this means

1. **The big table spreads are first-passage geometry, not forecasting.**
   A cell reading "09:30 RETURN pokes clear only 57%" is telling you the
   poke sits far from its target and near its kill barrier — not that the
   market is about to do something.
2. **The 0.57 wall stands** — now for the eighth independent time, and this
   is the cleanest demonstration yet, because the geometry-free version of
   the SAME events lands on 50.0% with tight intervals.
3. **The tables are still worth having — as a RISK instrument.** "Given
   where price sits between my levels, what do I hit first?" is exactly the
   question a stop/target/ratchet decision asks, and the tables answer it
   with measured probabilities instead of intuition. That is the same
   verdict the program reached for the owner's exit protocol (a variance
   machine, not an EV machine) and for the +-N bracket study (dead as a
   strategy, alive as a risk control).
4. **Do not build an alpha engine on these cells.** Any student model
   trained to predict them learns the distance ratio — which is already
   computable in closed form, for free, with no model.
