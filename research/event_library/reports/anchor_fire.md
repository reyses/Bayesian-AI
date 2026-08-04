# Anchor fire — detector sanity check on 2024_09_16

`2024_09_16` is the live-sim day: EXCLUDED from every table in this package. It is used here only to check that each detector fires on the tape state the owner named.

## 1. ULTRA_CHOP

fires: 37  |  in anchor window 10:23:50-10:24:31: 0

- 10:12:38  box 10.00pt  flips 32  ambient 17.00pt ratio 0.59  escape 38s dir +1
- 10:14:20  box 10.00pt  flips 30  ambient 17.00pt ratio 0.59  escape 60s dir +1
- 10:17:06  box 9.75pt  flips 32  ambient 17.00pt ratio 0.57  escape 20s dir -1
- 10:18:27  box 9.25pt  flips 30  ambient 17.12pt ratio 0.54  escape 109s dir -1
- 10:22:20  box 9.25pt  flips 30  ambient 17.62pt ratio 0.52  escape 83s dir -1
- 10:36:13  box 10.25pt  flips 34  ambient 18.00pt ratio 0.57  escape 23s dir +1
- 10:45:09  box 10.75pt  flips 30  ambient 18.38pt ratio 0.59  escape 41s dir -1
- 10:57:23  box 10.25pt  flips 30  ambient 17.75pt ratio 0.58  escape 114s dir -1
- 11:03:12  box 9.75pt  flips 32  ambient 17.75pt ratio 0.55  escape 18s dir -1
- 11:09:57  box 7.50pt  flips 30  ambient 17.75pt ratio 0.42  escape 33s dir +1
- 11:15:35  box 9.00pt  flips 30  ambient 17.75pt ratio 0.51  escape 151s dir +1
- 11:19:02  box 9.50pt  flips 32  ambient 17.25pt ratio 0.55  escape 306s dir +1
- 11:28:19  box 8.75pt  flips 30  ambient 16.75pt ratio 0.52  escape 254s dir +1
- 11:38:38  box 8.50pt  flips 30  ambient 15.38pt ratio 0.55  escape 130s dir -1
- 11:43:04  box 7.50pt  flips 30  ambient 15.12pt ratio 0.50  escape 63s dir -1
- 11:46:48  box 8.50pt  flips 30  ambient 14.75pt ratio 0.58  escape 93s dir -1
- 11:49:44  box 7.50pt  flips 31  ambient 14.00pt ratio 0.54  escape 34s dir -1
- 11:51:14  box 8.25pt  flips 30  ambient 13.88pt ratio 0.59  escape 34s dir +1
- 11:54:00  box 8.00pt  flips 33  ambient 13.50pt ratio 0.59  escape 34s dir +1
- 11:58:31  box 5.50pt  flips 30  ambient 13.75pt ratio 0.40  escape 7s dir +1

anchor-window 60s stats: box 15.50-24.00pt, flips 27-33
day RTH reference:       box p50 11.00 p90 21.00pt, flips p50 27 p90 32

## 2. LEG_DESCENT

defended pushes: 85  |  chain_n>=2: 53  |  in anchor 09:56-10:24: 11

- 09:31:55  N=2  step 19659.00->19643.75 (15.25pt)  chain descent 28.50pt  race NEW_LOW
- 09:32:50  N=3  step 19634.75->19623.25 (11.50pt)  chain descent 49.00pt  race NEW_LOW
- 09:34:45  N=4  step 19592.75->19579.75 (13.00pt)  chain descent 92.50pt  race NEW_LOW
- 09:39:55  N=2  step 19607.50->19594.25 (13.25pt)  chain descent 24.00pt  race NEW_LOW
- 09:42:05  N=3  step 19601.50->19590.25 (11.25pt)  chain descent 28.00pt  race STAIR_BREAK
- 09:56:35  N=1  step 19673.75->19661.00 (12.75pt)  chain descent 12.75pt  race STAIR_BREAK
- 09:57:05  N=1  step 19693.50->19679.75 (13.75pt)  chain descent 13.75pt  race NEW_LOW
- 09:59:35  N=2  step 19673.00->19663.00 (10.00pt)  chain descent 30.50pt  race NEW_LOW
- 10:01:20  N=1  step 19686.50->19670.75 (15.75pt)  chain descent 15.75pt  race NEW_LOW
- 10:02:40  N=2  step 19682.50->19666.75 (15.75pt)  chain descent 19.75pt  race NEW_LOW
- 10:05:30  N=1  step 19686.75->19674.75 (12.00pt)  chain descent 12.00pt  race STAIR_BREAK
- 10:08:10  N=2  step 19687.25->19676.50 (10.75pt)  chain descent 10.25pt  race NEW_LOW
- 10:11:25  N=3  step 19681.75->19671.50 (10.25pt)  chain descent 15.25pt  race NEW_LOW
- 10:16:25  N=1  step 19692.25->19680.50 (11.75pt)  chain descent 11.75pt  race NEW_LOW
- 10:19:35  N=2  step 19676.25->19666.50 (9.75pt)  chain descent 25.75pt  race NEW_LOW
- 10:23:15  N=3  step 19665.75->19653.75 (12.00pt)  chain descent 38.50pt  race NEW_LOW
- 10:26:15  N=4  step 19642.00->19622.00 (20.00pt)  chain descent 70.25pt  race NEW_LOW
- 10:27:40  N=5  step 19623.00->19608.25 (14.75pt)  chain descent 84.00pt  race NEW_LOW
- 10:30:10  N=6  step 19620.00->19606.75 (13.25pt)  chain descent 85.50pt  race NEW_LOW
- 10:32:55  N=7  step 19588.25->19575.00 (13.25pt)  chain descent 117.25pt  race NEW_LOW
- 10:34:00  N=8  step 19585.75->19573.00 (12.75pt)  chain descent 119.25pt  race NEW_LOW
- 10:34:40  N=9  step 19546.75->19536.25 (10.50pt)  chain descent 156.00pt  race NEW_LOW
- 10:43:25  N=2  step 19589.75->19578.50 (11.25pt)  chain descent 23.75pt  race NEW_LOW
- 10:45:20  N=3  step 19590.25->19577.25 (13.00pt)  chain descent 25.00pt  race NEW_LOW
- 10:48:15  N=4  step 19571.00->19556.50 (14.50pt)  chain descent 45.75pt  race NEW_LOW
- 10:54:35  N=2  step 19600.00->19580.00 (20.00pt)  chain descent 19.75pt  race NEW_LOW
- 10:56:00  N=3  step 19585.75->19567.75 (18.00pt)  chain descent 32.00pt  race STAIR_BREAK
- 10:57:40  N=4  step 19585.25->19575.25 (10.00pt)  chain descent 24.50pt  race STAIR_BREAK
- 11:00:20  N=2  step 19572.25->19544.25 (28.00pt)  chain descent 44.25pt  race NEW_LOW
- 11:00:50  N=3  step 19564.50->19553.00 (11.50pt)  chain descent 35.50pt  race NEW_LOW
- 11:02:10  N=4  step 19559.75->19547.50 (12.25pt)  chain descent 41.00pt  race NEW_LOW
- 11:03:30  N=5  step 19559.50->19541.00 (18.50pt)  chain descent 47.50pt  race NEW_LOW
- 11:06:45  N=6  step 19560.50->19549.75 (10.75pt)  chain descent 38.75pt  race STAIR_BREAK
- 11:21:00  N=2  step 19623.00->19611.50 (11.50pt)  chain descent 11.50pt  race NEW_LOW
- 11:29:30  N=2  step 19623.50->19612.75 (10.75pt)  chain descent 27.25pt  race NEW_LOW
- 11:35:40  N=2  step 19628.75->19613.00 (15.75pt)  chain descent 15.25pt  race NEW_LOW
- 11:44:10  N=3  step 19591.50->19577.50 (14.00pt)  chain descent 50.75pt  race NEW_LOW
- 11:45:55  N=4  step 19581.00->19566.00 (15.00pt)  chain descent 62.25pt  race NEW_LOW
- 11:50:20  N=5  step 19578.50->19562.50 (16.00pt)  chain descent 65.75pt  race NEW_LOW
- 12:03:20  N=2  step 19602.50->19591.50 (11.00pt)  chain descent 14.50pt  race NEW_LOW
- 12:16:20  N=2  step 19590.00->19577.25 (12.75pt)  chain descent 35.50pt  race NEW_LOW
- 12:36:35  N=2  step 19596.50->19586.00 (10.50pt)  chain descent 18.25pt  race NEW_LOW
- 12:41:40  N=3  step 19591.00->19581.25 (9.75pt)  chain descent 23.00pt  race NEW_LOW
- 12:45:20  N=4  step 19585.00->19573.75 (11.25pt)  chain descent 30.50pt  race STAIR_BREAK
- 12:47:15  N=5  step 19586.50->19575.00 (11.50pt)  chain descent 29.25pt  race NEW_LOW
- 12:59:50  N=2  step 19622.00->19613.25 (8.75pt)  chain descent 6.75pt  race STAIR_BREAK
- 13:02:10  N=3  step 19621.50->19610.75 (10.75pt)  chain descent 9.25pt  race NEW_LOW
- 13:14:50  N=4  step 19621.25->19610.25 (11.00pt)  chain descent 9.75pt  race NEW_LOW
- 13:21:40  N=5  step 19608.00->19592.25 (15.75pt)  chain descent 27.75pt  race STAIR_BREAK
- 13:33:55  N=2  step 19630.00->19622.00 (8.00pt)  chain descent 7.00pt  race NEW_LOW
- 14:01:55  N=2  step 19662.25->19650.50 (11.75pt)  chain descent 18.00pt  race NEW_LOW
- 14:07:40  N=3  step 19653.25->19639.00 (14.25pt)  chain descent 29.50pt  race STAIR_BREAK
- 14:25:00  N=2  step 19667.50->19657.25 (10.25pt)  chain descent 12.00pt  race NEW_LOW
- 14:38:25  N=2  step 19666.50->19651.00 (15.50pt)  chain descent 20.25pt  race NEW_LOW
- 14:45:10  N=3  step 19648.50->19630.50 (18.00pt)  chain descent 40.75pt  race NEW_LOW
- 15:04:35  N=2  step 19666.25->19656.50 (9.75pt)  chain descent 11.25pt  race NEW_LOW
- 15:12:25  N=3  step 19667.25->19656.50 (10.75pt)  chain descent 11.25pt  race NEW_LOW
- 15:21:25  N=4  step 19652.25->19641.00 (11.25pt)  chain descent 26.75pt  race NEW_LOW

## 3. FAKEOUT_POKE

armed pokes: 264  |  RETURN (the event): 148  |  STUCK: 0  |  BREAKOUT: 116

- 09:43:10  dir +1  ref 19607.50  poke +0.50pt  race RESUME  exceed_ref True
- 09:45:15  dir +1  ref 19622.75  poke +2.00pt  race RESUME  exceed_ref True
- 09:48:25  dir +1  ref 19627.50  poke +1.25pt  race RESUME  exceed_ref True
- 09:50:45  dir +1  ref 19650.00  poke +1.00pt  race RESUME  exceed_ref True
- 09:52:50  dir +1  ref 19659.00  poke +0.50pt  race REVERSE  exceed_ref True
- 09:55:10  dir +1  ref 19672.25  poke +1.50pt  race RESUME  exceed_ref True
- 09:56:35  dir -1  ref 19663.00  poke +1.25pt  race REVERSE  exceed_ref True
- 09:57:30  dir -1  ref 19670.50  poke +0.25pt  race RESUME  exceed_ref True
- 09:58:45  dir -1  ref 19664.75  poke +2.00pt  race REVERSE  exceed_ref True
- 09:59:20  dir +1  ref 19672.25  poke +0.75pt  race RESUME  exceed_ref True
- 10:00:00  dir -1  ref 19663.75  poke +0.75pt  race REVERSE  exceed_ref True
- 10:02:40  dir -1  ref 19668.50  poke +0.25pt  race RESUME  exceed_ref True
- 10:03:30  dir -1  ref 19645.50  poke +0.25pt  race REVERSE  exceed_ref True
- 10:04:25  dir +1  ref 19673.00  poke +0.25pt  race RESUME  exceed_ref True
- 10:05:25  dir +1  ref 19686.50  poke +0.25pt  race RESUME  exceed_ref True

## 4. STALL

candidates: 63  |  STALL (giveback<=30% for 10min): 0


## 5. DEFENDED_POKE_AT_SHELF

events: 5

- 09:58:00  shelf 19671.00 (dwell 13%)  poke 19661.25  bounce 13.5pt  class flushV  outcome HOLD
- 10:30:00  shelf 19671.00 (dwell 13%)  poke 19586.00  bounce 36.2pt  class flushV  outcome CRACK
- 14:23:00  shelf 19657.00 (dwell 8%)  poke 19655.75  bounce 10.5pt  class flushV  outcome HOLD
- 14:55:00  shelf 19657.00 (dwell 11%)  poke 19634.00  bounce 16.8pt  class flushV  outcome HOLD
- 15:28:00  shelf 19657.00 (dwell 11%)  poke 19645.50  bounce 8.5pt  class flushV  outcome HOLD

## 6. FLUSH_V_DAY

- is_flush=True  confirm 09:50:00  flush 110.2pt  rec 85%  v_low 19560.75 v_peak 19654.25  first PEAK_RECLAIM  close_frac 1.17

