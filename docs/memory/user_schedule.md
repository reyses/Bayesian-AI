---
name: user_schedule
description: User's daily schedule — wake, work, available hours for trading/dev
type: user
---

- Wake: 5:30 AM PST
- Leave for work: 6:45 AM PST
- Return from work: 6:00 PM PST
- Sleep: 10:00-11:00 PM PST

Available dev time: 5:30-6:45 AM (1h 15m) + 6:00-11:00 PM (5h) = ~6h/day
Market hours (CME): 3:00 PM - 2:00 PM PST next day (23h, 1h maintenance 2-3 PM)
Overlap with user awake: 5:30-6:45 AM + 6:00-11:00 PM = system runs unattended most of the day

Implications:
- System MUST run autonomously during work hours (6:45 AM - 6:00 PM)
- Morning session (5:30-6:45): quick checks, restart if crashed, review overnight results
- Evening session (6:00-11:00): main dev/research time, monitor live
- Flag time at 10:00 PM — remind user to wrap up
