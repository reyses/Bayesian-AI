import datetime
import pytz
import time

ts = 1708384170.0
tz = pytz.timezone('US/Central')

start = time.time()
for _ in range(80000):
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    ct = dt.astimezone(tz)
    is_near_maint = (ct.hour == 15 and ct.minute >= 55)
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")
