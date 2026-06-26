import datetime
import pytz

def check_time(ts):
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    ct = dt.astimezone(pytz.timezone('US/Central'))
    
    # Check if we are between 15:55 and 16:00 CT
    # Or if we are approaching the weekend...
    # Friday 15:55 CT is weekend approach. 
    # Any day 15:55 CT is maintenance approach.
    is_near_maint = (ct.hour == 15 and ct.minute >= 55)
    
    # Are there any other weekend hours we should worry about? 
    # The market is closed from Friday 16:00 CT to Sunday 17:00 CT.
    # So if we exit at Friday 15:55 CT, we are flat over the weekend.
    
    print(f"Time: {ct.strftime('%Y-%m-%d %H:%M:%S %Z')}, Near maint: {is_near_maint}")

check_time(1708384170.0) # From the example
