import os
import time

mailbox_file = "comms/mailbox.md"
last_mtime = 0

if os.path.exists(mailbox_file):
    last_mtime = os.stat(mailbox_file).st_mtime

print(f"Mailbox watcher started. Monitoring {mailbox_file}...", flush=True)

while True:
    time.sleep(1)
    if os.path.exists(mailbox_file):
        current_mtime = os.stat(mailbox_file).st_mtime
        if current_mtime != last_mtime:
            last_mtime = current_mtime
            print("\n*** WAKEUP TRIGGER ***", flush=True)
            print("comms/mailbox.md has been modified!", flush=True)
            print("Please read it and execute any new tasks from Claude.", flush=True)
            print("**********************\n", flush=True)
