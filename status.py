import glob, os, datetime

files = glob.glob('artifacts/stage1_segments_*.json')
print(f'Total Processed Days: {len(files)}')

if files:
    latest = max(files, key=os.path.getmtime)
    print(f'Most Recent File: {latest}')
    print(f'Last Update Time: {datetime.datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M:%S")}')
