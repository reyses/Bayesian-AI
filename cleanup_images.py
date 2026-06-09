import os
import glob

# Search in the root directory and 'research' folder, avoiding 'DATA' and 'example'
directories_to_search = [
    ".",
    "research",
    "research/Regression segments",
]

extensions = ['*.jpg', '*.jpeg', '*.png', '*.svg']
files_to_delete = []

for ext in extensions:
    # Use glob to find files in current directory
    for file in glob.glob(ext):
        if 'example' not in file.lower():
            files_to_delete.append(file)
    
    # Use rglob for deep search in 'research' to skip scanning the massive DATA folder
    for root, dirs, files in os.walk('research'):
        if 'example' in root.lower():
            continue
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.svg')):
                files_to_delete.append(os.path.join(root, file))

# Remove duplicates
files_to_delete = list(set(files_to_delete))

print(f"Found {len(files_to_delete)} images to delete.")
for f in files_to_delete:
    print(f"Deleting: {f}")
    try:
        os.remove(f)
    except Exception as e:
        print(f"Error deleting {f}: {e}")

print("Cleanup complete!")
