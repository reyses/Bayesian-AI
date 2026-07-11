import os
import shutil

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    archive_dir = os.path.join(base_dir, 'archive')
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        
    files_to_archive = ['AG_Joint_Model.md', 'AG_Joint_EDA.md']
    warning_banner = "> [!WARNING] INVALIDATED (AUDIT-ACC-01 §5)\n\n"
    
    for f in files_to_archive:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            with open(src, 'r') as file:
                content = file.read()
                
            dst = os.path.join(archive_dir, f)
            with open(dst, 'w') as file:
                file.write(warning_banner + content)
                
            os.remove(src)
            print(f"Moved and bannered {f}")
        else:
            print(f"File {f} not found.")
