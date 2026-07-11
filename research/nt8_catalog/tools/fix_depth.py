import os

base_dir = os.path.abspath('tests')
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith('ag_deepdive_') and file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            if "'magnitude', " in content:
                content = content.replace("'magnitude', ", "")
                with open(path, 'w') as f:
                    f.write(content)
                print('Patched', file)
