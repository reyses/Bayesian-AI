import re
import os

def update_architecture():
    with open("docs/ARCHITECTURE.md", "r") as f:
        content = f.read()

    # We will find all code blocks or tables that have lines count
    # Let's match table rows with file path
    lines = content.split('\n')

    total_lines = 0
    new_lines = []

    for line in lines:
        if line.startswith('| `') and '|' in line:
            # typical format: | `core/statistical_field_engine.py` | 489 | `StatisticalFieldEngine` | ... |
            parts = line.split('|')
            if len(parts) >= 4:
                filepath = parts[1].strip().strip('`')
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        file_lines = len(f.readlines())
                    parts[2] = f" {file_lines} "
                    line = '|'.join(parts)
                    total_lines += file_lines
        elif "## Total:" in line:
            line = f"## Total: ~{total_lines:,} lines across core/, live/, training/, visualization/"

        new_lines.append(line)

    with open("docs/ARCHITECTURE.md", "w") as f:
        f.write('\n'.join(new_lines))

if __name__ == "__main__":
    update_architecture()
