import os
import shutil
import glob
from pathlib import Path

# Base directories
BASE_DIR = Path(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI")
RESEARCH_DIR = BASE_DIR / "research"
FINDINGS_DIR = BASE_DIR / "reports" / "findings"

# Topic mapping rules based on filename prefixes/keywords
# Format: (Keyword tuple, Topic Directory Name)
TOPIC_RULES = [
    (("kalman",), "kalman_entry"),
    (("nmp",), "nmp_strategies"),
    (("edge_case",), "edge_case_triage"),
    (("phase", "regime", "smep", "segment"), "regime_clustering"),
    (("l5", "ldist"), "l5_distribution"),
    (("orange_line",), "orange_line_eda"),
    (("chaos",), "chaos_precursors"),
    (("geometric",), "geometric_exits"),
    (("surface", "orthogonal"), "response_surfaces"),
    (("kt1", "oracle"), "oracle_tests"),
]

def get_topic(filename: str) -> str:
    """Matches a filename to a topic directory."""
    name_lower = filename.lower()
    for keywords, topic in TOPIC_RULES:
        for keyword in keywords:
            if keyword in name_lower:
                return topic
    return "misc_archive"

def move_file(src_path: Path, dest_dir: Path):
    """Moves a file to the destination directory, creating it if necessary."""
    if not src_path.exists():
        return
        
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    
    # Handle conflicts gracefully
    if dest_path.exists():
        print(f"Skipping {src_path.name} - already exists in {dest_dir}")
        return
        
    shutil.move(str(src_path), str(dest_path))
    print(f"Moved {src_path.name} -> {dest_dir.relative_to(BASE_DIR)}")

def migrate_scripts():
    """Move all python scripts in research/ to their topic/tools folder."""
    print("\n--- Migrating Scripts ---")
    for file_path in glob.glob(str(RESEARCH_DIR / "*.py")):
        src = Path(file_path)
        topic = get_topic(src.name)
        dest_dir = RESEARCH_DIR / topic / "tools"
        move_file(src, dest_dir)

def migrate_reports():
    """Move all files in reports/findings/ to their topic/reports folder."""
    print("\n--- Migrating Reports ---")
    for root, _, files in os.walk(FINDINGS_DIR):
        for file in files:
            src = Path(root) / file
            topic = get_topic(src.name)
            
            # Special case: If it was in an edge_cases subfolder, ensure it goes to edge_case_triage
            if "edge_case" in root.lower() and topic == "misc_archive":
                topic = "edge_case_triage"
                
            dest_dir = RESEARCH_DIR / topic / "reports"
            move_file(src, dest_dir)

def setup_discipline_files():
    """Create README.md and project.md in each populated topic folder."""
    print("\n--- Creating Project Stubs ---")
    for item in RESEARCH_DIR.iterdir():
        if item.is_dir() and item.name not in ["misc_archive"]:
            readme = item / "README.md"
            if not readme.exists():
                readme.write_text(f"# {item.name}\n\nIndex of tools and reports for {item.name}.")
            
            project = item / "project.md"
            if not project.exists():
                project.write_text(f"# {item.name} - DMAIC/PDCA\n\n## Define\n\n## Measure\n\n## Analyze\n\n## Improve\n\n## Control\n")

if __name__ == "__main__":
    migrate_scripts()
    migrate_reports()
    setup_discipline_files()
    print("\nMigration Complete.")
