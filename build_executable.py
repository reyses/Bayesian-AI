import PyInstaller.__main__
import os
import sys
import platform
import subprocess

def run_verification():
    print("Running environment verification...")
    try:
        # Import the verification function directly to avoid subprocess issues if possible
        # but subprocess is safer to ensure clean state or separation
        result = subprocess.run([sys.executable, "verify_environment.py"], capture_output=False)
        if result.returncode != 0:
            print("Verification failed. Aborting build.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running verification: {e}")
        sys.exit(1)

def build():
    run_verification()

    print("Starting PyInstaller build...")

    # Determine separator for add-data
    sep = os.pathsep

    # Define arguments
    args = [
        'engine_core.py',
        '--onedir',
        '--name=ProjectX_Engine',
        f'--add-data=config{sep}config',
        f'--add-data=probability_table.pkl{sep}.',
        '--hidden-import=numba',
        '--hidden-import=llvmlite',
        '--collect-all=numba',
        '--collect-all=llvmlite',
        '--contents-directory=.', # Force flat structure in dist/ProjectX_Engine
        '--clean',
        '--noconfirm',
    ]

    # Print args for debugging
    print(f"PyInstaller arguments: {args}")

    PyInstaller.__main__.run(args)

    print("Build complete.")

    # Verify output exists
    dist_dir = os.path.join("dist", "ProjectX_Engine")
    exe_name = "ProjectX_Engine"
    if platform.system() == "Windows":
        exe_name += ".exe"

    exe_path = os.path.join(dist_dir, exe_name)

    if os.path.exists(exe_path):
        print(f"[SUCCESS] Executable found at: {exe_path}")
        print(f"Distribution directory: {dist_dir}")
        print("Required files check:")

        # Check for config dir in dist (should be in root now)
        config_dist = os.path.join(dist_dir, "config")
        if os.path.isdir(config_dist):
            print(f"  [OK] config directory: {config_dist}")
        else:
             print(f"  [ERROR] config directory missing in {dist_dir}")

        # Check for probability_table.pkl in dist (should be in root now)
        pkl_dist = os.path.join(dist_dir, "probability_table.pkl")
        if os.path.exists(pkl_dist):
             print(f"  [OK] probability_table.pkl: {pkl_dist}")
        else:
             print(f"  [ERROR] probability_table.pkl missing in {dist_dir}")

    else:
        print(f"[ERROR] Executable NOT found at: {exe_path}")
        sys.exit(1)

if __name__ == "__main__":
    build()
