"""
Main entry point for the VizEngine Plugin architecture.

Usage:
    python -m tools.viz.run --plugin trade_visualizer --day 2025_01_01
    python -m tools.viz.run --plugin swing_inspector
"""
import argparse
import sys
import importlib

from tools.viz.core.engine import VizEngine

def main():
    parser = argparse.ArgumentParser(description="VizEngine Central Launcher")
    parser.add_argument('--plugin', required=True, help="Name of the plugin to load (e.g., trade_visualizer)")
    parser.add_argument('--day', default=None, help="Initial day to load (e.g. 2026_01_02)")
    parser.add_argument('--tf', default='1m', help="Timeframe for the base chart (default 1m)")
    
    # Allow unknown args to be passed to the plugin
    args, unknown = parser.parse_known_args()
    
    try:
        module = importlib.import_module(f"tools.viz.plugins.{args.plugin}")
    except ModuleNotFoundError:
        print(f"Error: Plugin '{args.plugin}' not found in tools.viz.plugins")
        sys.exit(1)
        
    if not hasattr(module, 'get_plugin'):
        print(f"Error: Plugin '{args.plugin}' must define a 'get_plugin()' function.")
        sys.exit(1)
        
    plugin_instance = module.get_plugin(unknown)
    
    engine = VizEngine(plugin=plugin_instance, initial_day=args.day, tf=args.tf)
    engine.run()

if __name__ == '__main__':
    main()
