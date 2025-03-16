"""Main CLI entry point for owui-pipelines."""

import argparse
import sys


def main():
    """Run the owui-pipelines CLI."""
    parser = argparse.ArgumentParser(description="OWUI Pipelines")
    parser.add_argument("--version", action="store_true", help="Print version information")
    
    args = parser.parse_args()
    
    if args.version:
        from owui_pipelines import __version__
        print(f"owui-pipelines version {__version__}")
        return 0
        
    # Add your CLI logic here
    
    return 0


if __name__ == "__main__":
    sys.exit(main())