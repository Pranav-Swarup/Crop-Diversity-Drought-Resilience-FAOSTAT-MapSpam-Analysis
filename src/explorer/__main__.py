import sys
from src.explorer import diagram, build, collect

def main():
    print("--- Running Explorer Pipeline ---")
    try:
        diagram.generate_diagram()
        build.build_explorer()
        collect.collect_figures()
        print("--- Explorer Pipeline Complete ---")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
