import sys
from src.common.paths import FIGURES
from src.explorer import build, collect

def main():
    print("--- Running Explorer Pipeline ---")
    try:
        if not (FIGURES / "deck" / "pipeline.png").is_file():
            raise FileNotFoundError("Missing pipeline.png. Regenerate it with python -m src.explorer.diagram")
        build.build_explorer()
        collect.collect_figures()
        print("--- Explorer Pipeline Complete ---")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
