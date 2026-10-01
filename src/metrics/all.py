"""Run the whole results section: metrics -> stats -> robustness -> figures.

Run: python -m src.metrics.all        (USE_DUMMY=1 for dummy inputs)
"""
from . import figures, robustness, run, stats


def main():
    for step in (run, stats, robustness, figures):
        print(f"\n===== {step.__name__} =====")
        step.main()


if __name__ == "__main__":
    main()
