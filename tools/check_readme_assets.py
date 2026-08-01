from __future__ import annotations

from pathlib import Path

EXPECTED = [
    "Figs/framework.png",
    "Figs/utp9k.jpg",
    "Figs/motion.jpg",
    "Figs/comparison.jpg",
    "Figs/Demonstration.mov",
    "Figs/task_a_results.png",
    "Figs/task_b_results.png",
    "Figs/task_a_false_positive.png",
]


def main() -> None:
    missing = [path for path in EXPECTED if not Path(path).exists()]
    if missing:
        print("Missing README assets:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    print("All README assets are present.")


if __name__ == "__main__":
    main()
