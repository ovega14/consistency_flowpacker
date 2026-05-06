"""Script to compare eval metrics across multiple saved runs of the full pipeline."""
import json
from pathlib import Path


def main():
    for results_file in sorted(Path('../checkpoints').glob('*/results.json')):
        r = json.load(open(results_file))[-1]  # last entry
        print(f"{results_file.parent.name}")
        print(f"  MAE={r['results']['overall_mae_deg']:.2f}°  Acc={r['results']['overall_acc_pct']:.2f}%")


if __name__ == '__main__':
    main()
