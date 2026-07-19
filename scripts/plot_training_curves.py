"""Render training/validation curves from the training history for the README.

Reads model/training_artifacts/training_history.csv and writes a two-panel
figure (accuracy and loss, train vs. validation) to docs/training_curves.png.
The training artifacts are gitignored, so this committed PNG is how the curves
reach the README.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "model" / "training_artifacts" / "training_history.csv"
OUT = ROOT / "docs" / "training_curves.png"

TRAIN = "#2563eb"  # blue
VAL = "#f97316"    # orange
TOP3 = "#94a3b8"   # slate


def main() -> None:
    df = pd.read_csv(HISTORY)
    epochs = df["epoch"] + 1

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Accuracy panel
    ax_acc.plot(epochs, df["accuracy"], color=TRAIN, label="train top-1")
    ax_acc.plot(epochs, df["val_accuracy"], color=VAL, label="val top-1")
    ax_acc.plot(epochs, df["val_top_3_accuracy"], color=TOP3,
                linestyle="--", label="val top-3")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.6, 1.0)

    # Loss panel
    ax_loss.plot(epochs, df["loss"], color=TRAIN, label="train")
    ax_loss.plot(epochs, df["val_loss"], color=VAL, label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("categorical cross-entropy")

    for ax in (ax_acc, ax_loss):
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.margins(x=0.01)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
