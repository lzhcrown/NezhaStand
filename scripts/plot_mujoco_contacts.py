#!/usr/bin/env python3
"""Plot four-wheel contact and load metrics exported by MuJoCo."""

import argparse
import csv
from pathlib import Path


WHEELS = ("FL", "FR", "RL", "RR")
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def _csv_input_path(value):
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate

    # Accept a path that already exists for backward compatibility. A bare
    # filename is looked up directly in the standard results directory.
    cwd_candidate = candidate.resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return RESULTS_DIR / candidate.name


def _figure_output_path(value, csv_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        filename = candidate.name
        if not filename.lower().endswith(".png"):
            filename += ".png"
        return RESULTS_DIR / filename
    return RESULTS_DIR / f"{csv_path.stem}.png"


def _read_csv(path):
    import numpy as np

    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    required = {"time_s", "all_four_contact", "foot_load_fraction_error_rms"}
    required.update(f"{name}_contact" for name in WHEELS)
    required.update(f"{name}_vertical_force_n" for name in WHEELS)
    required.update(f"{name}_load_fraction" for name in WHEELS)
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def plot(args):
    import numpy as np

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = _csv_input_path(args.csv)
    data = _read_csv(csv_path)
    time_s = data["time_s"]
    colors = {"FL": "#0072B2", "FR": "#D55E00", "RL": "#009E73", "RR": "#CC79A7"}

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    for index, name in enumerate(WHEELS):
        axes[0].step(
            time_s,
            data[f"{name}_contact"] + 1.25 * index,
            where="post",
            color=colors[name],
            linewidth=1.2,
            label=name,
        )
    axes[0].plot(
        time_s,
        data["all_four_contact"] * 0.9 - 1.25,
        color="black",
        linewidth=1.4,
        label="all four",
    )
    axes[0].set_ylabel("contact state\n(offset traces)")
    axes[0].set_yticks([])
    axes[0].set_ylim(-1.4, 4.9)
    axes[0].legend(ncol=5, loc="upper right")
    axes[0].grid(True, axis="x", alpha=0.25)

    for name in WHEELS:
        axes[1].plot(
            time_s,
            data[f"{name}_vertical_force_n"],
            color=colors[name],
            linewidth=1.1,
            label=name,
        )
    axes[1].set_ylabel("vertical force (N)")
    axes[1].legend(ncol=4, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    for name in WHEELS:
        axes[2].plot(
            time_s,
            data[f"{name}_load_fraction"],
            color=colors[name],
            linewidth=1.1,
            label=name,
        )
    axes[2].axhline(0.25, color="black", linestyle="--", linewidth=1.0, label="ideal 0.25")
    axes[2].set_ylabel("load fraction")
    axes[2].set_ylim(bottom=0.0)
    axes[2].legend(ncol=5, loc="upper right")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(
        time_s,
        data["foot_load_fraction_error_rms"],
        color="#E69F00",
        linewidth=1.2,
    )
    axes[3].set_ylabel("load fraction\nerror RMS")
    axes[3].set_xlabel("simulation time (s)")
    axes[3].set_ylim(bottom=0.0)
    axes[3].grid(True, alpha=0.25)

    contact_fraction = float(np.mean(data["all_four_contact"]))
    load_error_rms = float(np.sqrt(np.mean(np.square(
        data["foot_load_fraction_error_rms"]
    ))))
    figure.suptitle(
        f"Nezha MuJoCo wheel contacts | all-contact={contact_fraction:.1%}, "
        f"load-error RMS={load_error_rms:.4f}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    output = _figure_output_path(args.output, csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"rows:                 {len(time_s)}")
    print(f"all-contact fraction: {contact_fraction:.6f}")
    print(f"load-error RMS:       {load_error_rms:.6f}")
    print(f"figure:               {output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        help="CSV written by MuJoCo; a bare filename is read from results/",
    )
    parser.add_argument(
        "--output",
        help="Output PNG filename in results/ (an absolute path overrides it)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(plot(parse_args()))
