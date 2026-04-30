import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OUTPUT_VIDEO_DIR = Path("Output_Video")
OUTPUT_DATA_DIR = Path("Output_Data")
ANALYSIS_DIR = Path("Analysis_Output")

VIDEO_CONFIGS = [
    {
        "clip_name": "Patrick",
        "display_name": "Patrick",
        "variants": ["frames", "residuals"],
        "splats": [10, 100, 1000],
    },
    {
        "clip_name": "Office_Car_Crash",
        "display_name": "Office Car Crash",
        "variants": ["frames", "residuals"],
        "splats": [10, 100, 1000],
    },
]


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

def ensure_output_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "graphs").mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "csv").mkdir(parents=True, exist_ok=True)


def sorted_image_paths(folder: Path) -> list[Path]:
    image_extensions = {".png", ".jpg", ".jpeg"}

    paths = [
        path for path in sorted(folder.iterdir())
        if path.suffix.lower() in image_extensions
    ]

    if not paths:
        raise FileNotFoundError(f"No image files found in {folder}")

    return paths


def find_video_path(clip_name: str, variant: str, splats: int) -> Path | None:
    """
    Expected naming convention:

        [Name]_splats_[number of splats]_[frames or residuals].mp4

    Examples:

        Patrick_splats_100_frames.mp4
        Patrick_splats_100_residuals.mp4
        Patrick_splats_1000_frames.mp4
    """

    video_name = f"{clip_name}_splats_{splats}_{variant}.mp4"
    video_path = OUTPUT_VIDEO_DIR / video_name

    if video_path.exists():
        return video_path

    return None


# ------------------------------------------------------------
# Frame loading
# ------------------------------------------------------------

def load_original_frames(clip_name: str) -> list[np.ndarray]:
    """
    Loads the ground-truth frames from:

        Output_Data/[Name]/frames
    """

    frames_dir = OUTPUT_DATA_DIR / clip_name / "frames"
    frame_paths = sorted_image_paths(frames_dir)

    frames = []

    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError(f"Could not read original frame: {path}")

        frames.append(frame)

    return frames


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """
    Loads all frames from an MP4 video.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        frames.append(frame)

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    return frames


# ------------------------------------------------------------
# PSNR calculation
# ------------------------------------------------------------

def mse_between_frames(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes mean squared error between two frames.

    If dimensions differ, the reconstructed frame is resized to match the
    original frame. This is useful if a 10-splat reconstruction accidentally
    produced smaller frames.
    """

    original_h, original_w = original.shape[:2]
    recon_h, recon_w = reconstructed.shape[:2]

    if (recon_h, recon_w) != (original_h, original_w):
        reconstructed = cv2.resize(
            reconstructed,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

    original_float = original.astype(np.float64)
    reconstructed_float = reconstructed.astype(np.float64)

    mse = np.mean((original_float - reconstructed_float) ** 2)

    return float(mse)


def psnr_from_mse(mse: float, max_pixel_value: float = 255.0) -> float:
    """
    Converts MSE to PSNR.

    If MSE is zero, PSNR is mathematically infinite.
    """

    if mse == 0:
        return float("inf")

    return 10.0 * math.log10((max_pixel_value ** 2) / mse)


def psnr_between_frames(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = mse_between_frames(original, reconstructed)
    return psnr_from_mse(mse)


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

def analyze_video(
    clip_name: str,
    display_name: str,
    variant: str,
    splats: int,
    video_path: Path,
    original_frames: list[np.ndarray],
) -> tuple[list[dict], dict]:
    """
    Returns:

    1. Per-frame PSNR rows
    2. Summary row for the whole video
    """

    reconstructed_frames = load_video_frames(video_path)

    frame_count = min(len(original_frames), len(reconstructed_frames))

    if len(original_frames) != len(reconstructed_frames):
        print(
            f"WARNING: Frame count mismatch for {video_path.name}. "
            f"Original={len(original_frames)}, reconstructed={len(reconstructed_frames)}. "
            f"Using first {frame_count} frames."
        )

    per_frame_rows = []
    psnr_values = []

    for frame_index in range(frame_count):
        original = original_frames[frame_index]
        reconstructed = reconstructed_frames[frame_index]

        psnr = psnr_between_frames(original, reconstructed)

        per_frame_rows.append(
            {
                "clip_name": clip_name,
                "display_name": display_name,
                "variant": variant,
                "splats": splats,
                "video_file": video_path.name,
                "frame_index": frame_index,
                "psnr": psnr,
            }
        )

        psnr_values.append(psnr)

    finite_psnr_values = [
        value for value in psnr_values
        if math.isfinite(value)
    ]

    if not finite_psnr_values:
        mean_psnr = float("inf")
        std_psnr = 0.0
        median_psnr = float("inf")
        min_psnr = float("inf")
        max_psnr = float("inf")
    else:
        psnr_array = np.array(finite_psnr_values, dtype=np.float64)

        mean_psnr = float(np.mean(psnr_array))
        std_psnr = float(np.std(psnr_array, ddof=1)) if len(psnr_array) > 1 else 0.0
        median_psnr = float(np.median(psnr_array))
        min_psnr = float(np.min(psnr_array))
        max_psnr = float(np.max(psnr_array))

    summary_row = {
        "clip_name": clip_name,
        "display_name": display_name,
        "variant": variant,
        "splats": splats,
        "video_file": video_path.name,
        "num_original_frames": len(original_frames),
        "num_reconstructed_frames": len(reconstructed_frames),
        "num_compared_frames": frame_count,
        "mean_psnr": mean_psnr,
        "std_psnr": std_psnr,
        "median_psnr": median_psnr,
        "min_psnr": min_psnr,
        "max_psnr": max_psnr,
    }

    return per_frame_rows, summary_row


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_frame_rows = []
    all_summary_rows = []

    for config in VIDEO_CONFIGS:
        clip_name = config["clip_name"]
        display_name = config["display_name"]

        print(f"\nLoading original frames for {display_name}...")
        original_frames = load_original_frames(clip_name)

        for variant in config["variants"]:
            for splats in config["splats"]:
                video_path = find_video_path(clip_name, variant, splats)

                if video_path is None:
                    print(
                        f"Skipping missing video: "
                        f"{clip_name}_{variant}_{splats}.mp4"
                    )
                    continue

                print(f"Analyzing {video_path.name}...")

                frame_rows, summary_row = analyze_video(
                    clip_name=clip_name,
                    display_name=display_name,
                    variant=variant,
                    splats=splats,
                    video_path=video_path,
                    original_frames=original_frames,
                )

                all_frame_rows.extend(frame_rows)
                all_summary_rows.append(summary_row)

    frame_df = pd.DataFrame(all_frame_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    return frame_df, summary_df


# ------------------------------------------------------------
# Graphs
# ------------------------------------------------------------

def make_label(display_name: str, variant: str) -> str:
    variant_label = "Frames" if variant == "frames" else "Residuals"
    return f"{display_name} {variant_label}"


def plot_mean_psnr_with_error_bars(summary_df: pd.DataFrame) -> None:
    """
    Graph:

        x-axis: number of splats
        y-axis: mean PSNR
        error bars: standard deviation of frame-level PSNR
        each line: clip + variant
    """

    graph_path = ANALYSIS_DIR / "graphs" / "mean_psnr_with_error_bars.png"

    plt.figure(figsize=(10, 6))

    grouped = summary_df.groupby(["display_name", "variant"])

    for (display_name, variant), group in grouped:
        group = group.sort_values("splats")

        label = make_label(display_name, variant)

        plt.errorbar(
            group["splats"],
            group["mean_psnr"],
            yerr=group["std_psnr"],
            marker="o",
            capsize=4,
            label=label,
        )

    plt.xscale("log")
    plt.xticks([10, 100, 1000], ["10", "100", "1000"])
    plt.xlabel("Number of Gaussian Splats")
    plt.ylabel("Mean PSNR, dB")
    plt.title("Mean Reconstruction Quality by Number of Gaussian Splats")
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"Saved graph: {graph_path}")


def plot_psnr_over_time_for_clip(
    frame_df: pd.DataFrame,
    clip_name: str,
    display_name: str,
) -> None:
    """
    Graph:

        x-axis: frame number
        y-axis: PSNR
        each line: variant + splat count

    One graph per clip.
    """

    graph_path = ANALYSIS_DIR / "graphs" / f"{clip_name}_psnr_over_frame_number.png"

    clip_df = frame_df[frame_df["clip_name"] == clip_name].copy()

    if clip_df.empty:
        print(f"No frame data found for {display_name}; skipping graph.")
        return

    plt.figure(figsize=(12, 6))

    grouped = clip_df.groupby(["variant", "splats"])

    for (variant, splats), group in grouped:
        group = group.sort_values("frame_index")

        variant_label = "Frames" if variant == "frames" else "Residuals"
        label = f"{display_name} {variant_label} {splats}"

        plt.plot(
            group["frame_index"],
            group["psnr"],
            label=label,
            linewidth=1.5,
        )

    plt.xlabel("Frame Number")
    plt.ylabel("PSNR, dB")
    plt.title(f"PSNR over Frame Number: {display_name}")
    plt.grid(True, axis="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"Saved graph: {graph_path}")


# ------------------------------------------------------------
# CSV export
# ------------------------------------------------------------

def export_csvs(frame_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    frame_csv_path = ANALYSIS_DIR / "csv" / "frame_level_psnr.csv"
    summary_csv_path = ANALYSIS_DIR / "csv" / "summary_psnr.csv"
    sheets_friendly_path = ANALYSIS_DIR / "csv" / "google_sheets_summary.csv"

    frame_df.to_csv(frame_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    sheets_df = summary_df[
        [
            "display_name",
            "variant",
            "splats",
            "mean_psnr",
            "std_psnr",
            "median_psnr",
            "min_psnr",
            "max_psnr",
            "num_compared_frames",
            "video_file",
        ]
    ].copy()

    sheets_df = sheets_df.sort_values(
        by=["display_name", "variant", "splats"]
    )

    sheets_df.to_csv(sheets_friendly_path, index=False)

    print(f"Saved CSV: {frame_csv_path}")
    print(f"Saved CSV: {summary_csv_path}")
    print(f"Saved Google Sheets-friendly CSV: {sheets_friendly_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    frame_df, summary_df = run_analysis()

    if frame_df.empty or summary_df.empty:
        print("No data was analyzed. Check your video paths and filenames.")
        return

    export_csvs(frame_df, summary_df)

    plot_mean_psnr_with_error_bars(summary_df)

    plot_psnr_over_time_for_clip(
        frame_df=frame_df,
        clip_name="Patrick",
        display_name="Patrick",
    )

    plot_psnr_over_time_for_clip(
        frame_df=frame_df,
        clip_name="Office_Car_Crash",
        display_name="Office Car Crash",
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()