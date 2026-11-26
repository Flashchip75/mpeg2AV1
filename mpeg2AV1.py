#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TASK_ID = int(os.environ.get("SLURM_PROCID", 0))
TOTAL_TASKS = int(os.environ.get("SLURM_NTASKS", 1))
JOB_ID = os.environ.get("SLURM_JOB_ID", "manual")
NODE = os.environ.get("SLURMD_NODENAME", "local")

PROGRESS_INTERVAL = 60       # seconds between progress lines
SKIP_MARKERS = ("_temp", "_debug", "_av1_test")

VERBOSE = False

THREADS_RANGE = (1, 1024)
CRF_RANGE = (0, 63)
PRESET_RANGE = (0, 13)

MAX_CONSECUTIVE_FAILURES = 3

# State needed by the signal handler
active_ffmpeg_proc: subprocess.Popen | None = None
active_tmp_output: Path | None = None


# --------------------------------------------------------------------------- #
# Logging / formatting helpers
# --------------------------------------------------------------------------- #

def log(msg: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}][T{TASK_ID}] {msg}", flush=True)


def vlog(msg: str = "") -> None:
    """Log only in verbose mode."""
    if VERBOSE:
        log(f"  [v] {msg}")


def fmt_secs(s: float) -> str:
    """Format seconds into an HH:MM:SS string."""
    s = int(max(s, 0))
    return f"{s // 3600:d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def fmt_size(n: float) -> str:
    """Convert a byte count into a human-readable string (B, KB, MB, GB, TB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def tmp_output_path(file_path: Path) -> Path:
    """Generate a temporary output file path on the same filesystem for atomic replacement."""
    return file_path.with_name(f".{file_path.stem}_av1_{JOB_ID}_t{TASK_ID}.mkv")


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #

def cleanup_active() -> None:
    """Remove any active temporary files."""
    if active_tmp_output is not None:
        active_tmp_output.unlink(missing_ok=True)


def graceful_exit(signum, frame):
    """Stop FFmpeg, clean up temporary output, and exit."""
    log(f"Caught signal {signum}. Stopping FFmpeg...")

    proc = active_ffmpeg_proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            log("FFmpeg didn't exit in time. Forcing kill...")
            proc.kill()
            proc.wait()

    cleanup_active()

    log("Clean exit. Original file untouched.")
    # os._exit: a prefetch copy in the worker thread can't be interrupted,
    # so sys.exit() would hang waiting for it.
    os._exit(128 + signum)


signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)


# --------------------------------------------------------------------------- #
# ffprobe helpers
# --------------------------------------------------------------------------- #

def probe_video(file_path: Path) -> dict | None:
    """Return metadata of the first video stream, or None."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,pix_fmt",
        "-of", "json",
        str(file_path),
    ]
    vlog(f"probe: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = json.loads(res.stdout).get("streams", [])
        return streams[0] if streams else None
    except Exception as e:
        log(f"  [Warning] ffprobe failed: {e}")
        return None


def get_video_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def build_ffmpeg_cmd(local_input: Path, tmp_output: Path, 
                     threads: int, crf: int, preset: int) -> list[str]:
    return [
        "ffmpeg", "-y", "-nostdin", "-nostats",
        "-loglevel", "warning",
        "-progress", "pipe:1",
        "-i", str(local_input),
        # Preserve all streams and metadata
        "-map", "0",
        "-map_metadata", "0",
        "-map_chapters", "0",
        # Copy everything, then re-encode the primary video stream
        "-c", "copy",
        "-c:v:0", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", "yuv420p10le",
        "-g", "250",
        "-svtav1-params", f"lp={threads}",
        str(tmp_output),
    ]


def run_ffmpeg_with_progress(cmd: list[str], total_dur: float) -> int:
    """Run FFmpeg and periodically report encoding progress."""
    global active_ffmpeg_proc

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, text=True, bufsize=1,
    )

    active_ffmpeg_proc = proc
    assert proc.stdout is not None

    start = time.time()
    last_print = 0.0
    info: dict[str, str] = {}

    for line in proc.stdout:
        key, _, val = line.strip().partition("=")
        info[key] = val

        if key != "progress":
            continue

        now = time.time()
        if info["progress"] == "continue" and now - last_print < PROGRESS_INTERVAL:
            continue
        last_print = now

        try:
            done = int(info.get("out_time_us", info.get("out_time_ms", "0"))) / 1e6
        except ValueError:
            done = 0.0

        elapsed = now - start
        speed = done / elapsed if elapsed > 0 else 0
        pct = f"{100 * done / total_dur:5.1f}%" if total_dur > 0 else "  ?  "
        eta = fmt_secs((total_dur - done) / speed) if speed > 0 and total_dur else "?"

        log(
            f"  {pct} | video {fmt_secs(done)}/{fmt_secs(total_dur)} | "
            f"fps {info.get('fps', '?')} | speed {info.get('speed', '?')} | "
            f"elapsed {fmt_secs(elapsed)} | ETA {eta}"
        )

    proc.wait()
    rc = proc.returncode
    active_ffmpeg_proc = None
    return rc

def verify_output(output: Path, original_duration: float) -> bool:
    """Verify the encoded file before it replaces the original."""
    log("  Verifying output...")

    if not output.exists() or output.stat().st_size == 0:
        log("  [Error] Output file missing or empty.")
        return False

    metadata = probe_video(output)
    if not metadata or metadata.get("codec_name") != "av1":
        log("  [Error] Output video codec is not AV1.")
        return False

    new_duration = get_video_duration(output)
    if original_duration <= 0 or new_duration <= 0:
        log("  [Error] Could not determine input/output duration.")
        return False

    difference = abs(original_duration - new_duration)
    vlog(f"duration: input {original_duration:.2f}s, output {new_duration:.2f}s")
    
    tolerance = max(0.5, original_duration * 0.01)

    if difference > tolerance:
        log(
            f"  [Error] Duration mismatch: "
            f"{original_duration:.2f}s vs {new_duration:.2f}s."
        )
        return False

    # Fully decode the video stream to catch corruption
    log("  Checking video decoding integrity...")
    cmd = [
        "ffmpeg", "-v", "error", "-xerror",
        "-i", str(output),
        "-map", "0:v:0",
        "-f", "null", "-",
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if res.returncode != 0:
            log(f"  [Error] Decode verification failed: {res.stderr[-1000:]}")
            return False
    except Exception as e:
        log(f"  [Error] Verification failed: {e}")
        return False

    log(f"  Verification successful. Duration difference: {difference:.3f}s")
    return True


def convert_to_av1(file_path: Path, threads: int, crf: int, preset: int) -> tuple[bool, int, int]:
    """Encode an already-staged file and atomically replace the original."""
    global active_tmp_output

    tmp_output = tmp_output_path(file_path)
    active_tmp_output = tmp_output

    try:
        total_dur = get_video_duration(file_path)
        if total_dur <= 0:
            log("  [Error] Invalid input duration.")
            return False, 0 ,0

        cmd = build_ffmpeg_cmd(file_path, tmp_output, threads, crf, preset)
        log(f"  Encoding MPEG-2 to AV1 (CRF={crf}, preset={preset}, threads={threads})")
        vlog(f"ffmpeg: {' '.join(cmd)}")

        t0 = time.time()
        rc = run_ffmpeg_with_progress(cmd, total_dur)
        if rc != 0:
            log(f"  [Error] FFmpeg exited with code {rc}.")
            return False, 0 ,0
        vlog(f"encode took {fmt_secs(time.time() - t0)}")

        if not verify_output(tmp_output, total_dur):
            log("  [Error] Verification failed. Original kept.")
            return False, 0 ,0

        original_size = file_path.stat().st_size
        output_size = tmp_output.stat().st_size

        # Atomic replacement: the original is only replaced after verification
        if output_size >= original_size:
            log(
                f"  Output is not smaller: {fmt_size(original_size)} -> "
                f"{fmt_size(output_size)}. Keeping original."
            )
            return False, original_size, output_size

        log("  Success: original replaced with verified AV1 file.")
        log(
            f"  Size: {fmt_size(original_size)} -> {fmt_size(output_size)} "
            f"({100 * output_size / original_size:.1f}%)"
        )
        return True, original_size, output_size

    except Exception as e:
        log(f"  [Error] Conversion failed: {e}")
        return False, 0, 0

    finally:
        tmp_output.unlink(missing_ok=True)
        active_tmp_output = None


# --------------------------------------------------------------------------- #
# Library processing
# --------------------------------------------------------------------------- #

def find_mkv_files(library_dir: Path) -> list[Path]:
    """Recursively search for valid .mkv video files, excluding skip-marked entries."""
    files = []
    for root, dirs, names in os.walk(library_dir, followlinks=False,):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            low = name.lower()
            if low.endswith(".mkv") and not any(k in low for k in SKIP_MARKERS) and not name.startswith("."):
                files.append(Path(root) / name)
    return sorted(files)


def log_file_header(i: int, total: int, file_path: Path, library_dir: Path) -> None:
    log("-" * 60)
    log(f"[{i}/{total}] {file_path.relative_to(library_dir)}")


def log_metadata(meta: dict) -> None:
    log(
        f"  Codec: {meta.get('codec_name', 'unknown')} | "
        f"Resolution: {meta.get('width', '?')}x{meta.get('height', '?')} | "
        f"Pixel format: {meta.get('pix_fmt', '?')}"
    )


def dry_run_library(library_dir: Path, my_files: list[Path], threads: int, 
                    crf: int, preset: int) -> None:
    """Report what a real run would do. Only reads (ffprobe + stat)."""
    log("DRY RUN: nothing will be copied, encoded or replaced.")

    would_convert = skipped = failed = 0
    total_bytes = 0
    largest = 0

    for i, file_path in enumerate(my_files, 1):
        log_file_header(i, len(my_files), file_path, library_dir)

        meta = probe_video(file_path)
        if meta is None:
            log("  [Error] Could not inspect video stream.")
            failed += 1
            continue

        log_metadata(meta)
        codec = meta.get("codec_name", "unknown")

        if codec != "mpeg2video":
            log(f"  Would skip: codec is {codec}.")
            skipped += 1
            continue

        size = file_path.stat().st_size
        total_bytes += size
        would_convert += 1

        tmp_out = tmp_output_path(file_path)
        log(f"  Would convert ({fmt_size(size)}).")
        vlog(f"would encode to: {tmp_out}")
        vlog(f"would replace: {file_path}")
        vlog(
            "ffmpeg: "
            + " ".join(build_ffmpeg_cmd(file_path, tmp_out, threads, crf, preset))
        )

    log("=" * 60)
    log(
        f"Dry run done. would_convert={would_convert} "
        f"skipped={skipped} failed={failed}"
    )
    log(f"Total input data to process: {fmt_size(total_bytes)}")


def process_library(library_dir: Path, threads: int, crf: int, 
                    preset: int, dry_run: bool) -> None:
    """Traverse the video library, manage background prefetch staging, and execute conversions."""
    try:
        mkv_files = find_mkv_files(library_dir)
    except OSError as e:
        log(f"[Error] Directory traversal failed: {e}")
        return

    my_files = mkv_files[TASK_ID::TOTAL_TASKS]
    log(f"Assigned {len(my_files)} of {len(mkv_files)} total files.")
    if not my_files:
        return

    if dry_run:
        dry_run_library(library_dir, my_files, threads, crf, preset)
        return

    converted = skipped = failed = 0
    consecutive_failures = 0
    total_original_bytes = 0
    total_output_bytes = 0

    t_start = time.time()

    for i, file_path in enumerate(my_files, 1):
        log_file_header(i, len(my_files), file_path, library_dir)

        meta = probe_video(file_path)

        if meta is None:
            log("  [Error] Could not inspect video stream.")
            failed += 1
            consecutive_failures += 1

        else:
            log_metadata(meta)

            codec = meta.get("codec_name", "unknown")

            if codec != "mpeg2video":
                log(f"  Skipping: codec is {codec}.")
                skipped += 1
                consecutive_failures = 0

            else:
                success, orig_sz, out_sz = convert_to_av1(
                    file_path, threads, crf, preset
                )

                if success:
                    converted += 1
                    consecutive_failures = 0
                    total_original_bytes += orig_sz
                    total_output_bytes += out_sz
                else:
                    failed += 1
                    consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log(
                f"[Error] {consecutive_failures} failures in a row; "
                "aborting this task (probably a systematic problem)."
            )
            break

    elapsed = time.time() - t_start

    log("=" * 60)
    log(f"Done. converted={converted} skipped={skipped} failed={failed}")
    log(f"Total time elapsed: {fmt_secs(elapsed)}")
    
    saved_bytes = total_original_bytes - total_output_bytes
    reduction_pct = (100 * saved_bytes / total_original_bytes if total_original_bytes > 0 else 0.0)
    log(
        f"Total converted: {fmt_size(total_original_bytes)} -> "
        f"{fmt_size(total_output_bytes)} "
        f"({reduction_pct:.1f}% reduction), "
        f"saved {fmt_size(saved_bytes)}"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def resolve_threads(cli_value: int | None) -> int:
    """Determine encoder thread count prioritizing CLI argument, Slurm environment, then CPU affinity."""
    if cli_value is not None:
        return cli_value
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return int(env)
    return len(os.sched_getaffinity(0))

def int_in_range(lo: int, hi: int):
    """argparse type: a whole number within [lo, hi]."""
    def check(value: str) -> int:
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value!r} is not a whole number")
        if not lo <= n <= hi:
            raise argparse.ArgumentTypeError(f"{n} is out of range (allowed: {lo}-{hi})")
        return n
    return check

def preflight() -> None:
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                sys.exit(f"{tool} not found in PATH")
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                            capture_output=True, text=True).stdout
        if "libsvtav1" not in out:
            sys.exit("This ffmpeg build has no libsvtav1 encoder")

def main() -> None:
    global VERBOSE

    ap = argparse.ArgumentParser(description="Convert MPEG-2 MKV files to AV1.")
    ap.add_argument("library", type=Path, help="Root directory of the video library")
    ap.add_argument("--threads", type=int_in_range(*THREADS_RANGE), default=None,
                    metavar="1..1024", help="SVT-AV1 threads per task")
    ap.add_argument("--crf", type=int_in_range(*CRF_RANGE), default=24,
                    metavar="0..63", help="Quality (lower = better/larger)")
    ap.add_argument("--preset", type=int_in_range(*PRESET_RANGE), default=4,
                    metavar="0..13", help="SVT-AV1 preset (lower = slower/better)")
    ap.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Probe files and report what would be done; copy/encode/replace nothing",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Detailed output (commands, paths, slot waits, timings)",
    )
    args = ap.parse_args()

    preflight()

    VERBOSE = args.verbose

    if not args.library.is_dir():
        sys.exit(f"Not a directory: {args.library}")

    threads = resolve_threads(args.threads)

    log(f"Using {threads} encoder threads per task.")
    log(f"Library: {args.library.resolve()}")
    log(f"Slurm task: id {TASK_ID} (ids 0-{TOTAL_TASKS - 1}, {TOTAL_TASKS} tasks total) on {NODE} (job {JOB_ID})")
    vlog(f"crf={args.crf} preset={args.preset} dry_run={args.dry_run}")

    process_library(args.library, threads, args.crf, args.preset, args.dry_run)


if __name__ == "__main__":
    main()