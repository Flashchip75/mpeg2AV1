#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import hashlib
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TASK_ID = int(os.environ.get("SLURM_PROCID", 0))
TOTAL_TASKS = int(os.environ.get("SLURM_NTASKS", 1))
JOB_ID = os.environ.get("SLURM_JOB_ID", "manual")
NODE = os.environ.get("SLURMD_NODENAME", "local")

DOWNLOAD_SLOTS_PER_NODE = 1
PROGRESS_INTERVAL = 60       # seconds between progress lines
SCRATCH_ROOT = Path("/tmp")
SKIP_MARKERS = ("_temp", "_debug", "_av1_test")

VERBOSE = False

MIN_FREE_PCT = 10.0             # keep at least this % of the scratch device free
SPACE_POLL_SECONDS = 30         # how often to re-check while waiting
SPACE_WAIT_TIMEOUT = 4 * 3600   # give up on a file after waiting this long

THREADS_RANGE = (1, 1024)
CRF_RANGE = (0, 63)
PRESET_RANGE = (0, 13)

MAX_CONSECUTIVE_FAILURES = 3

# State needed by the signal handler
active_ffmpeg_proc: subprocess.Popen | None = None
active_tmp_output: Path | None = None
active_staged: set[Path] = set()


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

def scratch_dir() -> Path:
    """Return the job- and task-specific scratch directory path."""
    return SCRATCH_ROOT / f"mpeg2_av1_{JOB_ID}_t{TASK_ID}"


def tmp_output_path(file_path: Path) -> Path:
    """Generate a temporary output file path on the same filesystem for atomic replacement."""
    return file_path.with_name(f".{file_path.stem}_av1_{JOB_ID}_t{TASK_ID}.mkv")


def staged_path(scratch: Path, idx: int, file_path: Path) -> Path:
    """Return the local scratch staging path for a video file."""
    return scratch / f"{idx}_{file_path.name}"


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #

def cleanup_active() -> None:
    """Remove any active temporary files and staged scratch files."""
    if active_tmp_output is not None:
        active_tmp_output.unlink(missing_ok=True)
    for p in list(active_staged):
        p.unlink(missing_ok=True)


def graceful_exit(signum, frame):
    """Handle termination signals, safely stop FFmpeg, clean up scratch files, and exit."""
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
    shutil.rmtree(scratch_dir(), ignore_errors=True)

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

def sha256_file(path: Path, chunk_size: Optional[int] = None) -> str:
    """Calculate SHA-256 without loading the entire file into memory.
    
    If chunk_size is not specified, dynamically determines the optimal size 
    based on the filesystem's preferred I/O block size (st_blksize).
    """
    digest = hashlib.sha256()

    if chunk_size is None:
        try:
            # st_blksize gives the filesystem's optimal block size (e.g., 4KB, 128KB, etc.)
            # We scale it up or enforce a minimum floor (1 MB) to keep Python loop 
            # overhead low while matching hardware I/O boundaries.
            stat = path.stat()
            io_block = getattr(stat, "st_blksize", 1048576)
            chunk_size = max(io_block, 1048576)  # At least 1 MB
        except (AttributeError, OSError):
            chunk_size = 1048576  # Fallback to 1 MB if stat fails

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()

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
    if difference > 2.0:
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


def convert_to_av1(file_path: Path, local_input: Path,
                   threads: int, crf: int, preset: int, expected_sha256: str) -> tuple[bool, int, int]:
    """Encode an already-staged file and atomically replace the original."""
    global active_tmp_output

    tmp_output = tmp_output_path(file_path)
    active_tmp_output = tmp_output

    try:
        total_dur = get_video_duration(local_input)
        if total_dur <= 0:
            log("  [Error] Invalid input duration.")
            return False, 0 ,0

        cmd = build_ffmpeg_cmd(local_input, tmp_output, threads, crf, preset)
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

        if sha256_file(file_path) != expected_sha256:
            log("  [Error] Original changed since staging. Original kept.")
            return False, 0, 0

        original_size = file_path.stat().st_size
        output_size = tmp_output.stat().st_size

        # Atomic replacement: the original is only replaced after verification
        os.replace(tmp_output, file_path)

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
        local_input.unlink(missing_ok=True)
        active_staged.discard(local_input)
        active_tmp_output = None


# --------------------------------------------------------------------------- #
# Staging / prefetching
# --------------------------------------------------------------------------- #


class NotEnoughSpace(Exception):
    pass


def wait_for_space(path: Path, incoming_bytes: int, min_free_pct: float | None = None) -> None:
    """Block until copying `incoming_bytes` still leaves >= min_free_pct free.

    Raises NotEnoughSpace if the file can never fit or the wait times out.
    """
    min_free_pct = MIN_FREE_PCT if min_free_pct is None else min_free_pct
    deadline = time.time() + SPACE_WAIT_TIMEOUT
    waited = False

    while True:
        usage = shutil.disk_usage(path)
        free_after = 100 * (usage.free - incoming_bytes) / usage.total

        if free_after >= min_free_pct:
            vlog(f"[prefetch] space ok: {free_after:.1f}% would remain free")
            return

        # Even an empty device couldn't hold it: waiting won't help
        if 100 * (usage.total - incoming_bytes) / usage.total < min_free_pct:
            raise NotEnoughSpace(
                f"{fmt_size(incoming_bytes)} can never fit on a {fmt_size(usage.total)} "
                f"device while keeping {min_free_pct:.0f}% free"
            )

        if time.time() >= deadline:
            raise NotEnoughSpace(
                f"waited {fmt_secs(SPACE_WAIT_TIMEOUT)} for scratch space "
                f"({free_after:.1f}% would remain, need {min_free_pct:.0f}%)"
            )

        if not waited:
            log(
                f"  [prefetch] waiting for scratch space: staging {fmt_size(incoming_bytes)} "
                f"would leave {free_after:.1f}% free (need {min_free_pct:.0f}%)"
            )
            waited = True
        time.sleep(SPACE_POLL_SECONDS)


@contextmanager
def download_slot():
    """Node-local limit on concurrent downloads (flock is released if we die)."""
    waited = False
    while True:
        for k in range(DOWNLOAD_SLOTS_PER_NODE):
            f = open(SCRATCH_ROOT / f"dlslot_{JOB_ID}_{NODE}_{k}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            vlog(f"[prefetch] acquired download slot {k}")
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
                vlog(f"[prefetch] released download slot {k}")
            return
        if not waited:
            vlog("[prefetch] all download slots busy, waiting...")
            waited = True
        time.sleep(1)


def prepare(idx: int, file_path: Path, scratch: Path) -> tuple[str, dict | None, Path | None]:
    """Probe a file and stage it locally. Runs in the background thread.

    Returns (status, metadata, local_path) with status in
    {"ready", "skipped", "failed"}.
    """
    meta = probe_video(file_path)
    if meta is None:
        return "failed", None, None
    if meta.get("codec_name") != "mpeg2video":
        return "skipped", meta, None

    local = staged_path(scratch, idx, file_path)
    part = local.with_name(local.name + ".part")
    active_staged.update((part, local))
    try:
        size = file_path.stat().st_size
        with download_slot():
            wait_for_space(scratch, size)
            t0 = time.time()
            shutil.copy2(file_path, part)
            os.replace(part, local)
            meta["_staged_sha256"] = sha256_file(local)
            elapsed = time.time() - t0
            size = local.stat().st_size
            rate = size / elapsed / 1e6 if elapsed > 0 else 0
            log(
                f"  [prefetch] staged {file_path.name} "
                f"({fmt_size(size)}) in {fmt_secs(elapsed)} ({rate:.0f} MB/s)"
            )
        active_staged.discard(part)
        return "ready", meta, local
    except Exception as e:
        log(f"  [prefetch] staging failed for {file_path.name}: {e}")
        part.unlink(missing_ok=True)
        local.unlink(missing_ok=True)
        active_staged.difference_update((part, local))
        return "failed", meta, None


# --------------------------------------------------------------------------- #
# Library processing
# --------------------------------------------------------------------------- #

def find_mkv_files(library_dir: Path) -> list[Path]:
    """Recursively search for valid .mkv video files, excluding skip-marked entries."""
    files = []
    for root, dirs, names in os.walk(library_dir, followlinks=True,):
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
    scratch = scratch_dir()

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
        largest = max(largest, size)
        would_convert += 1

        local = staged_path(scratch, i - 1, file_path)
        tmp_out = tmp_output_path(file_path)
        log(f"  Would convert ({fmt_size(size)}).")
        vlog(f"would stage to:   {local}")
        vlog(f"would encode to:  {tmp_out}")
        vlog(f"would replace:    {file_path}")
        vlog(
            "ffmpeg: "
            + " ".join(build_ffmpeg_cmd(local, tmp_out, threads, crf, preset))
        )

    log("=" * 60)
    log(
        f"Dry run done. would_convert={would_convert} "
        f"skipped={skipped} failed={failed}"
    )
    log(f"Total data to download: {fmt_size(total_bytes)}")

    # Scratch holds at most two staged files at once per task
    free = shutil.disk_usage(SCRATCH_ROOT).free
    log(
        f"Scratch free in {SCRATCH_ROOT}: {fmt_size(free)} "
        f"(peak need for this task ~{fmt_size(2 * largest)})"
    )
    if free < 2 * largest:
        log("  [Warning] Scratch space may be too small for two staged files.")


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

    scratch = scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)
    vlog(f"scratch dir: {scratch}")
    vlog(f"scratch free: {fmt_size(shutil.disk_usage(scratch).free)}")

    converted = skipped = failed = 0
    consecutive_failures = 0
    total_original_bytes = 0
    total_output_bytes = 0

    t_start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(prepare, 0, my_files[0], scratch)

            for i, file_path in enumerate(my_files):
                status, meta, local = fut.result()

                # Start staging the next file BEFORE encoding this one
                if i + 1 < len(my_files):
                    vlog(f"queueing prefetch of {my_files[i + 1].name}")
                    fut = pool.submit(prepare, i + 1, my_files[i + 1], scratch)

                log_file_header(i + 1, len(my_files), file_path, library_dir)

                if meta:
                    log_metadata(meta)

                if status == "skipped":
                    codec = meta.get("codec_name", "unknown") if meta else "unknown"
                    log(f"  Skipping: codec is {codec}.")
                    skipped += 1
                    consecutive_failures = 0
                elif status == "failed" or local is None or meta is None:
                    log("  [Error] Could not prepare file.")
                    failed += 1
                    consecutive_failures += 1
                else:
                    expected_sha256 = meta["_staged_sha256"]
                    if "_staged_sha256" not in meta:
                        log("  [Error] Missing staged SHA-256 hash.")
                        failed += 1
                        consecutive_failures += 1
                        continue
                    success, orig_sz, out_sz = convert_to_av1(file_path, local, threads, crf, preset, expected_sha256)
                    if success:
                        converted += 1
                        consecutive_failures = 0
                        total_original_bytes += orig_sz
                        total_output_bytes += out_sz
                    else:
                        failed += 1
                        consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log(f"[Error] {consecutive_failures} failures in a row; aborting this task "
                        f"(probably a systematic problem).")
                    break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    elapsed = time.time() - t_start

    log("=" * 60)
    log(f"Done. converted={converted} skipped={skipped} failed={failed}")
    log(f"Total time elapsed: {fmt_secs(elapsed)}")
    
    saved_bytes = total_original_bytes - total_output_bytes
    overall_pct = (100 * total_output_bytes / total_original_bytes) if total_original_bytes > 0 else 0.0
    log(
        f"Total library reduction: {fmt_size(total_original_bytes)} -> {fmt_size(total_output_bytes)} "
        f"({overall_pct:.1f}%), saved {fmt_size(saved_bytes)}"
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
    vlog(
        f"crf={args.crf} preset={args.preset} "
        f"download_slots_per_node={DOWNLOAD_SLOTS_PER_NODE} dry_run={args.dry_run}"
    )

    process_library(args.library, threads, args.crf, args.preset, args.dry_run)


if __name__ == "__main__":
    main()