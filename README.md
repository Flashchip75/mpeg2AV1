# MPEG-2 to AV1 Transcoder

Finds MPEG-2 `.mkv` files in a video library and re-encodes them to AV1 (SVT-AV1, 10-bit). Audio, subtitle, and chapter streams are preserved unchanged. Built for a Slurm cluster (RHEL 9) with the library on an NFS share; files are split across Slurm tasks with node-local background prefetching.

> ## ⚠️ Not for production use
>
> **This is a hobby project and it is destructive by design.** A successful
> conversion **replaces your original file**, and no backup is kept.
>
> - While output is checked via duration matching and full decoding integrity tests, edge cases can still occur.
> - Crashes, node failures, full disks or NFS problems at the wrong moment
>   could leave you with a damaged or missing file.
> - It has only been tested on my own library and setup.
>
> **Keep backups of everything you point it at, and try it on a few copies
> first.** Use at your own risk. See the license for the full disclaimer.

## How it works

1. Recursively scan the video library for `.mkv` files (skipping test/temp markers).
2. Split the file list across Slurm tasks (`TASK_ID` / `TOTAL_TASKS`).
3. Probe video streams with `ffprobe`: skip any files that are not `mpeg2video`.
4. Stage files locally using background prefetching with node-local concurrency limits.
5. Transcode MPEG-2 to AV1 using `libsvtav1` with real-time progress reporting.
6. Verify output codec, check duration matches within limits, and run a full stream decode pass to catch corruption.
7. Atomically replace the original file with the verified AV1 file (`os.replace`).

## Requirements

- Python 3.8+ (standard library only)
- `ffmpeg` and `ffprobe` with `libsvtav1` support
- Slurm (optional; it also runs standalone)
- Local scratch space in `/tmp` for staging/temporary files

## Usage

Under Slurm (configure paths and resource flags in your `.slurm` file):

```bash
sbatch mpeg2AV1.slurm
```

## Standalone

```bash
python3 -u mpeg2AV1.py /path/to/library [--threads N] [--crf 24] [--preset 5] [--dry-run] [--verbose]
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE). In short: you may use, modify and
share this for noncommercial purposes (personal use, hobby projects,
education, research, charities and similar), provided you keep the copyright
notice. Commercial use is not permitted. Not an OSI-approved open source
license.
