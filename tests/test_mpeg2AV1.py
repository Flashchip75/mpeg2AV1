import json
import os
import sys
import shutil
import subprocess
import argparse
from subprocess import CompletedProcess
from pathlib import Path

import pytest

import mpeg2AV1 as m

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def original_video(tmp_path):
    original = tmp_path / "ep.mkv"
    original.write_bytes(b"MPEG2" * 100)
    return original


def make_library(tmp_path, names):
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in names:
        (lib / name).write_bytes(b"x" * 10)
    return lib


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


# --------------------------------------------------------------------------- #
# Formatting and small helpers
# --------------------------------------------------------------------------- #

def test_fmt_secs():
    assert m.fmt_secs(3661) == "1:01:01"
    assert m.fmt_secs(0) == "0:00:00"
    assert m.fmt_secs(-5) == "0:00:00"


def test_fmt_size():
    assert m.fmt_size(512) == "512 B"
    assert m.fmt_size(1500) == "1.5 KB"
    assert m.fmt_size(2_000_000_000) == "2.0 GB"


def test_resolve_threads_priority(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "10")
    assert m.resolve_threads(4) == 4      # CLI wins
    assert m.resolve_threads(None) == 10  # then Slurm
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert m.resolve_threads(None) == len(os.sched_getaffinity(0))


def test_build_ffmpeg_cmd():
    cmd = m.build_ffmpeg_cmd(Path("in.mkv"), Path("out.mkv"), 8, 24, 5)
    assert cmd[cmd.index("-i") + 1] == "in.mkv"
    assert cmd[-1] == "out.mkv"
    assert "libsvtav1" in cmd
    assert cmd[cmd.index("-crf") + 1] == "24"
    assert cmd[cmd.index("-preset") + 1] == "5"
    assert cmd[cmd.index("-svtav1-params") + 1] == "lp=8"


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def test_find_mkv_files_filters_and_recurses(tmp_path):
    lib = tmp_path / "lib"
    (lib / "season1").mkdir(parents=True)
    (lib / ".cache").mkdir()   
    keep = [lib / "a.mkv", lib / "season1" / "b.MKV"]
    skip = [
        lib / "c_temp.mkv",
        lib / "d_debug.mkv",
        lib / "e_av1_test.mkv",
        lib / ".x_av1_1_t0.mkv",
        lib / "season1" / ".y_av1_2_t1.mkv",
        lib / ".cache" / "normal_name.mkv",
        lib / "f.mp4",
        lib / "season1" / "g.txt",
    ]
    for p in keep + skip:
        p.write_bytes(b"x")

    assert m.find_mkv_files(lib) == sorted(keep)


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value, ok", [
    ("0", True), ("24", True), ("63", True),
    ("-1", False), ("64", False), ("24.5", False), ("abc", False),
])
def test_crf_range(value, ok):
    check = m.int_in_range(*m.CRF_RANGE)
    if ok:
        assert check(value) == int(value)
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            check(value)

@pytest.mark.parametrize("value, ok", [
    ("0", True), ("4", True), ("13", True),
    ("-2", False), ("-1", False), ("14", False), ("4.5", False), ("fast", False),
])
def test_preset_range(value, ok):
    check = m.int_in_range(*m.PRESET_RANGE)
    if ok:
        assert check(value) == int(value)
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            check(value)

@pytest.mark.parametrize("value, ok", [
    ("1", True), ("10", True), ("1024", True),
    ("0", False), ("-1", False), ("1025", False), ("2.5", False), ("many", False),
])
def test_threads_range(value, ok):
    check = m.int_in_range(*m.THREADS_RANGE)
    if ok:
        assert check(value) == int(value)
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            check(value)


@pytest.mark.parametrize("bad", ["0", "-4", "1025", "9223372036854775807", "2.5", "many"])
def test_main_rejects_bad_threads_before_any_work(monkeypatch, tmp_path, capsys, bad):
    """--threads must be validated at parse time, before any file is touched."""
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path), "--threads", bad])
    monkeypatch.setattr(
        m, "process_library", lambda *a, **k: pytest.fail("work started despite bad --threads")
    )

    with pytest.raises(SystemExit) as exc:
        m.main()

    assert exc.value.code == 2                       # argparse usage error
    assert "--threads" in capsys.readouterr().err    # message names the bad flag


@pytest.mark.parametrize("value", [
    "9223372036854775807",     # int64 max
    "18446744073709551616",    # 2**64
    "0x7FFFFFFFFFFFFFFF",      # hex form
    "9" * 5000,                # beyond Python's digit limit
])
def test_int_in_range_rejects_huge_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        m.int_in_range(0, 63)(value)

# --------------------------------------------------------------------------- #
# ffprobe wrappers (subprocess mocked)
# --------------------------------------------------------------------------- #

def test_probe_video_returns_first_stream(monkeypatch):
    out = json.dumps({"streams": [{"codec_name": "mpeg2video", "width": 720}]})
    monkeypatch.setattr(m.subprocess, "run", 
                        lambda *a, **k: CompletedProcess(args=a[0], returncode=0, stdout=out))
    meta = m.probe_video(Path("x.mkv"))
    assert meta is not None
    assert meta["codec_name"] == "mpeg2video"


def test_probe_video_no_streams(monkeypatch):
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: completed(json.dumps({"streams": []}))
    )
    assert m.probe_video(Path("x.mkv")) is None


def test_probe_video_ffprobe_failure(monkeypatch):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(m.subprocess, "run", boom)
    assert m.probe_video(Path("x.mkv")) is None


def test_get_video_duration(monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: completed("123.45\n"))
    assert m.get_video_duration(Path("x.mkv")) == pytest.approx(123.45)

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: completed("garbage"))
    assert m.get_video_duration(Path("x.mkv")) == 0.0

# --------------------------------------------------------------------------- #
# verify_output
# --------------------------------------------------------------------------- #

def test_verify_rejects_missing_or_empty(tmp_path):
    out = tmp_path / "o.mkv"
    assert not m.verify_output(out, 100.0)
    out.write_bytes(b"")
    assert not m.verify_output(out, 100.0)


def test_verify_rejects_wrong_codec(monkeypatch, tmp_path):
    out = tmp_path / "o.mkv"
    out.write_bytes(b"x")
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "h264"})
    assert not m.verify_output(out, 100.0)


def test_verify_rejects_duration_mismatch(monkeypatch, tmp_path):
    out = tmp_path / "o.mkv"
    out.write_bytes(b"x")
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "av1"})
    monkeypatch.setattr(m, "get_video_duration", lambda p: 110.0)
    assert not m.verify_output(out, 100.0)


def test_verify_rejects_decode_errors(monkeypatch, tmp_path):
    out = tmp_path / "o.mkv"
    out.write_bytes(b"x")
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "av1"})
    monkeypatch.setattr(m, "get_video_duration", lambda p: 100.0)
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: completed(returncode=1, stderr="corrupt")
    )
    assert not m.verify_output(out, 100.0)


def test_verify_accepts_good_file(monkeypatch, tmp_path):
    out = tmp_path / "o.mkv"
    out.write_bytes(b"x")
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "av1"})
    monkeypatch.setattr(m, "get_video_duration", lambda p: 100.5)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: completed())
    assert m.verify_output(out, 100.0)


# --------------------------------------------------------------------------- #
# convert_to_av1 (encode + verify + replace)
# --------------------------------------------------------------------------- #

@pytest.fixture
def staged(tmp_path):
    original = tmp_path / "ep.mkv"
    original.write_bytes(b"MPEG2" * 100)
    local = tmp_path / "scratch" / "0_ep.mkv"
    local.parent.mkdir()
    local.write_bytes(b"copy")
    return original, local


def fake_encoder(rc=0):
    """Stand-in for run_ffmpeg_with_progress: writes the output file."""

    def _run(cmd, total_dur):
        if rc == 0:
            Path(cmd[-1]).write_bytes(b"AV1")
        return rc

    return _run


def test_convert_success_replaces_original(monkeypatch, original_video):
    original = original_video

    monkeypatch.setattr(m, "get_video_duration", lambda p: 100.0)
    monkeypatch.setattr(m, "run_ffmpeg_with_progress", fake_encoder())
    monkeypatch.setattr(m, "verify_output", lambda out, dur: True)

    success, _, _ = m.convert_to_av1(original, 4, 24, 5)

    assert success
    assert original.read_bytes() == b"AV1"
    assert not m.tmp_output_path(original).exists()
    assert m.active_tmp_output is None


@pytest.mark.parametrize(
    "duration, rc, verified",
    [(0.0, 0, True), (100.0, 1, True), (100.0, 0, False)],
    ids=["bad-duration", "ffmpeg-error", "verification-fails"],
)
def test_convert_failure_keeps_original(
    monkeypatch, original_video, duration, rc, verified
):
    original = original_video

    monkeypatch.setattr(m, "get_video_duration", lambda p: duration)
    monkeypatch.setattr(m, "run_ffmpeg_with_progress", fake_encoder(rc))
    monkeypatch.setattr(m, "verify_output", lambda out, dur: verified)

    success, _, _ = m.convert_to_av1(original, 4, 24, 5)

    assert not success
    assert original.read_bytes() == b"MPEG2" * 100
    assert not m.tmp_output_path(original).exists()
    assert m.active_tmp_output is None


# --------------------------------------------------------------------------- #
# process_library
# --------------------------------------------------------------------------- #

def test_dry_run_touches_nothing(monkeypatch, tmp_path, capsys):
    lib = make_library(tmp_path, ["a.mkv", "b.mkv"])
    codecs = {"a.mkv": "mpeg2video", "b.mkv": "h264"}
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": codecs[p.name]})
    before = {p.name: p.read_bytes() for p in lib.iterdir()}

    m.process_library(lib, 4, 24, 5, dry_run=True)

    out = capsys.readouterr().out
    assert "Would convert" in out
    assert "Would skip" in out
    assert "would_convert=1 skipped=1 failed=0" in out
    assert {p.name: p.read_bytes() for p in lib.iterdir()} == before


@pytest.mark.parametrize(
    "task_id, expected, excluded",
    [(0, ["f0.mkv", "f2.mkv", "f4.mkv"], ["f1.mkv", "f3.mkv"]),
     (1, ["f1.mkv", "f3.mkv"], ["f0.mkv", "f2.mkv", "f4.mkv"])],
)
def test_tasks_split_files_round_robin(
    monkeypatch, tmp_path, capsys, task_id, expected, excluded
):
    lib = make_library(tmp_path, [f"f{i}.mkv" for i in range(5)])
    monkeypatch.setattr(m, "TASK_ID", task_id)
    monkeypatch.setattr(m, "TOTAL_TASKS", 2)
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "mpeg2video"})

    m.process_library(lib, 4, 24, 5, dry_run=True)

    out = capsys.readouterr().out
    assert f"Assigned {len(expected)} of 5" in out
    assert all(name in out for name in expected)
    assert not any(name in out for name in excluded)


# --------------------------------------------------------------------------- #
# Integration test with real ffmpeg (auto-skipped if unavailable)
# --------------------------------------------------------------------------- #

def test_process_library_circuit_breaker(monkeypatch, tmp_path):
    # Setup dummy video files
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()

    for i in range(5):
        (lib_dir / f"video{i}.mkv").write_bytes(b"MPEG2")

    # Mock file discovery
    monkeypatch.setattr(
        m,
        "find_mkv_files",
        lambda d: sorted(lib_dir.glob("*.mkv"))
    )

    # Pretend every file contains MPEG-2
    monkeypatch.setattr(
        m,
        "probe_video",
        lambda p: {"codec_name": "mpeg2video"}
    )

    # Mock conversion to always fail
    call_count = 0

    def counting_convert(file_path, threads, crf, preset):
        nonlocal call_count
        call_count += 1
        return False, 0, 0

    monkeypatch.setattr(m, "convert_to_av1", counting_convert)

    monkeypatch.setattr(m, "TASK_ID", 0)
    monkeypatch.setattr(m, "TOTAL_TASKS", 1)

    m.process_library(
        lib_dir,
        threads=4,
        crf=24,
        preset=5,
        dry_run=False
    )

    assert call_count == m.MAX_CONSECUTIVE_FAILURES

def _ffmpeg_has_encoders(*names):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return all(n in out for n in names)


@pytest.mark.integration
@pytest.mark.skipif(
    not _ffmpeg_has_encoders("libsvtav1", "mpeg2video"),
    reason="needs ffmpeg/ffprobe with libsvtav1 and mpeg2video",
)
def test_end_to_end_conversion(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    src = lib / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=duration=2",
            "-pix_fmt", "yuv420p", "-c:v", "mpeg2video", "-c:a", "aac",
            str(src),
        ],
        check=True,
    )
    meta = m.probe_video(src)
    assert meta is not None, "ffprobe failed to read the generated file. Check if ffmpeg/ffprobe is installed and supports mpeg2video."
    assert meta["codec_name"] == "mpeg2video"

    m.process_library(lib, threads=2, crf=40, preset=12, dry_run=False)

    final_meta = m.probe_video(src)
    assert final_meta is not None, "ffprobe failed to read the converted AV1 file."
    assert final_meta["codec_name"] == "av1"
    assert [p.name for p in lib.iterdir()] == ["clip.mkv"]  # no temp leftovers