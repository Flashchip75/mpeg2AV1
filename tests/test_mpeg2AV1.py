import json
import os
import shutil
import subprocess
import argparse
from subprocess import CompletedProcess
from pathlib import Path
from collections import namedtuple

import pytest

import mpeg2AV1 as m

Usage = namedtuple("Usage", "total used free")
GB = 10**9

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Keep scratch and lock files inside tmp_path; reset module state."""
    scratch_root = tmp_path / "scratch_root"
    scratch_root.mkdir()
    monkeypatch.setattr(m, "SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(m, "TASK_ID", 0)
    monkeypatch.setattr(m, "TOTAL_TASKS", 1)
    monkeypatch.setattr(m, "VERBOSE", False)
    m.active_staged.clear()
    yield
    m.active_staged.clear()


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
# Tests for disk space management and prefetch wait logic
# --------------------------------------------------------------------------- #

def fake_usage(monkeypatch, *free_values, total=100 * GB):
    """Each disk_usage() call returns the next free value (the last one repeats)."""
    seq = list(free_values)

    def _du(path):
        free = seq.pop(0) if len(seq) > 1 else seq[0]
        return Usage(total, total - free, free)

    monkeypatch.setattr(m.shutil, "disk_usage", _du)


def test_wait_for_space_passes_when_enough_free(monkeypatch, tmp_path):
    fake_usage(monkeypatch, 50 * GB)
    m.wait_for_space(tmp_path, 10 * GB, min_free_pct=15)   # 40% would remain


def test_wait_for_space_rejects_file_that_can_never_fit(monkeypatch, tmp_path):
    fake_usage(monkeypatch, 95 * GB)
    with pytest.raises(m.NotEnoughSpace):
        m.wait_for_space(tmp_path, 90 * GB, min_free_pct=15)


def test_wait_for_space_waits_then_proceeds(monkeypatch, tmp_path):
    fake_usage(monkeypatch, 20 * GB, 20 * GB, 60 * GB)     # space frees up on 3rd check
    sleeps = []
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    m.wait_for_space(tmp_path, 10 * GB, min_free_pct=15)
    assert len(sleeps) == 2


def test_wait_for_space_times_out(monkeypatch, tmp_path):
    fake_usage(monkeypatch, 20 * GB)
    monkeypatch.setattr(m, "SPACE_WAIT_TIMEOUT", 0)
    with pytest.raises(m.NotEnoughSpace):
        m.wait_for_space(tmp_path, 10 * GB, min_free_pct=15)

# --------------------------------------------------------------------------- #
# ffprobe wrappers (subprocess mocked)
# --------------------------------------------------------------------------- #

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
# prepare (staging)
# --------------------------------------------------------------------------- #

def test_prepare_skips_non_mpeg2(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "h264"})
    status, meta, local = m.prepare(0, tmp_path / "a.mkv", tmp_path)
    assert (status, local) == ("skipped", None)
    assert meta is not None
    assert meta["codec_name"] == "h264"


def test_prepare_fails_when_probe_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "probe_video", lambda p: None)
    assert m.prepare(0, tmp_path / "a.mkv", tmp_path) == ("failed", None, None)


def test_prepare_stages_mpeg2_file(monkeypatch, tmp_path):
    src = tmp_path / "a.mkv"
    src.write_bytes(b"video-data")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "mpeg2video"})

    status, _meta, local = m.prepare(3, src, scratch)

    assert status == "ready"
    assert local is not None
    assert local == scratch / "3_a.mkv"
    assert local.read_bytes() == b"video-data"
    assert not list(scratch.glob("*.part"))
    assert local in m.active_staged


def test_prepare_cleans_up_when_copy_fails(monkeypatch, tmp_path):
    src = tmp_path / "a.mkv"
    src.write_bytes(b"x")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(m, "probe_video", lambda p: {"codec_name": "mpeg2video"})

    def boom(*a, **k):
        raise OSError("network gone")

    monkeypatch.setattr(m.shutil, "copy2", boom)

    status, _meta, local = m.prepare(0, src, scratch)

    assert (status, local) == ("failed", None)
    assert not list(scratch.iterdir())
    assert not m.active_staged


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


def test_convert_success_replaces_original(monkeypatch, staged):
    original, local = staged
    monkeypatch.setattr(m, "get_video_duration", lambda p: 100.0)
    monkeypatch.setattr(m, "run_ffmpeg_with_progress", fake_encoder())
    monkeypatch.setattr(m, "verify_output", lambda out, dur: True)

    assert m.convert_to_av1(original, local, 4, 24, 5)

    assert original.read_bytes() == b"AV1"
    assert not local.exists()
    assert not m.tmp_output_path(original).exists()
    assert m.active_tmp_output is None


@pytest.mark.parametrize(
    "duration, rc, verified",
    [(0.0, 0, True), (100.0, 1, True), (100.0, 0, False)],
    ids=["bad-duration", "ffmpeg-error", "verification-fails"],
)
def test_convert_failure_keeps_original(monkeypatch, staged, duration, rc, verified):
    original, local = staged
    monkeypatch.setattr(m, "get_video_duration", lambda p: duration)
    monkeypatch.setattr(m, "run_ffmpeg_with_progress", fake_encoder(rc))
    monkeypatch.setattr(m, "verify_output", lambda out, dur: verified)

    assert not m.convert_to_av1(original, local, 4, 24, 5)

    assert original.read_bytes() == b"MPEG2" * 100   # untouched
    assert not m.tmp_output_path(original).exists()  # temp output removed
    assert not local.exists()                        # staged copy removed


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
    assert not m.scratch_dir().exists()


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


def test_process_library_counts_and_cleans_scratch(monkeypatch, tmp_path, capsys):
    lib = make_library(tmp_path, ["a.mkv", "b.mkv", "c.mkv"])

    def fake_prepare(idx, file_path, scratch):
        if file_path.name == "a.mkv":
            local = scratch / f"{idx}_{file_path.name}"
            local.write_bytes(b"x")
            return "ready", {"codec_name": "mpeg2video"}, local
        if file_path.name == "b.mkv":
            return "skipped", {"codec_name": "h264"}, None
        return "failed", None, None

    converted = []

    def fake_convert(file_path, local, threads, crf, preset):
        converted.append(file_path.name)
        return True

    monkeypatch.setattr(m, "prepare", fake_prepare)
    monkeypatch.setattr(m, "convert_to_av1", fake_convert)

    m.process_library(lib, 4, 24, 5, dry_run=False)

    assert converted == ["a.mkv"]
    assert "converted=1 skipped=1 failed=1" in capsys.readouterr().out
    assert not m.scratch_dir().exists()


# --------------------------------------------------------------------------- #
# Integration test with real ffmpeg (auto-skipped if unavailable)
# --------------------------------------------------------------------------- #

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