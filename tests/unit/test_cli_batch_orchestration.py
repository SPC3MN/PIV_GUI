"""Regression tests for cli/main.py's process_pairs_planar/_stereo/
_dual_planar batch-vs-parallelism gating.

interactive_preview used to ALSO gate Tier-3 parallelism (`and not
interactive_preview` in each function's entry condition) -- but
interactive_preview is True for every non-batch (single .set file) CLI
run regardless of how many pairs that one .set actually contains, which
is a normal, common workflow (e.g. this app's own dataset-validation
work: three real single-.set batches of 1000-1500 pairs each). That
conflation meant a real, large single-.set run silently processed fully
serial with no warning, no matter how many cores were available. These
tests lock in the fix: parallelism now engages whenever n_workers > 1
regardless of interactive_preview, with the (now-independent) preview
render skipped -- loudly, via a printed note -- only in that specific
overlap case.

No real image data or ProcessPoolExecutor involved -- run_*_batch_parallel
is monkeypatched to a stub that just records how it was called, since
what's under test is the ORCHESTRATION decision (which path gets taken),
not the parallel workers' own numerics (see test_parallel_planar.py /
test_parallel_stereo.py / test_parallel_dual_planar.py for that).
"""

import piv_suite.cli.main as cli_main
import piv_suite.processing.parallel_dual_planar as parallel_dual_planar
import piv_suite.processing.parallel_planar as parallel_planar
import piv_suite.processing.parallel_stereo as parallel_stereo
from piv_suite.config.schema import ProjectConfig


def _cfg(verbose=True):
    cfg = ProjectConfig()
    cfg.project.backend = "cpu"
    cfg.correlation.use_tiling = False
    cfg.output.verbose = verbose
    return cfg


def _fake_parallel_run(recorder):
    def _run(pair_source, cfg, output_dir, *rest, on_pair_finished=None, on_pair_error=None, **kwargs):
        # n_workers is always the LAST positional arg in `rest`, for
        # every mode's batch-runner signature.
        recorder["called"] = True
        recorder["n_workers"] = rest[-1]
        return [], False
    return _run


def test_process_pairs_planar_uses_parallel_path_despite_interactive_preview(monkeypatch, capsys):
    cfg = _cfg()
    monkeypatch.setattr(cli_main, "recommended_workers", lambda override: 8)
    recorder = {"called": False}
    monkeypatch.setattr(parallel_planar, "run_planar_batch_parallel", _fake_parallel_run(recorder))

    rows = cli_main.process_pairs_planar(iter([]), cfg, "out", interactive_preview=True)

    assert recorder["called"] is True
    assert recorder["n_workers"] == 8
    assert rows == []
    assert "first-snapshot preview skipped" in capsys.readouterr().out


def test_process_pairs_planar_serial_path_when_auto_detect_gives_one_worker(monkeypatch, capsys):
    cfg = _cfg()
    monkeypatch.setattr(cli_main, "recommended_workers", lambda override: 1)
    recorder = {"called": False}
    monkeypatch.setattr(parallel_planar, "run_planar_batch_parallel", _fake_parallel_run(recorder))

    rows = cli_main.process_pairs_planar(iter([]), cfg, "out", interactive_preview=True)

    assert recorder["called"] is False  # n_workers<=1 always takes the serial loop
    assert rows == []
    # nothing to skip -- the serial loop (which ran zero iterations on the
    # empty pair_source here) is what would have rendered the preview
    assert "first-snapshot preview skipped" not in capsys.readouterr().out


def test_process_pairs_stereo_uses_parallel_path_despite_interactive_preview(monkeypatch, capsys):
    cfg = _cfg()
    monkeypatch.setattr(cli_main, "recommended_workers", lambda override: 4)
    recorder = {"called": False}
    monkeypatch.setattr(parallel_stereo, "run_stereo_batch_parallel", _fake_parallel_run(recorder))

    rows = cli_main.process_pairs_stereo(iter([]), cfg, "out", interactive_preview=True)

    assert recorder["called"] is True
    assert recorder["n_workers"] == 4
    assert rows == []
    assert "first-snapshot preview skipped" in capsys.readouterr().out


def test_process_pairs_dual_planar_uses_parallel_path_despite_interactive_preview(monkeypatch, capsys):
    cfg = _cfg()
    monkeypatch.setattr(cli_main, "recommended_workers", lambda override: 4)
    recorder = {"called": False}
    monkeypatch.setattr(parallel_dual_planar, "run_dual_planar_batch_parallel", _fake_parallel_run(recorder))

    rows = cli_main.process_pairs_dual_planar(iter([]), cfg, "out", interactive_preview=True)

    assert recorder["called"] is True
    assert recorder["n_workers"] == 4
    assert rows == []
    assert "first-snapshot preview skipped" in capsys.readouterr().out


def test_process_pairs_planar_does_not_print_skip_note_when_no_preview_wanted(monkeypatch, capsys):
    # A real batch run (many .set files, is_batch=True -> interactive_
    # preview=False) using parallelism has nothing to skip -- must not
    # print a note about a preview nobody asked for.
    cfg = _cfg()
    monkeypatch.setattr(cli_main, "recommended_workers", lambda override: 8)
    recorder = {"called": False}
    monkeypatch.setattr(parallel_planar, "run_planar_batch_parallel", _fake_parallel_run(recorder))

    cli_main.process_pairs_planar(iter([]), cfg, "out", interactive_preview=False)

    assert recorder["called"] is True
    assert "first-snapshot preview skipped" not in capsys.readouterr().out
