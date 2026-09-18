import pytest

from sft_script import exclusive_run_lock


def test_same_run_cannot_be_locked_twice(tmp_path):
    with exclusive_run_lock(str(tmp_path), "one-run"):
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_run_lock(str(tmp_path), "one-run"):
                pass


def test_different_runs_have_independent_locks(tmp_path):
    with exclusive_run_lock(str(tmp_path), "run-a"):
        with exclusive_run_lock(str(tmp_path), "run-b"):
            pass


def test_lock_is_released_after_context_exit(tmp_path):
    with exclusive_run_lock(str(tmp_path), "one-run"):
        pass
    with exclusive_run_lock(str(tmp_path), "one-run"):
        pass
