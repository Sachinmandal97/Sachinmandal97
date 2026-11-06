"""
Standard library automated test discovery and execution script for submission.
"""

import inspect
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure cached models are used directly in test runner without network timeout retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Add submission root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.test_submission_pipeline as t_sub
import tests.test_nli_diagnostic as t_diag
import tests.test_universal_verification as t_univ
import tests.test_evidence_hotfix as t_hotfix
import tests.test_universal_fact_checking as t_factcheck
import tests.test_phase_p0_fixes as t_p0
import tests.test_benchmark_infrastructure as t_bench
import tests.test_audit_regressions as t_audit
import tests.test_news_intelligence as t_news
import tests.test_audit_fixes as t_fixes
import tests.test_rhetoric_integration as t_rhetoric


def run_all():
    print("=" * 70)
    print("RUNNING SUBMISSION PIPELINE INTEGRITY & EXECUTION TESTS")
    print("=" * 70)

    t0 = time.perf_counter()
    modules = [t_sub, t_diag, t_univ, t_hotfix, t_factcheck, t_p0, t_bench, t_audit, t_news, t_fixes, t_rhetoric]
    functions = []
    for mod in modules:
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_"):
                functions.append((name, func))


    passed = 0
    failed = 0
    shared_pipeline = None

    for name, func in functions:
        sig = inspect.signature(func)
        kwargs = {}
        if "tmp_path" in sig.parameters:
            temp_dir = tempfile.mkdtemp()
            kwargs["tmp_path"] = Path(temp_dir)
        if "pipeline" in sig.parameters:
            if shared_pipeline is None:
                from src.pipeline import MultimodalVerificationPipeline
                shared_pipeline = MultimodalVerificationPipeline()
            kwargs["pipeline"] = shared_pipeline

        try:
            func(**kwargs)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    elapsed = time.perf_counter() - t0
    print("=" * 70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED in {elapsed:.3f}s")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    if not success:
        sys.exit(1)
