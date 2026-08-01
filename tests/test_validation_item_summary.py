from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The sweep imports its adapters as `llm_reader.*`, so tools/ must be importable too.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

# The sweep module has a hyphenated filename, so it is loaded by path rather than imported.
_SPEC = importlib.util.spec_from_file_location(
    "gpt2_factual_sweep",
    REPO_ROOT / "tools" / "llm-reader-gpt2-factual-sweep.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_validation_items = _MODULE.summarize_validation_items


class ValidationItemSummaryTest(unittest.TestCase):
    """A control that never ran must not raise the pass rate it is summarised into.

    When a negative fact's answer is not a single GPT-2 token, the item is recorded with
    status "skipped", pass True and targetLogitDrop 0.0 so the attempt leaves a trace. The
    summary used to count every item in both count and passed, so those placeholders lifted
    negativeFactSpecificityPassRate and their 0.0 drops entered the median, biasing a
    specificity statistic in the flattering direction.
    """

    @staticmethod
    def _ran(passed: bool, drop: float) -> dict:
        return {"status": "ok", "pass": passed, "targetLogitDrop": drop, "ratioToPrimaryGraphDrop": drop}

    @staticmethod
    def _skipped() -> dict:
        return {
            "status": "skipped",
            "skipReason": "negative fact answer is not one GPT-2 token with a leading space",
            "pass": True,
            "targetLogitDrop": 0.0,
            "ratioToPrimaryGraphDrop": 0.0,
        }

    def test_skipped_items_do_not_inflate_the_pass_rate(self) -> None:
        observed = summarize_validation_items([
            self._ran(True, 2.0),
            self._ran(False, 0.1),
            self._skipped(),
            self._skipped(),
        ])

        # Two ran, one passed. Counting the skipped pair would have reported 3 of 4.
        self.assertEqual(observed["count"], 2)
        self.assertEqual(observed["passed"], 1)
        self.assertEqual(observed["passRate"], 0.5)
        self.assertEqual(observed["skippedCount"], 2)
        self.assertEqual(observed["itemCount"], 4)

    def test_skipped_placeholders_do_not_enter_the_median(self) -> None:
        with_skips = summarize_validation_items([
            self._ran(True, 2.0),
            self._ran(True, 3.0),
            self._skipped(),
            self._skipped(),
            self._skipped(),
        ])
        without_skips = summarize_validation_items([self._ran(True, 2.0), self._ran(True, 3.0)])

        self.assertEqual(with_skips["medianTargetLogitDrop"], without_skips["medianTargetLogitDrop"])
        self.assertGreater(with_skips["medianTargetLogitDrop"], 0.0)

    def test_an_all_skipped_set_reports_nothing_rather_than_a_perfect_score(self) -> None:
        observed = summarize_validation_items([self._skipped(), self._skipped()])

        self.assertEqual(observed["count"], 0)
        self.assertEqual(observed["passed"], 0)
        self.assertEqual(observed["passRate"], 0.0)
        self.assertEqual(observed["skippedCount"], 2)

    def test_a_set_with_no_skips_is_unchanged(self) -> None:
        observed = summarize_validation_items([self._ran(True, 1.0), self._ran(True, 5.0)])

        self.assertEqual(observed["count"], 2)
        self.assertEqual(observed["passRate"], 1.0)
        self.assertEqual(observed["skippedCount"], 0)
        self.assertEqual(observed["maxTargetLogitDrop"], 5.0)

    def test_an_empty_set_is_still_handled(self) -> None:
        observed = summarize_validation_items([])

        self.assertEqual(observed["count"], 0)
        self.assertEqual(observed["passRate"], 0.0)


if __name__ == "__main__":
    unittest.main()
