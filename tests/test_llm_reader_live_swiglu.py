from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "gcl_live_reader_test",
    TOOLS_DIR / "llm-reader-gpt2-live.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the existing live LLM reader.")
LIVE_READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE_READER)

from llm_reader.adapters.registry import create_adapter_for_model_type  # noqa: E402


class LiveSwiGLUReaderTests(unittest.TestCase):
    def test_selects_existing_swiglu_adapters_by_model_type(self) -> None:
        self.assertEqual(create_adapter_for_model_type("qwen2").adapter_id, "qwen2-swiglu")
        self.assertEqual(create_adapter_for_model_type("llama").adapter_id, "llama-swiglu")
        self.assertEqual(create_adapter_for_model_type("mistral").adapter_id, "mistral-swiglu")

    def test_layer_selection_supports_all_explicit_and_legacy_default(self) -> None:
        self.assertEqual(LIVE_READER.parse_layer_selection("all", 4, 0), [0, 1, 2, 3])
        self.assertEqual(LIVE_READER.parse_layer_selection("1, 3", 4, 0), [1, 3])
        self.assertEqual(LIVE_READER.parse_layer_selection("", 4, 2), [2])

    def test_rejects_empty_duplicate_and_out_of_range_layer_selections(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "at least one"):
            LIVE_READER.parse_layer_selection(" , ", 4, 0)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            LIVE_READER.parse_layer_selection("1,1", 4, 0)
        with self.assertRaisesRegex(RuntimeError, "outside model block range"):
            LIVE_READER.parse_layer_selection("4", 4, 0)

    def test_factory_rejects_unknown_architectures_instead_of_guessing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No live LLM reader adapter"):
            create_adapter_for_model_type("unknown-transformer")


if __name__ == "__main__":
    unittest.main()
