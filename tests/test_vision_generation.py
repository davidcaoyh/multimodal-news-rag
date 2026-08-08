import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src import generate as generation


def hit(path):
    return SimpleNamespace(
        article_id="article-1",
        headline="A headline",
        caption="A visible event",
        image_path=str(path),
        split="pool",
        score=1.0,
        s_text=0.8,
        s_img=0.7,
        passages=["A supported passage."],
    )


class VisionGenerationTests(unittest.TestCase):
    def test_text_arm_request_remains_a_string(self):
        content, evidence, paths = generation.build_request_content(
            "topic", [hit("unused.jpg")], "M"
        )
        self.assertIsInstance(content, str)
        self.assertIn('[IMAGE 1: "A visible event"]', evidence)
        self.assertEqual(paths, [])

    def test_vision_arm_adds_real_image_block_after_same_text_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (12, 12), "red").save(path)
            h = hit(path)
            m_prompt, m_evidence, _ = generation.build_request_content("topic", [h], "M")
            content, evidence, paths = generation.build_request_content(
                "topic", [h], generation.VISION
            )
            self.assertEqual(content[0]["text"], m_prompt)
            self.assertEqual(evidence, m_evidence)
            self.assertEqual(paths, [str(path)])
            self.assertEqual(content[2]["type"], "image_url")
            self.assertEqual(content[2]["image_url"]["detail"], "low")
            self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_cache_safe_content_does_not_persist_base64(self):
        content = [{"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64,abc", "detail": "low"
        }}]
        safe = generation._cache_safe_content(content)
        self.assertTrue(safe[0]["image_url"]["url"].startswith("sha256:"))
        self.assertNotIn("base64", json.dumps(safe))

    def test_summarize_records_vision_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (12, 12), "blue").save(path)
            h = hit(path)
            with patch.object(generation, "retrieve", return_value=[h]), \
                 patch.object(generation, "_get_test_ids", return_value=set()), \
                 patch.object(generation, "generate", return_value="Grounded summary.") as call:
                row = generation.summarize("topic", config=generation.VISION)
            self.assertEqual(row["summary"], "Grounded summary.")
            self.assertEqual(row["image_paths"], str(path))
            self.assertIsInstance(call.call_args.args[0], list)


if __name__ == "__main__":
    unittest.main()
