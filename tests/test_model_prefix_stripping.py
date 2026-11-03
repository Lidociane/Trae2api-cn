"""Regression: the public ``trae/`` provider prefix must never reach upstream.

CLIProxyAPI registers these models with ``model-prefix=trae``, so callers send
``trae/deepseek-v4.1-flash``.  Trae's gateways match ``config_name`` exactly and
answer ``4001 the param is invalid`` for the prefixed spelling, which surfaced
as a 502 "All upstream paths failed" for text *and* tool requests.
"""

from __future__ import annotations

import unittest

from src import trae_client


class ModelPrefixTests(unittest.TestCase):
    def test_convert_model_name_strips_prefix(self):
        self.assertEqual(
            trae_client.convert_model_name("trae/deepseek-v4.1-flash"),
            trae_client.convert_model_name("deepseek-v4.1-flash"),
        )
        self.assertNotIn("trae/", trae_client.convert_model_name("trae/deepseek-v4.1-flash"))

    def test_prefix_stripping_is_case_insensitive_and_space_tolerant(self):
        self.assertEqual(
            trae_client.strip_model_prefix("  TRAE/DeepSeek-V4-Pro  "), "DeepSeek-V4-Pro"
        )
        self.assertEqual(trae_client.strip_model_prefix("deepseek-v4.1-flash"), "deepseek-v4.1-flash")
        self.assertEqual(trae_client.strip_model_prefix(""), "")

    def test_legacy_chat_body_never_carries_the_prefix(self):
        body = trae_client.build_llm_chat_body(
            [{"role": "user", "content": "hi"}], "trae/deepseek-v4.1-flash", True
        )
        for key in ("model", "config_name"):
            self.assertNotIn("trae/", str(body.get(key) or ""), key)


if __name__ == "__main__":
    unittest.main()
