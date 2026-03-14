from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.stage_gs2mesh_reconstruct import (
    _build_gs_train_args,
    _normalize_runtime_profile,
)


class TestGsTrainArgs(unittest.TestCase):
    def test_invalid_runtime_profile_falls_back_to_auto(self) -> None:
        self.assertEqual(_normalize_runtime_profile("invalid"), "auto")

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_auto_profile_uses_sparse_adam_when_available(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = True

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                5000,
                "auto",
            )
        )

        self.assertEqual(resolved_profile, "auto")
        self.assertEqual(optimizer_type, "sparse_adam")
        self.assertIsNone(fallback_reason)
        self.assertIn("--disable_viewer", args)
        self.assertEqual(
            args[args.index("--test_iterations") + 1],
            "-1",
        )
        self.assertEqual(
            args[args.index("--save_iterations") + 1],
            "5001",
        )
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "sparse_adam",
        )

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_auto_profile_falls_back_to_default_when_unavailable(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = False

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                30000,
                "auto",
            )
        )

        self.assertEqual(resolved_profile, "auto")
        self.assertEqual(optimizer_type, "default")
        self.assertEqual(fallback_reason, "SparseGaussianAdam unavailable")
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "default",
        )

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_compat_profile_forces_default_optimizer(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = True

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                7000,
                "compat",
            )
        )

        self.assertEqual(resolved_profile, "compat")
        self.assertEqual(optimizer_type, "default")
        self.assertIsNone(fallback_reason)
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "default",
        )


if __name__ == "__main__":
    unittest.main()
