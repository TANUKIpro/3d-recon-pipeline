"""Git branch detection and slug utilities for branch-based object isolation."""

from __future__ import annotations

import os
import re
import subprocess

BRANCH_DIR_PREFIX = "@"


def detect_branch() -> str:
    """Return the current git branch name.

    Priority:
    1. ``GIT_BRANCH`` environment variable (set in docker-compose.yml)
    2. ``git rev-parse --abbrev-ref HEAD`` (local dev with .git present)
    3. Fallback to ``"main"``
    """
    env_branch = os.environ.get("GIT_BRANCH", "").strip()
    if env_branch:
        return env_branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "main"


def branch_slug(branch: str) -> str:
    """Sanitize a branch name into a filesystem-safe slug.

    ``feat/replace-pi3x-with-gs2mesh`` → ``feat-replace-pi3x-with-gs2mesh``
    """
    slug = branch.strip().replace("/", "-").replace("\\", "-")
    slug = re.sub(r"[^\w.-]", "-", slug, flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    slug = slug[:80]
    return slug or "main"
