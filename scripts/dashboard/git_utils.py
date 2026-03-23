"""Git branch detection and slug utilities for branch-based object isolation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

BRANCH_DIR_PREFIX = "@"

# Written by the host before ``docker compose build`` and COPY'd into the image.
_BRANCH_FILE_PATHS = (
    Path("/app/.git_branch"),   # inside Docker container
    Path(".git_branch"),        # local dev (project root)
)


def detect_branch() -> str:
    """Return the current git branch name.

    Priority:
    1. ``GIT_BRANCH`` environment variable
    2. ``.git_branch`` file (baked into Docker image at build time)
    3. ``git rev-parse --abbrev-ref HEAD`` (local dev with .git present)
    4. Fallback to ``"main"``
    """
    env_branch = os.environ.get("GIT_BRANCH", "").strip()
    if env_branch:
        return env_branch
    for p in _BRANCH_FILE_PATHS:
        try:
            content = p.read_text().strip()
            if content:
                return content
        except Exception:
            pass
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
