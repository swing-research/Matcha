from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path, PurePosixPath


PACKAGE_NAME = "matcha"


def get_packaged_path(relative_path: str | Path) -> Path | None:
    rel = PurePosixPath(str(relative_path).replace(os.sep, "/"))
    parts = [part for part in rel.parts if part not in ("", ".")]
    candidate = files(PACKAGE_NAME).joinpath(*parts)
    try:
        if candidate.is_file():
            return Path(str(candidate))
    except FileNotFoundError:
        return None
    return None
