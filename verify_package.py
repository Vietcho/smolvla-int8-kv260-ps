"""Verify all copied W2 package files against package_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "package"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((PACKAGE / "package_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = PACKAGE / entry["path"]
        if not path.is_file():
            failures.append({"path": entry["path"], "error": "missing"})
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append({"path": entry["path"], "error": "size mismatch"})
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            failures.append({"path": entry["path"], "error": "sha256 mismatch"})
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked_files": len(manifest["files"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
