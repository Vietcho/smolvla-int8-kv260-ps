"""Verify all copied W2 package files against package_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "package"


def canonical_bytes(path: Path, hash_mode: str) -> bytes:
    data = path.read_bytes()
    if hash_mode == "lf-normalized":
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if hash_mode != "binary":
        raise ValueError(f"Unsupported manifest hash_mode={hash_mode!r}")
    return data


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((PACKAGE / "package_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = PACKAGE / entry["path"]
        if not path.is_file():
            failures.append({"path": entry["path"], "error": "missing"})
            continue
        hash_mode = entry.get("hash_mode", "binary")
        try:
            data = canonical_bytes(path, hash_mode)
        except ValueError as error:
            failures.append({"path": entry["path"], "error": str(error)})
            continue
        if len(data) != entry["bytes"]:
            failures.append({"path": entry["path"], "error": "size mismatch"})
            continue
        actual = sha256_bytes(data)
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
