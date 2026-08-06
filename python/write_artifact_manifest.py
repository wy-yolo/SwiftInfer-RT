#!/usr/bin/env python3
"""Write deterministic SHA256 manifests for generated artifact directories."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = (args.output or directory / "manifest.json").resolve()
    files = []
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        if path.resolve() == output:
            continue
        files.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "directory": str(directory),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
