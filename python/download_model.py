#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_modelscope(model: str, output: Path) -> dict:
    api_url = f"https://modelscope.cn/api/v1/models/{model}/repo/files"
    response = requests.get(api_url, params={"Revision": "master", "Root": ""}, timeout=30)
    response.raise_for_status()
    records = response.json()["Data"]["Files"]
    required = {
        "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    }
    manifest_files = {}
    for record in records:
        name = record["Name"]
        if name not in required:
            continue
        target = output / name
        expected_size = int(record["Size"])
        expected_sha = record["Sha256"]
        if not target.exists() or target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
            url = f"https://modelscope.cn/models/{model}/resolve/master/{name}"
            with requests.get(url, stream=True, timeout=(30, 300)) as download:
                download.raise_for_status()
                temporary = target.with_suffix(target.suffix + ".partial")
                with temporary.open("wb") as handle:
                    for chunk in download.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                temporary.replace(target)
        actual_sha = sha256_file(target)
        if target.stat().st_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(f"checksum mismatch for {name}")
        manifest_files[name] = {
            "sha256": actual_sha,
            "size": expected_size,
            "revision": record["Revision"],
        }
    missing = required.difference(manifest_files)
    if missing:
        raise RuntimeError(f"ModelScope repository missing files: {sorted(missing)}")
    return {"model": model, "source": "modelscope", "revision": "master", "files": manifest_files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.source == "modelscope":
        manifest = download_modelscope(args.model, args.output)
        path = args.output
    else:
        info = HfApi().model_info(args.model)
        revision = info.sha
        path = snapshot_download(
            repo_id=args.model,
            revision=revision,
            local_dir=args.output,
            local_dir_use_symlinks=False,
        )
        files = {}
        for file in sorted(Path(path).rglob("*")):
            if file.is_file() and ".cache" not in file.parts:
                digest = sha256_file(file)
                files[str(file.relative_to(path))] = {"sha256": digest, "size": file.stat().st_size}
        manifest = {"model": args.model, "source": "huggingface", "revision": revision, "files": files}
    (args.output / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"path": str(Path(path).resolve()), "revision": manifest["revision"],
                      "source": manifest["source"]}, indent=2))


if __name__ == "__main__":
    main()
