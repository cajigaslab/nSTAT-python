from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "examples" / "help_topics" / "figure_contract.json"
REFERENCE_ROOT = PROJECT_ROOT / "reference" / "matlab_helpfigures"
MANIFEST_PATH = REFERENCE_ROOT / "manifest.json"


def _load_contract() -> dict[str, dict[str, object]]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid or empty figure contract: {CONTRACT_PATH}")
    return topics


def _topic_candidates(helpfiles: Path, topic: str) -> list[Path]:
    exact_prefix = re.compile(rf"^{re.escape(topic)}(?:$|[_-].*)")
    numbered = re.compile(rf"^{re.escape(topic)}_(\d+)$")

    all_png = sorted(helpfiles.glob("*.png"))
    candidates: list[Path] = []
    for path in all_png:
        stem = path.stem
        lower = stem.lower()
        if "_eq" in lower:
            continue
        if topic == "DecodingExample" and stem.startswith("DecodingExampleWithHist"):
            continue
        if topic == "AnalysisExamples" and stem.startswith("AnalysisExamples2"):
            continue
        if not exact_prefix.match(stem):
            continue
        candidates.append(path)

    with_index: list[tuple[int, Path]] = []
    fallback: list[Path] = []
    for path in candidates:
        m = numbered.match(path.stem)
        if m:
            with_index.append((int(m.group(1)), path))
        else:
            fallback.append(path)

    with_index.sort(key=lambda item: item[0])
    fallback.sort(key=lambda p: p.name)

    ordered = [p for _, p in with_index]
    ordered.extend(fallback)
    return ordered


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_helpfiles_hint() -> Path | None:
    hints = [
        PROJECT_ROOT / ".." / "nSTAT" / "helpfiles",
        PROJECT_ROOT / ".." / "nSTAT_currentRelease_Local" / "helpfiles",
    ]
    for hint in hints:
        h = hint.resolve()
        if h.exists() and h.is_dir():
            return h
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync MATLAB help PNG baselines into reference/matlab_helpfigures")
    parser.add_argument(
        "--matlab-helpfiles",
        default=None,
        help="Path to MATLAB nSTAT helpfiles directory containing *.png",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing topic subfolders under reference/matlab_helpfigures before sync",
    )
    args = parser.parse_args()

    helpfiles = Path(args.matlab_helpfiles).expanduser().resolve() if args.matlab_helpfiles else _default_helpfiles_hint()
    if helpfiles is None or not helpfiles.exists():
        raise RuntimeError("Unable to resolve MATLAB helpfiles path. Pass --matlab-helpfiles.")

    contract = _load_contract()
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for child in REFERENCE_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    manifest: dict[str, dict[str, object]] = {"topics": {}}

    for topic, info in sorted(contract.items()):
        expected = int(info.get("expected_figures", 0))
        topic_dir = REFERENCE_ROOT / topic
        if topic_dir.exists():
            for p in topic_dir.glob("fig_*.png"):
                p.unlink()
        topic_dir.mkdir(parents=True, exist_ok=True)

        selected: list[Path] = []
        if expected > 0:
            ordered = _topic_candidates(helpfiles, topic)
            if len(ordered) < expected:
                raise RuntimeError(
                    f"Insufficient baseline PNGs for {topic}: expected {expected}, found {len(ordered)} in {helpfiles}"
                )
            selected = ordered[:expected]

        baseline_files: list[str] = []
        sha_map: dict[str, str] = {}
        for idx, src in enumerate(selected, start=1):
            dst = topic_dir / f"fig_{idx:03d}.png"
            shutil.copyfile(src, dst)
            rel = str(dst.relative_to(REFERENCE_ROOT))
            baseline_files.append(rel)
            sha_map[rel] = _sha256(dst)

        manifest["topics"][topic] = {
            "expected_figures": expected,
            "baseline_files": baseline_files,
            "sha256": sha_map,
            "matlab_target": str(info.get("matlab_target", "")),
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "matlab_helpfiles": str(helpfiles),
                "topics": len(manifest["topics"]),
                "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                "reference_root": str(REFERENCE_ROOT.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
