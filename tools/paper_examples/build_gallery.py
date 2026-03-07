from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat.paper_gallery import write_gallery_outputs


if __name__ == "__main__":
    manifest_path, markdown_path = write_gallery_outputs(REPO_ROOT)
    print(manifest_path)
    print(markdown_path)
