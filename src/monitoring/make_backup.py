"""Create a timestamped tar.gz backup of the whole project to a location OUTSIDE the repo, keeping the
5 most recent. A convenience safety-net (git is the primary version control). Excludes heavy/ignored
dirs (.git/.venv/__pycache__/caches).

Inputs:  the whole PROJECT_ROOT (minus excluded dirs); env SDF_BACKUP_DIR (default ~/sdf_backups).
Outputs: <SDF_BACKUP_DIR>/sdf_replication_backup_<timestamp>.tar.gz (keeps the 5 most recent).
Pipeline: standalone utility - run anytime to snapshot artifacts.
Run: python src/monitoring/make_backup.py
"""
import glob
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from config import PROJECT_ROOT  # noqa: E402

DST = os.path.expanduser(os.environ.get("SDF_BACKUP_DIR", "~/sdf_backups"))
os.makedirs(DST, exist_ok=True)
EXCLUDE = {".git", ".venv", "__pycache__", "backups", ".DS_Store", ".pytest_cache"}
out = os.path.join(DST, f"sdf_replication_backup_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz")


def _filter(ti):
    return None if any(p in EXCLUDE for p in ti.name.split("/")) else ti


with tarfile.open(out, "w:gz") as tar:
    tar.add(PROJECT_ROOT, arcname="sdf-replication", filter=_filter)

print(f"backup -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
backs = sorted(glob.glob(os.path.join(DST, "sdf_replication_backup_*.tar.gz")))
for old in backs[:-5]:
    os.remove(old)
print(f"  kept {len(backs[-5:])} backups in {DST}")
