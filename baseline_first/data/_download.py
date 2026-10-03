"""Download a file once, verify its checksum, and cache it."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_CHUNK = 1 << 20


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, md5: str) -> Path:
    """Download `url` to `dest` unless a file with the expected md5 is already there.

    The download goes to a ``.part`` file and is renamed only after the checksum
    matches, so an interrupted or altered download never looks like a cached one.
    """
    dest = Path(dest)
    if dest.exists():
        if md5sum(dest) != md5:
            raise ValueError(f"{dest} exists but its md5 does not match {md5}; delete it")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "baseline-first"})
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        while chunk := response.read(_CHUNK):
            out.write(chunk)

    actual = md5sum(part)
    if actual != md5:
        part.unlink()
        raise ValueError(f"md5 mismatch for {url}: expected {md5}, got {actual}")
    part.replace(dest)
    return dest
