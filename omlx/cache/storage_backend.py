# SPDX-License-Identifier: Apache-2.0
"""
Storage backend abstraction for paged-SSD KV cache.

The paged SSD cache writes one safetensors blob per (block, all layers)
and rebuilds its index by walking the filesystem at startup. The blob
shape is independent of where the bytes live; this module factors that
location out so the same cache can persist to a local filesystem or to
a remote, multi-tenant key/value store such as MemKV.

The default backend (`LocalFSBackend`) is byte-for-byte identical to
the legacy direct-FS path. New backends plug in by subclassing
`StorageBackend` and registering with the factory.
"""

from __future__ import annotations

import json
import os
import struct
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# Subdirectory fanout — must match the legacy paged-SSD layout exactly so
# pre-existing files on disk remain discoverable.
SUBDIR_CHARS = "0123456789abcdef"


class StorageBackend(ABC):
    """Interface a paged-SSD cache uses to persist and retrieve safetensors blobs."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> int:
        """Store `data` under `key`. Atomic: a concurrent reader either sees
        the previous value (if any) or the new value, never a torn write.
        Returns the byte count written."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the stored bytes, or None if `key` is absent."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if `key` is present, without paying for the data."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Best-effort delete; no error if the key is absent."""

    @abstractmethod
    def list_keys(self) -> Iterator[str]:
        """Enumerate all keys this backend currently holds. Used at startup
        to rebuild the in-memory index."""

    @abstractmethod
    def get_metadata_only(self, key: str) -> dict | None:
        """Return the safetensors header metadata for `key` without loading
        tensor data. Returns None if the key is missing or unreadable."""

    @contextmanager
    @abstractmethod
    def open_for_load(self, key: str) -> Iterator["Path | None"]:
        """Yield a filesystem path that callers can hand to a path-based
        loader (e.g. `mx.load`) for the duration of the `with` block.

        For local-fs backends this is the actual on-disk path and the
        `with` block does nothing on exit. For remote backends the
        implementation materialises the bytes to a temp file and unlinks
        it on exit. Yields `None` if the key is absent.
        """

    def flush(self) -> None:
        """Force any backend-side buffered puts to the wire. No-op for
        backends that don't buffer; remote backends override to push the
        accumulated chunks in one EXISTS + batch_put round-trip."""
        return

    def put_manifest(self, name: str, data: bytes) -> None:
        """Atomically publish a name-addressed blob (the second namespace
        in the kv_store_v1 ABI). Used by the cache for periodic index
        snapshots so a remote backend can restore the in-memory index
        across process restarts. Default uses `put`; override on
        backends that have a dedicated manifest namespace."""
        self.put(f"_manifest/{name}", data)

    def get_manifest(self, name: str) -> bytes | None:
        """Companion to `put_manifest`. Returns None if absent."""
        return self.get(f"_manifest/{name}")

    def close(self) -> None:
        """Release resources. Default is no-op; remote backends override."""


def _peek_safetensors_metadata(blob_or_path) -> dict | None:
    """Decode the `__metadata__` dict from the safetensors header without
    loading any tensor data. Accepts either a raw bytes-like or a path-like
    that supports binary reads.

    The safetensors layout is:
        [8 B little-endian uint64: header_size]
        [header_size bytes: JSON header (tensors + optional __metadata__)]
        [tensor data]
    """
    if isinstance(blob_or_path, (bytes, bytearray, memoryview)):
        if len(blob_or_path) < 8:
            return None
        header_size = struct.unpack_from("<Q", blob_or_path, 0)[0]
        if 8 + header_size > len(blob_or_path):
            return None
        header_json = bytes(blob_or_path[8:8 + header_size])
    else:
        try:
            with open(blob_or_path, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    return None
                header_size = struct.unpack("<Q", head)[0]
                header_json = f.read(header_size)
                if len(header_json) < header_size:
                    return None
        except OSError:
            return None
    try:
        header = json.loads(header_json)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    md = header.get("__metadata__")
    return md if isinstance(md, dict) else None


class LocalFSBackend(StorageBackend):
    """Filesystem-backed storage. Writes ``<cache_dir>/<key[0]>/<key>.safetensors``
    with a 16-way subdirectory fanout — the same layout the legacy paged-SSD
    cache uses, so existing on-disk files stay readable."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        for c in SUBDIR_CHARS:
            (self._cache_dir / c).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._cache_dir / key[0] / f"{key}.safetensors"

    def put(self, key: str, data: bytes) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + "_tmp.safetensors")
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(str(tmp), str(path))
        return len(data)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass

    def list_keys(self) -> Iterator[str]:
        for c in SUBDIR_CHARS:
            sub = self._cache_dir / c
            if not sub.exists():
                continue
            for f in sub.iterdir():
                if f.is_file() and f.suffix == ".safetensors":
                    yield f.stem

    def get_metadata_only(self, key: str) -> dict | None:
        return _peek_safetensors_metadata(self._path(key))

    @contextmanager
    def open_for_load(self, key: str) -> Iterator["Path | None"]:
        path = self._path(key)
        yield path if path.exists() else None

    @property
    def cache_dir(self) -> Path:
        """Filesystem directory this backend manages. Used by callers that
        need to expose a real path (e.g. for `mx.load`, which is
        path-based today)."""
        return self._cache_dir


def make_backend(uri_or_path) -> StorageBackend:
    """Factory: returns a backend instance based on the URI scheme.

    - No scheme (or `file://`) → `LocalFSBackend` rooted at the path.
    - `memkv://...` → `MemkvBackend` (defined in storage_memkv.py; lazy
      import to keep the optional dependency truly optional).
    """
    if isinstance(uri_or_path, Path):
        return LocalFSBackend(uri_or_path)
    s = str(uri_or_path)
    if "://" not in s or s.startswith("file://"):
        path = s.removeprefix("file://") if s.startswith("file://") else s
        return LocalFSBackend(Path(path))
    scheme = s.split("://", 1)[0]
    if scheme == "memkv":
        from .storage_memkv import MemkvBackend
        return MemkvBackend(s)
    raise ValueError(f"Unknown storage backend scheme: {scheme!r}")
