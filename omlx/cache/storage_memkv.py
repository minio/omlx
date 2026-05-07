# SPDX-License-Identifier: Apache-2.0
"""
MemKV backend for the paged-SSD KV store, using the kv_store_v1 dlopen ABI.

Loads `libkv_store_memkv.{so,dylib}` (the same cdylib llama.cpp loads for
its v2 chunked slot save format) via ctypes and routes safetensors blob
puts/gets through MemKV instead of the local filesystem.

URI form:    memkv://<host>:<port>/<namespace>
Auth key:    env MEMKV_AUTH_KEY (64 hex chars / 32 bytes)
.so search:  KV_STORE_LIBRARY_PATH (absolute dir), then the system loader
             (LD_LIBRARY_PATH on Linux). On macOS DYLD_LIBRARY_PATH is
             stripped from many child processes; KV_STORE_LIBRARY_PATH is
             the supported fallback.

The full ABI specification is at https://min.io/memkv/kv-store-abi.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .storage_backend import StorageBackend, _peek_safetensors_metadata


# ---- C ABI typedefs that mirror tools/server/kv_store_abi.h --------------


class _KvStoreV1(ctypes.Structure):
    """Opaque handle returned by the backend's `open(uri)`."""


_KvStorePtr = ctypes.POINTER(_KvStoreV1)
_U8Ptr = ctypes.POINTER(ctypes.c_uint8)
_U8PtrPtr = ctypes.POINTER(_U8Ptr)
_SizePtr = ctypes.POINTER(ctypes.c_size_t)


_OpenFn = ctypes.CFUNCTYPE(_KvStorePtr, ctypes.c_char_p)
_CloseFn = ctypes.CFUNCTYPE(None, _KvStorePtr)
_PutChunkFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    _U8Ptr, ctypes.c_size_t,
    _U8Ptr, ctypes.c_size_t,
)
_GetChunkFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    _U8Ptr, ctypes.c_size_t,
    _U8PtrPtr, _SizePtr,
)
_PutManifestFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    ctypes.c_char_p,
    _U8Ptr, ctypes.c_size_t,
)
_GetManifestFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    ctypes.c_char_p,
    _U8PtrPtr, _SizePtr,
)
_DeleteManifestFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    ctypes.c_char_p,
)
_PrefetchChunksFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _KvStorePtr,
    _U8Ptr, ctypes.c_size_t, ctypes.c_size_t,
)


class _KvStoreVtable(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("open", _OpenFn),
        ("close", _CloseFn),
        ("put_chunk", _PutChunkFn),
        ("get_chunk", _GetChunkFn),
        ("put_manifest", _PutManifestFn),
        ("get_manifest", _GetManifestFn),
        ("delete_manifest", _DeleteManifestFn),
        # version 2
        ("prefetch_chunks", _PrefetchChunksFn),
    ]


# ---- shared library load --------------------------------------------------


def _shlib_name(scheme: str) -> str:
    if sys.platform == "darwin":
        return f"libkv_store_{scheme}.dylib"
    return f"libkv_store_{scheme}.so"


def _load_lib(scheme: str) -> ctypes.CDLL:
    name = _shlib_name(scheme)
    # Honour KV_STORE_LIBRARY_PATH first — macOS strips DYLD_LIBRARY_PATH
    # from many child processes, so the env var below is the supported
    # cross-platform way to point at a custom directory.
    dir_hint = os.environ.get("KV_STORE_LIBRARY_PATH")
    if dir_hint:
        candidate = Path(dir_hint) / name
        if candidate.exists():
            return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_LOCAL)
    # Fall back to the system loader.
    return ctypes.CDLL(name, mode=ctypes.RTLD_LOCAL)


def _load_libc_free():
    """Resolve `free()` so we can release malloc'd buffers the backend
    returns. The output buffers from get_chunk/get_manifest are allocated
    with the C `malloc`; the backend ABI is explicit that the consumer
    frees them."""
    libc_name = "msvcrt" if sys.platform == "win32" else (
        ctypes.util.find_library("c") or "libc.so.6"
    )
    libc = ctypes.CDLL(libc_name)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    return libc.free


# ---- backend --------------------------------------------------------------


class MemkvBackend(StorageBackend):
    """A `StorageBackend` that persists each safetensors blob as one chunk
    in MemKV. The block hash (sha256, 32 B) is the chunk's content key.

    paged-SSD's runtime handle is the raw `bytes(block_hash)` which oMLX
    already keeps as the file basename; we reuse the same hex-encoded
    identifier so existing index code doesn't need to learn a new key
    shape.

    Recovery on startup is currently no-op: the cache starts cold and
    repopulates as new requests come in. A persisted index manifest is on
    the roadmap; for the prototype, "warm up the next request after a
    server restart" is acceptable behaviour."""

    _scheme = "memkv"

    def __init__(self, uri: str):
        if not uri.startswith("memkv://"):
            raise ValueError(f"MemkvBackend expects a memkv:// URI, got {uri!r}")
        self._uri = uri
        self._lib = _load_lib(self._scheme)

        getter = self._lib.kv_store_get_vtable
        getter.argtypes = []
        getter.restype = ctypes.POINTER(_KvStoreVtable)
        vt_ptr = getter()
        if not vt_ptr:
            raise RuntimeError(f"{_shlib_name(self._scheme)}: kv_store_get_vtable returned NULL")
        self._vt: _KvStoreVtable = vt_ptr.contents
        if self._vt.version < 1:
            raise RuntimeError(
                f"{_shlib_name(self._scheme)}: unsupported vtable version {self._vt.version}"
            )

        # Open the per-instance handle. The URI passed is the full one the
        # caller gave us; the backend strips any trailing slash internally.
        self._handle = self._vt.open(uri.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"kv_store open failed for {uri!r}")

        self._free = _load_libc_free()
        self._closed = False
        self._lock = threading.Lock()
        # Set when a put has not yet been flushed via put_manifest. Reads
        # auto-flush so writes are visible read-after-write on the same
        # backend instance; the writer thread flushes when its queue
        # drains to amortise round-trips across many puts.
        self._unflushed = False

    # ---- StorageBackend impl ---------------------------------------------

    def put(self, key: str, data: bytes) -> int:
        """Buffer the put; the backend coalesces buffered chunks into a
        single EXISTS + batch_put round-trip on the next `flush()` or
        `put_manifest()` call. Callers that need read-after-write
        consistency must flush before reading the same key."""
        self._check_open()
        hash_bytes, hash_len = _decode_key_hex(key)
        rc = self._vt.put_chunk(
            self._handle,
            hash_bytes, hash_len,
            (ctypes.c_uint8 * len(data)).from_buffer_copy(data), len(data),
        )
        if rc < 0:
            raise OSError(f"put_chunk for {key} returned {rc}")
        self._unflushed = True
        return len(data)

    def flush(self) -> None:
        """Force any buffered puts to the server. Cheap if the buffer is
        empty. The MemKV backend implements this by writing a sentinel
        manifest, which makes the kv_store_v1 backend run its
        EXISTS-skip + batch_put pipeline in one shot."""
        if not getattr(self, "_unflushed", False):
            return
        self._flush_via_sentinel_manifest()
        self._unflushed = False

    def get(self, key: str) -> bytes | None:
        self._check_open()
        # Read-after-write consistency: ensure any buffered puts are
        # visible to a `get` for the same key from the same backend
        # instance.
        if self._unflushed:
            self.flush()
        hash_bytes, hash_len = _decode_key_hex(key)
        out_data = _U8Ptr()
        out_len = ctypes.c_size_t(0)
        rc = self._vt.get_chunk(
            self._handle,
            hash_bytes, hash_len,
            ctypes.byref(out_data), ctypes.byref(out_len),
        )
        if rc < 0:
            return None
        try:
            return bytes((ctypes.c_uint8 * out_len.value).from_address(
                ctypes.addressof(out_data.contents)
            ))
        finally:
            self._free(ctypes.cast(out_data, ctypes.c_void_p))

    def exists(self, key: str) -> bool:
        # No dedicated chunk-level EXISTS in the v1 ABI for the consumer
        # surface; round-trip a get and discard. `get` already flushes
        # any buffered puts, so read-after-write on the same instance is
        # consistent.
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        # The kv_store_v1 ABI has no chunk-level delete — chunks are
        # content-addressed, GC'd via manifest refcount. For oMLX's
        # eviction the call is a no-op; the next save with the same hash
        # will be a dedup hit and the unreferenced chunk eventually GC'd
        # by a separate sweep.
        return

    def list_keys(self) -> Iterator[str]:
        # No remote enumeration today — see class docstring.
        return iter(())

    def get_metadata_only(self, key: str) -> dict | None:
        blob = self.get(key)
        if blob is None:
            return None
        return _peek_safetensors_metadata(blob)

    def put_manifest(self, name: str, data: bytes) -> None:
        """Use the kv_store_v1 vtable's name-addressed manifest namespace
        directly so the consumer's index snapshots travel under the same
        keying convention the spec defines."""
        self._check_open()
        with self._lock:
            rc = self._vt.put_manifest(
                self._handle,
                name.encode("utf-8"),
                (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else (ctypes.c_uint8 * 0)(),
                len(data),
            )
            if rc < 0:
                raise OSError(f"put_manifest({name!r}) returned {rc}")

    def get_manifest(self, name: str) -> bytes | None:
        self._check_open()
        out_data = _U8Ptr()
        out_len = ctypes.c_size_t(0)
        rc = self._vt.get_manifest(
            self._handle,
            name.encode("utf-8"),
            ctypes.byref(out_data), ctypes.byref(out_len),
        )
        if rc < 0:
            return None
        try:
            return bytes((ctypes.c_uint8 * out_len.value).from_address(
                ctypes.addressof(out_data.contents)
            ))
        finally:
            self._free(ctypes.cast(out_data, ctypes.c_void_p))

    @contextmanager
    def open_for_load(self, key: str) -> Iterator[Path | None]:
        """Materialise the blob as a tempfile so callers can pass a path
        to MLX's `mx.load` (which is path-based today). The tempfile is
        unlinked when the context exits."""
        blob = self.get(key)
        if blob is None:
            yield None
            return
        # Use a named tempfile in the system temp dir; mlx loads it then
        # we unlink. The file lives only for the duration of the load.
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"omlx-{key[:16]}-", suffix=".safetensors", delete=False
        )
        try:
            tmp.write(blob)
            tmp.flush()
            tmp.close()
            yield Path(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ---- internals -------------------------------------------------------

    _SENTINEL_NAME = b"__omlx_index__"

    def _flush_via_sentinel_manifest(self) -> None:
        """Force the backend to flush its pending-puts buffer by writing a
        sentinel manifest. The kv_store_v1 contract says backends MAY
        buffer put_chunk calls until the next put_manifest, so for
        oMLX's per-block save model we issue a manifest write after every
        chunk to make puts immediately visible. This is wasteful per-call
        but acceptable while the prototype settles."""
        with self._lock:
            rc = self._vt.put_manifest(
                self._handle, self._SENTINEL_NAME,
                (ctypes.c_uint8 * 0)(), 0,
            )
            if rc < 0:
                raise OSError(f"sentinel put_manifest returned {rc}")

    def _check_open(self) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("MemkvBackend is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle and self._vt.close:
            self._vt.close(self._handle)
            self._handle = None  # type: ignore[assignment]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _decode_key_hex(key: str) -> tuple[ctypes.Array, int]:
    """Convert a hex-string block hash into a ctypes uint8 array suitable
    for the C ABI."""
    raw = bytes.fromhex(key)
    arr = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    return arr, len(raw)
