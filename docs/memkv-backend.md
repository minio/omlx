# oMLX paged-SSD KV cache — pluggable storage backend

Status: design + initial implementation
Branch: `feat/memkv-backend`
Base: upstream `main`

## Motivation

oMLX's paged-SSD KV cache writes one safetensors blob per (block, all
layers) directly to the local filesystem. The cache survives a single
restart on one machine — but it is not shared across machines, has no
authentication on the persistence path, and grows linearly with disk
size on the host.

A multi-tenant fleet wants the same blocks to be reachable from any
oMLX node so a request landing on node B can resume a context that was
prefilled on node A. The persistence layer also wants HMAC auth and
replication, neither of which a local filesystem provides.

This change introduces a storage-backend abstraction in front of the
five filesystem touches in `omlx/cache/paged_ssd_cache.py`. The default
backend is the existing local FS path, byte-for-byte identical. A new
out-of-tree `memkv` backend writes the same safetensors bytes to a
[MemKV](https://github.com/miniohq/memkv) cluster.

Out of scope:
- No changes to the block hashing scheme (chain-sha256 stays).
- No changes to the cache file format or the `omlx_cache_format_version`
  field. Bytes on the wire are byte-identical to bytes on disk.
- No changes to the prefix-cache index, the tiered manager, or the
  hot-cache OrderedDict.
- No changes to recovery semantics: with a remote backend, the index
  on restart is rebuilt by enumerating known keys (or skipped for an
  ephemeral start; configurable).

## Surface area

`omlx/cache/storage_backend.py` (new): abstract `StorageBackend` plus
two implementations.

```python
class StorageBackend:
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list_keys(self) -> Iterator[str]: ...   # for restart-time index rebuild
    def get_metadata_only(self, key: str) -> dict | None: ...
        # safetensors header peek without pulling tensor bytes
```

`omlx/cache/paged_ssd_cache.py` is refactored so:

- `_write_safetensors_no_mx` constructs a `bytes` blob and hands it to
  `backend.put(key, blob)` instead of opening a path.
- The three `mx.load(path, return_metadata=True)` call sites pull bytes
  through `backend.get(key)`, write them to a temporary file, and call
  `mx.load`. (MLX's safetensors loader is path-based today; a future
  buffer-based variant would let us skip the temp-file hop.)
- The recovery scan in `_scan_existing_files` walks `backend.list_keys()`
  and pulls metadata-only headers via `backend.get_metadata_only(key)`
  rather than `mx.load(file_path, return_metadata=True)`.

## Backends

### `LocalFSBackend` (default)

Same paths as today: `<cache_dir>/<hash[0]>/<hash>.safetensors`. 16-way
fanout. `list_keys()` walks the existing tree; `get_metadata_only()`
reads the safetensors header without loading tensor bytes.

### `MemkvBackend`

`memkv://<host>:<port>/<namespace>` URIs. Auth via env
`MEMKV_AUTH_KEY`. Internally calls into `libslot_store_memkv.{so,dylib}`
through `ctypes`, reusing the cdylib that already exists for the
`llama.cpp` v2 chunked slot save integration.

Key layout:

- `<namespace>/blocks/<hex-block-hash>` — full safetensors blob
- `<namespace>/index/manifest` — optional snapshot of known block
  hashes, written periodically so a fresh oMLX process can boot
  without an enumerate-everything scan.

`list_keys()` reads the index manifest if present; otherwise returns
empty (oMLX warms up from new requests). `get_metadata_only()` issues
a partial-read request for the safetensors header bytes.

### Eviction

LRU eviction stays oMLX-side. When `evict_until_size()` evicts a
block, the backend's `delete(key)` is called. Inline LRU is unchanged.

## CLI / config

```
omlx serve \
    --cache-backend memkv \
    --cache-uri memkv://10.0.0.1:9900/omlx-prod \
    --cache-dir /var/cache/omlx-snapshots   # local mirror for restart-time index
```

A `local-fs` backend keeps the existing `--cache-dir` semantics and is
the default when `--cache-backend` is not specified.

## Why this is worth it

- **Multi-tenant prefix sharing**: every oMLX worker on the cluster
  resolves the same block hash to the same MemKV key. A coding-agent
  prefix that's been prefilled by one user is reused across the whole
  fleet.
- **Restart durability beyond one host**: a node failing or being
  redeployed loses its hot cache, but the cold tier survives in MemKV.
- **HMAC-authenticated persistence**: every byte ride a signed MemKV
  message; no tenant can poison another's keys.
- **Storage scaling decoupled from compute**: scale MemKV nodes
  independently of inference workers.

## Wire-compat with llama.cpp v2

Both consumers — llama.cpp v2 and oMLX — talk to MemKV through the
same `slot_store_v1` C ABI. They use different key prefixes
(`<ns>/c/...` for llama.cpp chunks, `<ns>/blocks/...` for oMLX blocks)
so a single MemKV cluster can serve both engines without collision.
The two engines do NOT share blocks today — the file shape is
different — but a future "common chunk store" abstraction at the
MemKV layer is plausible if it becomes useful.

## Implementation order

1. `storage_backend.py` with `StorageBackend` ABC + `LocalFSBackend`.
   Move all FS calls in `paged_ssd_cache.py` through it. Default
   behaviour byte-identical to today.
2. `MemkvBackend` via ctypes around `libslot_store_memkv`.
3. CLI plumbing for `--cache-backend` / `--cache-uri`.
4. Optional index-manifest snapshotting for restart speed.
5. Bench against local-fs + jundot/omlx upstream.

## Testing

- Unit: round-trip a block through both backends; verify byte-identical
  retrieval; verify metadata-only fast path.
- Integration: spin up MemKV + oMLX with a small model, drive a
  multi-turn agent workload (e.g. via Claude Code), confirm cache hits
  on the second turn.
- Restart: kill oMLX mid-conversation, restart with the same
  `--cache-uri`, confirm prefixes are resumed without recomputation.
