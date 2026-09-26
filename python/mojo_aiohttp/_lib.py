"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below stay `c_int64` for addresses; `c_int` truncates
them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aiohttp.so"

_I = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.ah_ws_mask.restype = None
    lib.ah_ws_mask.argtypes = [_I, _I, _I]

    lib.ah_ws_frame_header.restype = _I
    lib.ah_ws_frame_header.argtypes = [_I, _I, _I, _I]

    lib.ah_hex_to_u64.restype = ctypes.c_int64
    lib.ah_hex_to_u64.argtypes = [_I, _I, _I]

    lib.ah_chunked_parse.restype = _I
    lib.ah_chunked_parse.argtypes = [_I, _I, _I, _I, _I, _I, _I]

    lib.ah_headers_parse.restype = _I
    lib.ah_headers_parse.argtypes = [_I, _I, _I, _I, _I, _I, _I, _I]

    lib.ah_is_tchar.restype = _I
    lib.ah_is_tchar.argtypes = [_I]

    lib.ah_content_length_plan.restype = _I
    lib.ah_content_length_plan.argtypes = [_I, _I, _I, _I]

    lib.ah_bytes_until_limit.restype = _I
    lib.ah_bytes_until_limit.argtypes = [_I, _I, _I, _I]
    return lib


lib = _load()

# ah_ws_frame_header out-array layout.
WS_FIN, WS_RSV1, WS_RSV2, WS_RSV3, WS_OPCODE = 0, 1, 2, 3, 4
WS_HAS_MASK, WS_LEN_FLAG, WS_PAYLOAD_LEN, WS_HEADER_LEN = 5, 6, 7, 8
WS_STATUS, WS_MASK_OFFSET, WS_PAYLOAD_START = 9, 10, 11
WS_OUT_SLOTS = 12

# ah_chunked_parse out-array layout (offsets from max_chunks).
CHUNKED_N_CHUNKS = 0
CHUNKED_DECODED = 1
CHUNKED_TRAILER_OFF = 2
CHUNKED_STATUS = 3
CHUNKED_BODY_END = 4

# ah_headers_parse out-array layout.
HDR_NAME_OFF, HDR_NAME_LEN, HDR_VAL_OFF, HDR_VAL_LEN = 0, 1, 2, 3
HDR_COUNT, HDR_STATUS, HDR_FOLDS, HDR_VALUE_BYTES = 0, 1, 2, 3

DEFAULT_MAX_LINE_SIZE = 8190
DEFAULT_MAX_FIELD_SIZE = 8190
DEFAULT_MAX_TRAILERS = 128
DEFAULT_MAX_HEADERS = 256


class ParseStatus(Exception):
    """Raised when a kernel rejects its input; ``status`` is the raw code."""

    def __init__(self, status: int, what: str):
        super().__init__(f"{what} failed with status {status}")
        self.status = status


def _u8(buf) -> np.ndarray:
    """Normalise bytes-like input to a contiguous uint8 view, no copy when the
    input already is one."""
    if (
        isinstance(buf, np.ndarray)
        and buf.dtype == np.uint8
        and buf.flags.c_contiguous
    ):
        return buf
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return np.frombuffer(buf, dtype=np.uint8)
    return np.ascontiguousarray(buf, dtype=np.uint8)


def _i64(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def ws_mask(data: bytearray, mask: bytes) -> bytearray:
    """RFC 6455 5.3 masking, in place. `mask` is exactly 4 bytes."""
    if len(mask) != 4:
        raise ValueError(f"WebSocket mask must be 4 bytes, got {len(mask)}")
    if not data:
        return data
    buf = _u8(data)
    if not buf.flags.writeable:
        raise ValueError("ws_mask needs a mutable buffer")
    m = _u8(mask)
    lib.ah_ws_mask(m.ctypes.data, buf.ctypes.data, buf.size)
    return data


def ws_frame_header(buf, start: int = 0) -> dict:
    """Decode one frame header. Returns the RFC 6455 fields as a dict.

    ``status`` is 0 on success and 1 when `buf` does not yet hold a complete
    header; ``payload_start`` is the offset of the first payload byte.
    """
    b = _u8(buf)
    out = _i64(WS_OUT_SLOTS)
    ret = lib.ah_ws_frame_header(b.ctypes.data, b.size, start, out.ctypes.data)
    return {
        "fin": bool(out[WS_FIN]),
        "rsv1": bool(out[WS_RSV1]),
        "rsv2": bool(out[WS_RSV2]),
        "rsv3": bool(out[WS_RSV3]),
        "opcode": int(out[WS_OPCODE]),
        "has_mask": bool(out[WS_HAS_MASK]),
        "len_flag": int(out[WS_LEN_FLAG]),
        "payload_len": int(out[WS_PAYLOAD_LEN]),
        "header_len": int(out[WS_HEADER_LEN]),
        "mask_offset": int(out[WS_MASK_OFFSET]),
        "payload_start": int(out[WS_PAYLOAD_START]),
        "status": int(out[WS_STATUS]),
        "return": int(ret),
    }


def hex_to_u64(buf, start: int, n: int) -> int:
    """Parse `n` hex digits as an unsigned int, or -1 if any digit is bad."""
    b = _u8(buf)
    return int(lib.ah_hex_to_u64(b.ctypes.data, start, n))


def parse_chunked(
    body: bytes,
    max_chunks: int = 1024,
    max_line_size: int = DEFAULT_MAX_LINE_SIZE,
    max_field_size: int = DEFAULT_MAX_FIELD_SIZE,
    max_trailers: int = DEFAULT_MAX_TRAILERS,
) -> dict:
    """Walk a chunked body. Returns the chunk sizes and where trailers start.

    Raises `ParseStatus` when the kernel rejects the framing.
    """
    b = _u8(body)
    out = _i64(max_chunks + 5)
    ret = lib.ah_chunked_parse(
        b.ctypes.data,
        b.size,
        out.ctypes.data,
        max_chunks,
        max_line_size,
        max_field_size,
        max_trailers,
    )
    n = int(out[max_chunks + CHUNKED_N_CHUNKS])
    status = int(out[max_chunks + CHUNKED_STATUS])
    if status != 0 or ret < 0:
        raise ParseStatus(status, "chunked parse")
    return {
        "sizes": out[:n].copy(),
        "decoded_length": int(out[max_chunks + CHUNKED_DECODED]),
        "trailer_offset": int(out[max_chunks + CHUNKED_TRAILER_OFF]),
        "body_end": int(out[max_chunks + CHUNKED_BODY_END]),
    }


def parse_headers(
    block: bytes,
    lax: bool = False,
    max_headers: int = DEFAULT_MAX_HEADERS,
    max_field_size: int = DEFAULT_MAX_FIELD_SIZE,
) -> list:
    """Parse an RFC 9110 field block into ``[(name, value), ...]``.

    `block` is the raw CRLF-separated field section *including* the terminating
    empty line, which is what aiohttp's `HeadersParser` consumes. Raises
    `ParseStatus` on a malformed field line.
    """
    b = _u8(block)
    out = _i64(4 * max_headers + 4)
    vbuf = np.zeros(max(b.size, 1), dtype=np.uint8)
    n = lib.ah_headers_parse(
        b.ctypes.data,
        b.size,
        out.ctypes.data,
        max_headers,
        1 if lax else 0,
        max_field_size,
        vbuf.ctypes.data,
        vbuf.size,
    )
    # tolist() the used prefix once: indexing a numpy array element by element
    # costs ~100ns a pop, which dwarfs the kernel on a many-header block
    flat = out[: 4 * int(n)].tolist()
    base = 4 * max_headers
    status = int(out[base + HDR_STATUS])
    if status != 0 or n < 0:
        raise ParseStatus(status, "header parse")
    src = b.tobytes()
    vals = vbuf.tobytes()
    pairs = []
    for i in range(int(n)):
        s = 4 * i
        no, nl = flat[s + HDR_NAME_OFF], flat[s + HDR_NAME_OFF + 1]
        vo, vl = flat[s + HDR_VAL_OFF], flat[s + HDR_VAL_OFF + 1]
        pairs.append((src[no : no + nl], vals[vo : vo + vl]))
    return pairs


def parse_header_offsets(
    block: bytes,
    lax: bool = False,
    max_headers: int = DEFAULT_MAX_HEADERS,
    max_field_size: int = DEFAULT_MAX_FIELD_SIZE,
):
    """Parse a field block and return the raw kernel result, unmarshalled.

    Returns ``(offsets, values, count, folds)`` where ``offsets`` is an
    ``(n, 4)`` int64 array of (name offset, name length, value offset in
    `values`, value length) and ``values`` is the assembled value buffer. This
    is the parse without the per-header Python object construction, which is
    what the kernel-side benchmark measures.
    """
    b = _u8(block)
    out = _i64(4 * max_headers + 4)
    vbuf = np.zeros(max(b.size, 1), dtype=np.uint8)
    n = lib.ah_headers_parse(
        b.ctypes.data,
        b.size,
        out.ctypes.data,
        max_headers,
        1 if lax else 0,
        max_field_size,
        vbuf.ctypes.data,
        vbuf.size,
    )
    base = 4 * max_headers
    status = int(out[base + HDR_STATUS])
    if status != 0 or n < 0:
        raise ParseStatus(status, "header parse")
    return out[: 4 * int(n)].reshape(int(n), 4), vbuf, int(n), int(
        out[base + HDR_FOLDS]
    )


def header_fold_count(
    block: bytes, lax: bool = True, max_field_size: int = DEFAULT_MAX_FIELD_SIZE
) -> int:
    """Number of obs-fold continuation lines consumed by `parse_headers`."""
    b = _u8(block)
    out = _i64(4 * DEFAULT_MAX_HEADERS + 4)
    vbuf = np.zeros(max(b.size, 1), dtype=np.uint8)
    lib.ah_headers_parse(
        b.ctypes.data,
        b.size,
        out.ctypes.data,
        DEFAULT_MAX_HEADERS,
        1 if lax else 0,
        max_field_size,
        vbuf.ctypes.data,
        vbuf.size,
    )
    return int(out[4 * DEFAULT_MAX_HEADERS + HDR_FOLDS])


def content_length_plan(content_length: int, feed_sizes):
    """Replay aiohttp's PARSE_LENGTH bookkeeping.

    Returns ``(consumed_per_feed, trailing_bytes, still_outstanding)``.
    """
    feeds = np.ascontiguousarray(feed_sizes, dtype=np.int64)
    out = _i64(feeds.size + 1)
    remaining = lib.ah_content_length_plan(
        content_length, feeds.ctypes.data, feeds.size, out.ctypes.data
    )
    return out[: feeds.size].copy(), int(out[feeds.size]), int(remaining)


def is_tchar(c: int) -> bool:
    return bool(lib.ah_is_tchar(int(c)))


def projected_message_size(limit: int, have: int, chunk_sizes) -> int:
    b = np.ascontiguousarray(chunk_sizes, dtype=np.int64)
    return int(lib.ah_bytes_until_limit(limit, have, b.ctypes.data, b.size))
