"""Correctness-gated benchmark for mojo-aiohttp.

Every case checks agreement with the reference implementation before timing, so
a regression in the Mojo kernels shows up as a correctness failure rather than
a suspiciously good number.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_aiohttp as mh  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_mask(n: int = 1 << 22):
    """RFC 6455 masking, against NumPy strided views. NumPy's `translate` is
    not available for a 4-byte cyclic mask, so the fair vectorised baseline is
    four strided XORs with wraparound via a tiled mask."""
    rng = np.random.default_rng(0)
    data = bytearray(rng.integers(0, 256, n, dtype=np.uint8).tolist())
    mask = b"\x37\xfa\x21\x3d"

    got = bytes(mh.ws_mask(bytearray(data), mask))
    tiled = np.tile(np.frombuffer(mask, dtype=np.uint8), n // 4 + 1)[:n]
    expect = (np.frombuffer(bytes(data), dtype=np.uint8) ^ tiled).tobytes()
    assert got == expect, "mask mismatch"

    ref = np.frombuffer(bytes(data), dtype=np.uint8)

    def run_numpy():
        buf = ref.copy()
        buf[0::4] ^= mask[0]
        buf[1::4] ^= mask[1]
        buf[2::4] ^= mask[2]
        buf[3::4] ^= mask[3]

    def run_mojo():
        mh.ws_mask(bytearray(data), mask)

    return f"ws_mask n={n}", _time(run_numpy), _time(run_mojo)


def bench_mask_pure_python(n: int = 1 << 16):
    """The same masking against aiohttp's own pure-Python fallback, which is
    what runs when the Cython extension is unavailable."""
    from aiohttp._websocket.helpers import _websocket_mask_python

    data = bytes(np.random.default_rng(1).integers(0, 256, n, dtype=np.uint8).tolist())
    mask = b"\xde\xad\xbe\xef"
    ref = bytearray(data)
    _websocket_mask_python(mask, ref)
    got = bytes(mh.ws_mask(bytearray(data), mask))
    assert got == bytes(ref), "mask mismatch vs aiohttp"

    def run_theirs():
        buf = bytearray(data)
        _websocket_mask_python(mask, buf)

    def run_mine():
        mh.ws_mask(bytearray(data), mask)

    return f"ws_mask vs pure-python n={n}", _time(run_theirs), _time(run_mine)


def bench_chunked(body_len: int = 1 << 20):
    """Chunked framing walk, against a bytearray `find` loop in Python. There
    is no vectorised NumPy formulation of a CRLF chunk-size grammar, so the
    honest baseline is the same walk written in Python."""
    chunk = 4096
    parts = [b"%x\r\n" % chunk + b"x" * chunk + b"\r\n" for _ in range(body_len // chunk)]
    raw = b"".join(parts) + b"0\r\n\r\n"
    result = mh.parse_chunked(raw)
    assert result["decoded_length"] == chunk * (body_len // chunk)

    def run_python():
        pos = 0
        total = 0
        n = len(raw)
        while True:
            nl = raw.find(b"\r\n", pos)
            if nl < 0:
                break
            size = int(raw[pos:nl], 16)
            pos = nl + 2
            if size == 0:
                break
            total += size
            pos += size + 2
        return total

    assert run_python() == result["decoded_length"]
    return f"parse_chunked {body_len}B", _time(run_python), _time(
        lambda: mh.parse_chunked(raw)
    )


def bench_headers(count: int = 20000):
    max_headers = count + 8
    """Field parsing, against a Python `bytes.split` + `strip` per line."""
    lines = [b"X-Header-%05d: value-%05d-with-some-padding" % (i, i) for i in range(count)]
    blob = b"\r\n".join(lines) + b"\r\n\r\n"
    assert len(mh.parse_headers(blob, max_headers=max_headers)) == count

    def run_python():
        out = []
        for line in blob.split(b"\r\n"):
            if not line:
                break
            name, value = line.split(b":", 1)
            out.append((name, value.strip(b" \t")))
        return out

    assert len(run_python()) == count
    return f"parse_headers {count}", _time(run_python, 3), _time(
        lambda: mh.parse_headers(blob, max_headers=max_headers), 3
    )


def _python_offsets(blob: bytes):
    """Reference: the same field walk in Python, emitting the same offsets.

    Value offsets index an assembled value buffer, matching what the kernel
    writes into `vbuf`, so the two sides are directly comparable.
    """
    pos = 0
    out = []
    vbuf = bytearray()
    while True:
        eol = blob.find(b"\r\n", pos)
        if eol < 0:
            raise ValueError("no CRLF")
        if eol == pos:
            break
        colon = blob.find(b":", pos, eol)
        vstart = colon + 1
        while blob[vstart : vstart + 1] in (b" ", b"\t"):
            vstart += 1
        vend = eol
        while blob[vend - 1 : vend] in (b" ", b"\t"):
            vend -= 1
        out.append((pos, colon - pos, len(vbuf), vend - vstart))
        vbuf += blob[vstart:vend]
        pos = eol + 2
    return out


def bench_headers_kernel(count: int = 20000):
    """Field parsing, kernel side only: the same (offset, length) tuples a
    Python `find`-based walk produces, so neither side pays for building
    Python bytes objects."""
    max_headers = count + 8
    lines = [b"X-Header-%05d: value-%05d-with-some-padding" % (i, i) for i in range(count)]
    blob = b"\r\n".join(lines) + b"\r\n\r\n"
    offsets, _values, n, _folds = mh.parse_header_offsets(blob, max_headers=max_headers)
    assert n == count
    expect = _python_offsets(blob)
    assert [tuple(int(x) for x in row) for row in offsets] == expect, "offset mismatch"

    return f"parse_headers kernel {count}", _time(lambda: _python_offsets(blob), 3), _time(
        lambda: mh.parse_header_offsets(blob, max_headers=max_headers), 3
    )


def main():
    print(f"{'case':<32}{'reference':>13}{'mojo-aiohttp':>15}{'ratio':>10}")
    print("-" * 70)
    for fn in (
        bench_mask,
        bench_mask_pure_python,
        bench_chunked,
        bench_headers_kernel,
        bench_headers,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<32}{ref*1e3:>11.2f}ms{got*1e3:>13.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
