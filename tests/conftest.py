import asyncio
import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-aiohttp.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-aiohttp.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Drivers for the real aiohttp parsers.
#
# The parity tests below compare against these, not against a reimplementation
# written in the test file, so a plausible bug in the Mojo kernel cannot hide
# behind a reference that shares the same misunderstanding.
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _stream_reader(loop=None):
    from aiohttp.base_protocol import BaseProtocol
    from aiohttp.streams import StreamReader

    class _P(BaseProtocol):
        def pause_reading(self):
            pass

        def resume_reading(self, resume_parser=True):
            pass

    return StreamReader(_P(loop or asyncio.get_event_loop()), 2**18)


def real_decode_chunked(raw: bytes, feed: int = 0):
    """Feed `raw` through aiohttp's own chunked payload parser."""
    from aiohttp.http_parser import HeadersParser, HttpPayloadParser

    async def go():
        loop = asyncio.get_running_loop()
        reader = _stream_reader(loop)
        parser = HttpPayloadParser(
            reader, chunked=True, headers_parser=HeadersParser()
        )
        state, leftover = b"", b""
        if feed:
            for i in range(0, len(raw), feed):
                state, leftover = parser.feed_data(raw[i : i + feed])
        else:
            state, leftover = parser.feed_data(raw)
        body = b"".join(bytes(c) for c in reader._buffer)
        return body, state, leftover

    return _run(go())


def real_decode_length(raw: bytes, content_length: int, feed: int = 3):
    """Feed `raw` through aiohttp's PARSE_LENGTH payload parser in `feed` slices.

    Returns ``(consumed, trailing)`` byte strings accumulated over all feeds.
    """
    from aiohttp.http_parser import HeadersParser, HttpPayloadParser

    async def go():
        loop = asyncio.get_running_loop()
        reader = _stream_reader(loop)
        parser = HttpPayloadParser(
            reader, length=content_length, headers_parser=HeadersParser()
        )
        consumed = b""
        trailing = b""
        for i in range(0, len(raw), feed):
            state, leftover = parser.feed_data(raw[i : i + feed])
            # drain only what this feed added; the deque is not self-clearing
            while reader._buffer:
                consumed += bytes(reader._buffer.popleft())
            trailing += leftover
            if state.name == "PAYLOAD_COMPLETE":
                break
        return consumed, trailing

    return _run(go())


def real_parse_headers(lines, lax: bool = False):
    """Run aiohttp's own HeadersParser over a list of field lines."""
    from aiohttp.http_parser import HeadersParser

    return HeadersParser(lax=lax).parse_headers(list(lines))
