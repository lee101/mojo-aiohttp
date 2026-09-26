"""mojo-aiohttp: the wire-format byte loops of aiohttp, in Mojo.

aiohttp itself is an asyncio HTTP/WebSocket client and server. The connection
pool, the request/response state machines, the streams plumbing and the
WebSocket session management are not ported and are not part of this package's
API. What is ported is the set of byte-level loops aiohttp runs on every single
message, taken from `aiohttp/_websocket/helpers.py`,
`aiohttp/_websocket/reader_py.py` and `aiohttp/http_parser.py`.

Installable alongside the real `aiohttp`, which the parity tests compare
against directly.
"""

from ._lib import (
    CHUNKED_BODY_END,
    CHUNKED_DECODED,
    CHUNKED_N_CHUNKS,
    CHUNKED_STATUS,
    CHUNKED_TRAILER_OFF,
    DEFAULT_MAX_FIELD_SIZE,
    DEFAULT_MAX_HEADERS,
    DEFAULT_MAX_LINE_SIZE,
    DEFAULT_MAX_TRAILERS,
    ParseStatus,
    content_length_plan,
    header_fold_count,
    hex_to_u64,
    is_tchar,
    parse_chunked,
    parse_header_offsets,
    parse_headers,
    projected_message_size,
    ws_frame_header,
    ws_mask,
)

__all__ = [
    "content_length_plan",
    "header_fold_count",
    "hex_to_u64",
    "is_tchar",
    "parse_chunked",
    "parse_header_offsets",
    "parse_headers",
    "projected_message_size",
    "ws_frame_header",
    "ws_mask",
    "ParseStatus",
]
__version__ = "0.1.0"
