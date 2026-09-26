# mojo-aiohttp

`mojo-aiohttp` is a Mojo port of the wire-format byte loops inside
[aiohttp](https://docs.aiohttp.org/) 3.14, callable from Python.

**aiohttp is I/O plumbing.** It is an asyncio HTTP client and server: the
connection pool, the connector, the request/response state machines, the
`StreamReader` plumbing, TLS, proxying, the WebSocket session state machine and
the whole event loop are plumbing, and none of it is ported here. What is left
once the plumbing is stripped away is a set of tight, per-message byte loops
that aiohttp runs on every frame it sends or receives. Those loops are the
numeric core of the package, they are the only part worth a compiled inner
loop, and they are what this project implements.

The Python package is `mojo_aiohttp`, so it installs alongside the real
`aiohttp` and the parity tests compare the two directly.

## Covered subset

| area | upstream source | ported API |
| --- | --- | --- |
| WebSocket masking | `aiohttp/_websocket/helpers.py` (`_websocket_mask_python`) | `ws_mask` |
| WebSocket frame headers | `aiohttp/_websocket/reader_py.py` (`_feed_data`) | `ws_frame_header` |
| Chunked transfer decoding | `aiohttp/http_parser.py` (`HttpPayloadParser`, `PARSE_CHUNKED`) | `parse_chunked`, `hex_to_u64` |
| Field parsing | `aiohttp/http_parser.py` (`HeadersParser.parse_headers`) | `parse_headers`, `parse_header_offsets`, `header_fold_count`, `is_tchar` |
| Content-Length bookkeeping | `aiohttp/http_parser.py` (`HttpPayloadParser`, `PARSE_LENGTH`) | `content_length_plan` |
| `max_msg_size` guard | `aiohttp/_websocket/reader_py.py` | `projected_message_size` |

Every function is byte-exact against aiohttp. These are bit, index and
arithmetic operations, so the parity tests use exact equality rather than a
tolerance: aiohttp itself never approximates a mask byte or a chunk size.

**Not implemented, and not attempted:** the connector, cookie handling,
`ClientSession`/`web.Application`, multipart, websocket handshake
(`WSKey`, `Sec-WebSocket-Accept`), permessage-deflate, HTTP/2, connection
pooling, resolvers, timeouts, tracing, TLS. Those are I/O, control flow and
string handling. `aiohttp` also enforces a singleton-duplicate-header rule
(`SINGLETON_HEADERS`) that needs cross-header state; this port parses the field
block but does not implement that check, and says so rather than pretending.
Use the real `aiohttp` for all of it.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aiohttp.so
PYTHONPATH=python python -m pytest tests -q
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo = "==1.2.0.dev2026092605"`). Do not run `pixi install` in this tree; the
shared environment at `/nvme0n1-disk/mojo-toolchain` is the environment.

```python
import mojo_aiohttp as mh

buf = bytearray(b"a masked websocket payload")
mh.ws_mask(buf, b"\x01\x02\x03\x04")

mh.ws_frame_header(raw_frame)["payload_len"]
mh.parse_chunked(b"4\r\nWiki\r\n0\r\n\r\n")["sizes"]     # -> [4]
mh.parse_headers(b"Host: example.com\r\n\r\n")          # -> [(b'Host', b'example.com')]
```

## Performance

Best-of-seven wall clock, same process, every case checked against its
reference before timing. `reference` is the fastest fair pure-Python
formulation, not a strawman.

| case | reference | mojo-aiohttp | result |
| --- | ---: | ---: | ---: |
| `ws_mask` n=4194304 | 9.14 ms | 2.22 ms | 4.11x faster |
| `ws_mask` vs aiohttp's pure-Python fallback, n=65536 | 0.13 ms | 0.04 ms | 2.90x faster |
| `parse_chunked` 1 MiB body | 0.19 ms | 0.04 ms | 5.30x faster |
| `parse_headers` 20000 fields, kernel side | 47.17 ms | 8.11 ms | 5.81x faster |
| `parse_headers` 20000 fields, end to end | 20.13 ms | 61.39 ms | **0.33x, a slowdown** |

The last row is a real loss and worth explaining rather than hiding. The
compiled field parser is 5.8x faster than the Python walk, but
`parse_headers` has to hand the result back as a list of `(bytes, bytes)`
tuples, and building 20000 of those in Python costs more than the parse. The
Python reference reaches the same list with one `bytes.split` plus a `strip`
per line, which CPython does at C speed in a single pass. On a block of many
small fields the marshalling, not the parsing, is the cost, and the shim loses.
`parse_header_offsets` returns the kernel result unmarshalled and is the API to
reach for when the caller is going to walk the fields itself.

Reproduce with:

```bash
python bench/bench.py
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-aiohttp.so`.

The `python/mojo_aiohttp` layer owns every array. Buffers cross the C ABI as
64-bit addresses (`ctypes.c_int64`; `c_int` truncates and segfaults) and are
reconstructed in Mojo as `Pointer[UInt8, AnyOrigin[mut=True]]`, which keeps the
exported symbols non-parametric. Parsers report a numeric status rather than
raising across the ABI; the shim turns a non-zero status into `ParseStatus`.

Two details are worth calling out because getting them wrong is silent:

- An obs-folded field value is **discontinuous** in the input: RFC 9112 puts a
  CRLF between the segments. The kernel concatenates the segments into a
  caller-provided value buffer, because reading the value back as one slice of
  the field block would keep the CRLF. `test_obs_fold_drops_the_crlf_between_segments`
  exists for exactly that.
- `ah_content_length_plan` models the parser's leftover behaviour, which is
  asymmetric: a feed that completes the body hands its surplus to the next
  message, and every feed after that is pipelined data. The plan sums the
  surplus rather than keeping only the last one.

## Tests

149 parity tests, run against aiohttp itself:

- masking against `aiohttp._websocket.helpers._websocket_mask_python` and
  against the shipped Cython `websocket_mask`, at lengths 0..4096 including
  every tail length modulo 4;
- frame headers, including every strict prefix of a 14-byte header (an
  off-by-one in the "need more data" bound fails) and a round trip through
  `WebSocketReader` at 0, 5, 125, 126, 300 and 70000 byte payloads;
- field parsing against `HeadersParser`, in strict and lax mode, with 12
  malformed field lines that aiohttp rejects and one (HTAB in a value) that it
  accepts;
- chunked decoding against a real `HttpPayloadParser` fed one byte at a time
  as well as in one shot;
- `Content-Length` bookkeeping against a real `PARSE_LENGTH` parser.

## License

MIT
