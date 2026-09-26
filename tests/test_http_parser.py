"""Parity against aiohttp's own HTTP message parsers.

`aiohttp.http_parser` ships a pure-Python `HeadersParser` and
`HttpPayloadParser`; both are driven here directly so the Mojo kernels are
compared with the implementation they replace, not with a transcription of it.
"""

import pytest

import mojo_aiohttp as mh
from conftest import real_decode_chunked, real_decode_length, real_parse_headers

CRLF = b"\r\n"


def block(*lines: bytes) -> bytes:
    """Field block as aiohttp hands it over: lines plus the terminating blank."""
    return CRLF.join(lines) + CRLF + CRLF


def lines_of(blob: bytes):
    """The `list[bytes]` aiohttp's HeadersParser consumes for a block."""
    assert blob.endswith(CRLF + CRLF)
    return blob[: -len(CRLF)].split(CRLF)


def as_pairs(headers):
    return [(k.encode(), v.encode()) for k, v in headers.items()]


# ---------------------------------------------------------------------------
# RFC 9110 field parsing
# ---------------------------------------------------------------------------

BLOCKS = [
    block(b"Content-Type: text/html"),
    block(b"Content-Type:text/html", b"Content-Length: 42"),
    block(b"X-Trace:   spaced   ", b"X-Other:\tvalue\t"),
    block(b"Server: nginx/1.25.3", b"Date: Mon, 26 Sep 2026 10:00:00 GMT"),
    block(
        b"Host: example.com",
        b"User-Agent: mojo-aiohttp/0.1",
        b"Accept: */*",
        b"X-Tricky: !#$%&'*+-.^_`|~",
    ),
    block(b"Set-Cookie: a=1", b"Set-Cookie: b=2"),
    block(b"Empty-Value:"),
    block(b"Long: " + b"y" * 4000),
    block(b"X-Multi: one", b"X-Multi: two", b"X-Multi: three"),
]


@pytest.mark.parametrize("blob", BLOCKS)
def test_parse_headers_matches_aiohttp(blob):
    mine = mh.parse_headers(blob)
    theirs, _raw = real_parse_headers(lines_of(blob))
    assert mine == as_pairs(theirs)
    assert len(mine) == len(theirs)


def test_parse_headers_strips_optional_whitespace_on_both_sides():
    assert mh.parse_headers(block(b"X-A: \t v \t ")) == [(b"X-A", b"v")]


def test_parse_headers_lax_obs_fold():
    blob = block(b"X-A: a", b"  cont", b"\tmore", b"X-B: b")
    mine = mh.parse_headers(blob, lax=True)
    theirs, _ = real_parse_headers(lines_of(blob), lax=True)
    assert mine == as_pairs(theirs)
    assert mine[0] == (b"X-A", b"a  cont\tmore")


def test_obs_fold_drops_the_crlf_between_segments():
    """A kernel that read the value as one slice of the block would keep the
    CRLF that ends the first line."""
    blob = block(b"X-A: first", b"  second", b"")
    name, value = mh.parse_headers(blob, lax=True)[0]
    assert b"\r" not in value and b"\n" not in value
    assert value == b"first  second"


def test_obs_fold_strips_each_segments_own_trailing_ows():
    blob = block(b"X-A: a", b"  b   ", b"")
    assert mh.parse_headers(blob, lax=True)[0] == (b"X-A", b"a  b")


def test_strict_mode_does_not_fold():
    """obs-fold is deprecated; the strict parser rejects it. A kernel that
    folded unconditionally would wrongly accept it."""
    blob = block(b"X-A: a", b"  cont", b"")
    with pytest.raises(mh.ParseStatus):
        mh.parse_headers(blob, lax=False)
    with pytest.raises(Exception):
        real_parse_headers(lines_of(blob), lax=False)


def test_header_fold_count_matches_the_number_of_continuations():
    blob = block(b"X-A: a", b"  one", b"\ttwo", b"X-B: b")
    assert mh.header_fold_count(blob) == 2
    assert mh.header_fold_count(block(b"X-A: a")) == 0


@pytest.mark.parametrize(
    "bad",
    [
        b"X-A v",  # no colon
        b": value",  # empty name
        b" X-A: v",  # leading space in the name
        b"X-A : v",  # trailing space in the name
        b"\tX-A: v",
        b"X-A\t: v",
        b"X A: v",  # space is not a tchar
        b"X(A): v",  # parentheses are not tchars
        b"X-A: v\x00",  # NUL in the value
        b"X-A: v\x1f",  # control char in the value
        b"X-A: v\x7f",  # DEL in the value
        b"X-\xc3\xa9: v",  # non-ASCII name
    ],
)
def test_parse_headers_rejects_what_aiohttp_rejects(bad):
    blob = block(bad)
    with pytest.raises(mh.ParseStatus):
        mh.parse_headers(blob)
    with pytest.raises(Exception):
        real_parse_headers(lines_of(blob))


def test_parse_headers_accepts_a_tab_inside_a_value():
    """HTAB is legal inside a field value; a kernel that banned every control
    character would wrongly reject this."""
    blob = block(b"X-A: a\tb")
    mine = mh.parse_headers(blob)
    theirs, _ = real_parse_headers(lines_of(blob))
    assert mine == [(b"X-A", b"a\tb")]
    assert mine == as_pairs(theirs)


def test_parse_headers_rejects_an_over_long_folded_value():
    blob = block(b"X-A: a", b"  " + b"x" * 40)
    with pytest.raises(mh.ParseStatus) as exc:
        mh.parse_headers(blob, lax=True, max_field_size=16)
    assert exc.value.status == -2


def test_tchar_table_agrees_with_aiohttp():
    from aiohttp.http_parser import TOKENRE

    for c in range(256):
        assert mh.is_tchar(c) is bool(TOKENRE.fullmatch(chr(c))), f"U+{c:04X}"


# ---------------------------------------------------------------------------
# RFC 9112 chunked transfer decoding
# ---------------------------------------------------------------------------

CHUNKED_BODIES = [
    b"4\r\nWiki\r\n6\r\npedia \r\nE\r\n in\r\n\r\nchunks.\r\n0\r\n\r\n",
    b"3\r\nabc\r\n0\r\n\r\n",
    b"0\r\n\r\n",
    b"1\r\na\r\n1\r\nb\r\n1\r\nc\r\n0\r\n\r\n",
    b"5;ext=1\r\nhello\r\n0\r\n\r\n",
    b"4\r\nWiki\r\n0\r\nX-Trailer: 1\r\nX-Other: 2\r\n\r\n",
    b"4\r\nWiki\r\n0\r\n\r\nLEFTOVER",
    b"20\r\n" + b"z" * 0x20 + b"\r\n0\r\n\r\n",
    b"0\r\nX-Empty: \r\n\r\n",
    b"a\r\n" + b"q" * 0x0A + b"\r\na\r\n" + b"q" * 0x0A + b"\r\n0\r\n\r\n",
    b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
]


@pytest.mark.parametrize("body", CHUNKED_BODIES)
def test_parse_chunked_sizes_sum_to_the_decoded_length(body):
    decoded, _state, _leftover = real_decode_chunked(body)
    result = mh.parse_chunked(body)
    assert result["decoded_length"] == len(decoded)
    assert int(result["sizes"].sum()) == len(decoded)


@pytest.mark.parametrize("body", CHUNKED_BODIES)
def test_parse_chunked_reassembles_the_same_payload(body):
    decoded, _state, _leftover = real_decode_chunked(body)
    sizes = mh.parse_chunked(body)["sizes"]
    pos = 0
    rebuilt = bytearray()
    for size in sizes:
        nl = body.index(CRLF, pos)
        pos = nl + 2
        rebuilt += body[pos : pos + int(size)]
        pos += int(size) + 2
    assert bytes(rebuilt) == decoded


@pytest.mark.parametrize("body", CHUNKED_BODIES)
def test_parse_chunked_survives_byte_at_a_time_feeding(body):
    """aiohttp fed one byte at a time must produce the same payload as one
    shot; the kernel walks the whole body, so its chunk boundaries must agree
    with a parser that never sees more than one byte."""
    decoded, _state, _leftover = real_decode_chunked(body, feed=1)
    result = mh.parse_chunked(body)
    assert int(result["sizes"].sum()) == len(decoded)


def test_parse_chunked_records_the_chunk_count():
    body = b"1\r\na\r\n1\r\nb\r\n1\r\nc\r\n0\r\n\r\n"
    assert mh.parse_chunked(body)["sizes"].tolist() == [1, 1, 1]


def test_parse_chunked_body_end_matches_aiohttp_leftover():
    body = b"4\r\nWiki\r\n0\r\n\r\nLEFTOVER"
    _decoded, _state, leftover = real_decode_chunked(body)
    result = mh.parse_chunked(body)
    assert body[result["body_end"] :] == leftover == b"LEFTOVER"


@pytest.mark.parametrize(
    "body",
    [
        b"zz\r\nabc\r\n0\r\n\r\n",  # size line is not hex
        b"4\r\nWikiXX0\r\n\r\n",  # CRLF missing after the chunk data
        b"3\r\nabc\r\n0\r\n",  # truncated terminator
        b"\r\n0\r\n\r\n",  # empty size line
    ],
)
def test_parse_chunked_rejects_bad_framing(body):
    with pytest.raises(mh.ParseStatus):
        mh.parse_chunked(body)


def test_parse_chunked_enforces_max_line_size():
    body = b"4\r\nWiki\r\n" + b"f" * 100 + b"\r\n0\r\n\r\n"
    with pytest.raises(mh.ParseStatus) as exc:
        mh.parse_chunked(body, max_line_size=16)
    assert exc.value.status == -2


def test_parse_chunked_rejects_too_many_trailers():
    body = b"0\r\n" + b"".join(b"X-%d: 1\r\n" % i for i in range(6)) + b"\r\n"
    with pytest.raises(mh.ParseStatus) as exc:
        mh.parse_chunked(body, max_trailers=3)
    assert exc.value.status == -4


def test_parse_chunked_locates_the_trailer_section():
    body = b"4\r\nWiki\r\n0\r\nX-T: 1\r\n\r\n"
    result = mh.parse_chunked(body)
    assert body[result["trailer_offset"] :] == b"X-T: 1\r\n\r\n"


# ---------------------------------------------------------------------------
# Hex chunk-size decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digits,expect",
    [
        (b"0", 0),
        (b"4", 4),
        (b"ff", 255),
        (b"FF", 255),
        (b"1f3a", 7994),
        (b"00000010", 16),
    ],
)
def test_hex_to_u64_values(digits, expect):
    assert mh.hex_to_u64(digits, 0, len(digits)) == expect


@pytest.mark.parametrize("bad", [b"g", b"1g", b"0x10", b" 4", b"4 ", b"-1"])
def test_hex_to_u64_rejects_non_hex(bad):
    assert mh.hex_to_u64(bad, 0, len(bad)) == -1


def test_hex_to_u64_reads_from_an_offset():
    assert mh.hex_to_u64(b"junk1f3amore", 4, 4) == 7994


# ---------------------------------------------------------------------------
# Content-Length payload bookkeeping
# ---------------------------------------------------------------------------


def test_content_length_plan_matches_aiohttp():
    """Replay the same feed sequence through aiohttp's PARSE_LENGTH parser and
    compare what each feed contributed. aiohttp refuses to be fed again once
    the body is complete, so the comparison stops there too."""
    consumed, trailing = real_decode_length(b"abcdefghij", 7, feed=3)
    got, tail, remaining = mh.content_length_plan(7, [3, 3, 3])
    assert consumed == b"abcdefg"
    assert got.tolist() == [3, 3, 1]
    assert sum(got.tolist()) == len(consumed) == 7
    assert tail == len(trailing) == 2
    assert remaining == 0


def test_content_length_plan_under_feeds_multiple_times():
    # 3 + 3 fills the 7-byte body; the remaining 2 bytes of the third feed and
    # all 3 of the fourth are carried over to the next message.
    consumed, tail, remaining = mh.content_length_plan(7, [3, 3, 3, 3])
    assert consumed.tolist() == [3, 3, 1, 0]
    assert tail == 5
    assert remaining == 0


def test_content_length_plan_reports_the_shortfall():
    consumed, tail, remaining = mh.content_length_plan(10, [4, 4])
    assert consumed.tolist() == [4, 4]
    assert tail == 0
    assert remaining == 2


def test_content_length_plan_over_feeds_pass_the_tail_through():
    consumed, tail, remaining = mh.content_length_plan(4, [10])
    assert consumed.tolist() == [4]
    assert tail == 6
    assert remaining == 0


def test_content_length_plan_leftover_matches_aiohttp():
    """The 4-byte body completes during the second 3-byte feed, so the parser
    hands the remaining 2 bytes of that feed back as the next message."""
    consumed, trailing = real_decode_length(b"abcdefghij", 4, feed=3)
    assert consumed == b"abcd"
    assert trailing == b"ef"
    _got, tail, _remaining = mh.content_length_plan(4, [3, 3])
    assert tail == len(trailing) == 2


def test_content_length_plan_accumulates_pipelined_feeds():
    """Feeds past the end of the body are pipelined data; the plan sums them."""
    got, tail, remaining = mh.content_length_plan(4, [3, 3, 3, 3])
    assert got.tolist() == [3, 1, 0, 0]
    assert tail == 2 + 3 + 3
    assert remaining == 0


def test_content_length_plan_empty_input():
    consumed, tail, remaining = mh.content_length_plan(4, [])
    assert consumed.tolist() == []
    assert tail == 0
    assert remaining == 4
