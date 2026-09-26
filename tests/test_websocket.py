"""Parity against aiohttp's own WebSocket framing code.

The reference is `aiohttp._websocket.helpers` (masking) and
`aiohttp._websocket.reader_py.WebSocketReader` (frame headers), both of which
ship with aiohttp and are imported here directly.
"""

import numpy as np
import pytest

import mojo_aiohttp as mh
from conftest import _run, _stream_reader


def _aiohttp_mask(data: bytes, mask: bytes) -> bytes:
    from aiohttp._websocket.helpers import _websocket_mask_python

    buf = bytearray(data)
    _websocket_mask_python(mask, buf)
    return bytes(buf)


def _reader(message: bytes, max_msg_size: int = 0, compress: bool = False):
    """Feed `message` through aiohttp's own WebSocketReader, collect frames.

    Any protocol error aiohttp raises is re-raised, so the caller can assert
    on the rejection itself rather than on an empty result.
    """
    import asyncio

    from aiohttp._websocket.reader_py import WebSocketDataQueue, WebSocketReader

    async def go():
        loop = asyncio.get_running_loop()
        protocol = _stream_reader(loop)._protocol
        queue = WebSocketDataQueue(protocol, 2**16, loop=loop)
        reader = WebSocketReader(queue, max_msg_size, compress, False)
        reader.feed_data(message)
        # feed_eof clears the recorded exception, so surface it first
        failure = queue.exception()
        if failure is not None:
            raise failure
        reader.feed_eof()
        out = []
        while True:
            try:
                frame = await queue.read()
            except Exception as exc:  # EofStream means "no more frames"
                if type(exc).__name__ == "EofStream":
                    break
                raise
            out.append((frame.type, bytes(frame.data)))
        return out

    return _run(go())


def _frame(fin, opcode, payload, mask=None):
    import struct

    b0 = (0x80 if fin else 0) | opcode
    n = len(payload)
    if n < 126:
        head = bytes([b0, (0x80 if mask else 0) | n])
    elif n < 65536:
        head = bytes([b0, (0x80 if mask else 0) | 126]) + struct.pack("!H", n)
    else:
        head = bytes([b0, (0x80 if mask else 0) | 127]) + struct.pack("!Q", n)
    if mask is None:
        return head + payload
    masked = bytearray(payload)
    mh.ws_mask(masked, mask)
    return head + mask + bytes(masked)


# ---------------------------------------------------------------------------
# RFC 6455 5.3 masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 63, 64, 65, 4096]
)
def test_mask_matches_aiohttp_pure_python(n):
    rng = np.random.default_rng(n)
    data = bytes(rng.integers(0, 256, n, dtype=np.uint8).tolist())
    mask = bytes([0x37, 0xFA, 0x21, 0x3D])
    got = bytes(mh.ws_mask(bytearray(data), mask))
    assert got == _aiohttp_mask(data, mask)


def test_mask_matches_aiohttp_cython_extension():
    """aiohttp ships a Cython mask; both of its implementations and the Mojo
    kernel must agree, which makes this a parity test and not self-consistency."""
    from aiohttp._websocket.helpers import websocket_mask

    data = bytes(range(256)) * 3
    mask = b"\xde\xad\xbe\xef"
    ref = bytearray(data)
    websocket_mask(mask, ref)
    assert bytes(mh.ws_mask(bytearray(data), mask)) == bytes(ref)


def test_mask_is_its_own_inverse():
    data = bytes(range(200))
    once = bytes(mh.ws_mask(bytearray(data), b"\x11\x22\x33\x44"))
    twice = bytes(mh.ws_mask(bytearray(once), b"\x11\x22\x33\x44"))
    assert twice == data


def test_mask_tail_uses_the_right_mask_bytes():
    """A tail that reused mask[0] instead of mask[i % 4] would still be an
    involution, so pin the exact bytes."""
    data = b"\x00\x00\x00\x00\x00"
    got = bytes(mh.ws_mask(bytearray(data), b"\x01\x02\x03\x04"))
    assert got == b"\x01\x02\x03\x04\x01"


def test_mask_rejects_wrong_mask_length():
    with pytest.raises(ValueError):
        mh.ws_mask(bytearray(b"abc"), b"\x01\x02\x03")


def test_mask_rejects_a_read_only_buffer():
    with pytest.raises(ValueError):
        mh.ws_mask(b"abc", b"\x01\x02\x03\x04")


# ---------------------------------------------------------------------------
# RFC 6455 5.2 frame headers
# ---------------------------------------------------------------------------


def test_frame_header_short_payload():
    f = mh.ws_frame_header(_frame(True, 0x1, b"Hello"))
    assert f["status"] == 0
    assert f["fin"] is True
    assert f["opcode"] == 0x1
    assert f["has_mask"] is False
    assert f["len_flag"] == 5
    assert f["payload_len"] == 5
    assert f["header_len"] == 2
    assert f["payload_start"] == 2
    assert f["mask_offset"] == -1


def test_frame_header_16_bit_extended_length():
    f = mh.ws_frame_header(_frame(True, 0x2, b"x" * 300))
    assert f["len_flag"] == 126
    assert f["payload_len"] == 300
    assert f["header_len"] == 4
    assert f["payload_start"] == 4


def test_frame_header_64_bit_extended_length():
    f = mh.ws_frame_header(_frame(True, 0x2, b"x" * 70000))
    assert f["len_flag"] == 127
    assert f["payload_len"] == 70000
    assert f["header_len"] == 10
    assert f["payload_start"] == 10


def test_frame_header_reserved_bits():
    raw = bytearray(_frame(True, 0x1, b"abc"))
    raw[0] |= 0x70  # set rsv1..rsv3
    f = mh.ws_frame_header(raw)
    assert (f["rsv1"], f["rsv2"], f["rsv3"]) == (True, True, True)


def test_frame_header_unset_fin():
    f = mh.ws_frame_header(_frame(False, 0x1, b"frag"))
    assert f["fin"] is False
    assert f["payload_len"] == 4


def test_frame_header_mask_offsets():
    raw = _frame(True, 0x1, b"payload!", b"\x01\x02\x03\x04")
    f = mh.ws_frame_header(raw)
    assert f["has_mask"] is True
    assert f["header_len"] == 6
    assert f["mask_offset"] == 2
    assert f["payload_start"] == 6
    assert raw[2:6] == b"\x01\x02\x03\x04"


def test_frame_header_masked_64_bit_length():
    raw = _frame(True, 0x2, b"y" * 70000, b"\x09\x08\x07\x06")
    f = mh.ws_frame_header(raw)
    assert f["len_flag"] == 127
    assert f["payload_len"] == 70000
    assert f["mask_offset"] == 10
    assert f["header_len"] == 14
    assert f["payload_start"] == 14


def test_frame_header_control_frame():
    f = mh.ws_frame_header(_frame(True, 0x9, b"ping"))
    assert f["opcode"] == 0x9
    assert f["payload_len"] == 4
    assert f["fin"] is True


@pytest.mark.parametrize("cut", list(range(0, 15)))
def test_frame_header_incomplete_buffers_report_need_more(cut):
    """Every strict prefix of a 14-byte header must report 'need more' until
    the last byte lands. An off-by-one in the bounds check fails here."""
    raw = _frame(True, 0x2, b"y" * 70000, b"\x09\x08\x07\x06")
    f = mh.ws_frame_header(raw[:cut])
    if cut < 14:
        assert f["status"] == 1, f"cut={cut}"
        assert f["return"] == -1
    else:
        assert f["status"] == 0
        assert f["return"] == 14


def test_frame_header_empty_buffer_reports_need_more():
    assert mh.ws_frame_header(b"")["status"] == 1


# ---------------------------------------------------------------------------
# End to end through aiohttp's own frame reader
# ---------------------------------------------------------------------------


def test_masked_frame_round_trips_through_aiohttp_reader():
    """If the mask were applied with the wrong stride, or the frame header
    disagreed with aiohttp's by a byte, the reassembled message differs."""
    payload = bytes(range(256)) * 2
    frames = _reader(_frame(True, 0x1, payload, b"\x2b\x7e\x11\x9c"))
    assert len(frames) == 1
    assert frames[0][1] == payload


def test_unmasked_large_frame_round_trips_through_aiohttp_reader():
    payload = bytes((i * 7) % 251 for i in range(70000))
    frames = _reader(_frame(True, 0x2, payload))
    assert frames[0][1] == payload


def test_fragments_reassemble_through_aiohttp_reader():
    mask = b"\x01\x02\x03\x04"
    a, b = b"first half ", b"second half"
    raw = _frame(False, 0x1, a, mask) + _frame(True, 0x0, b, mask)
    frames = _reader(raw)
    assert frames[0][1] == a + b


@pytest.mark.parametrize("n", [0, 5, 125, 126, 300, 70000])
def test_our_header_agrees_with_aiohttp_reader_on_payload_offsets(n):
    """The reader must consume exactly the bytes our header says it will."""
    mask = b"\xaa\xbb\xcc\xdd"
    raw = _frame(True, 0x1, b"m" * n, mask)
    f = mh.ws_frame_header(raw)
    frames = _reader(raw)
    assert len(frames[0][1]) == n == f["payload_len"]
    assert raw[f["payload_start"] :] == raw[f["header_len"] :]
    assert raw[f["mask_offset"] : f["payload_start"]] == mask


# ---------------------------------------------------------------------------
# max_msg_size guard arithmetic
# ---------------------------------------------------------------------------


def test_projected_message_size_matches_the_reader_limit():
    from aiohttp._websocket.models import WebSocketError

    # 60 + 40 == 100, and the reader's guard is `projected >= max_msg_size`
    assert mh.projected_message_size(100, 60, [40]) == 100
    with pytest.raises(WebSocketError):
        _reader(_frame(True, 0x1, b"x" * 100), max_msg_size=100)


def test_projected_message_size_under_the_limit_is_accepted():
    assert mh.projected_message_size(100, 60, [39]) == 99
    frames = _reader(_frame(True, 0x1, b"x" * 99), max_msg_size=100)
    assert len(frames[0][1]) == 99
