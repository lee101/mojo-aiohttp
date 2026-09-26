"""Compiled byte-level kernels for the aiohttp wire-format subset.

aiohttp is an asyncio HTTP/WebSocket stack: the I/O, the connection pool, the
event scheduling and the WebSocket session management are plumbing. What is
left once the plumbing is stripped out is a set of tight byte loops that run on
every message -- RFC 6455 masking, RFC 6455 frame header decoding, RFC 9112
chunked transfer decoding, RFC 9110 field parsing and content-length
bookkeeping. Those loops are what this compilation unit implements.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

Status codes the parsers write to their out array (the caller never guesses):

    0  complete / parsed
    1  need more input
   -1  malformed input for this state
   -2  line exceeds the configured max line size
   -3  framing mismatch (expected CRLF)
   -4  too many headers / trailers
   -5  the caller's scratch buffer is too small
"""

comptime WS_OUT_SLOTS = 12


def uptr(addr: Int) -> Pointer[UInt8, AnyOrigin[mut=True]]:
    return Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=addr)


def lptr(addr: Int) -> Pointer[Int64, AnyOrigin[mut=True]]:
    return Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# RFC 6455 section 5.3 -- client-to-server frame masking
# ---------------------------------------------------------------------------


@export("ah_ws_mask")
def ah_ws_mask(mask_addr: Int, data_addr: Int, n: Int) abi("C"):
    """XOR `data` in place with the 4-byte WebSocket masking key.

    RFC 6455 5.3: `transformed[i] = original[i] XOR mask[i MOD 4]`. The body is
    unrolled four bytes at a time so the mask index never has to be computed;
    the scalar tail handles `n % 4`.
    """
    if n <= 0:
        return
    var m = uptr(mask_addr)
    var d = uptr(data_addr)
    var m0 = m[unsafe_offset=0]
    var m1 = m[unsafe_offset=1]
    var m2 = m[unsafe_offset=2]
    var m3 = m[unsafe_offset=3]
    var i = 0
    while i + 4 <= n:
        d[unsafe_offset=i] = d[unsafe_offset=i] ^ m0
        d[unsafe_offset=i + 1] = d[unsafe_offset=i + 1] ^ m1
        d[unsafe_offset=i + 2] = d[unsafe_offset=i + 2] ^ m2
        d[unsafe_offset=i + 3] = d[unsafe_offset=i + 3] ^ m3
        i += 4
    while i < n:
        d[unsafe_offset=i] = d[unsafe_offset=i] ^ m[unsafe_offset=i % 4]
        i += 1


# ---------------------------------------------------------------------------
# RFC 6455 section 5.2 -- frame header decoding
# ---------------------------------------------------------------------------

# out slots: 0 fin, 1 rsv1, 2 rsv2, 3 rsv3, 4 opcode, 5 has_mask, 6 len_flag,
#            7 payload_len, 8 header_len (mask included), 9 status,
#           10 mask_offset (-1 when absent), 11 payload_start


@export("ah_ws_frame_header")
def ah_ws_frame_header(
    buf_addr: Int, blen: Int, start: Int, out_addr: Int
) abi("C") -> Int:
    """Decode one WebSocket frame header at byte `start`.

    Returns the offset of the first payload byte, or -1 when the buffer does
    not yet hold a complete header (status slot 9 is then 1).
    """
    var p = uptr(buf_addr)
    var o = lptr(out_addr)
    for k in range(WS_OUT_SLOTS):
        o[unsafe_offset=k] = 0
    o[unsafe_offset=10] = -1
    if start < 0 or start + 2 > blen:
        o[unsafe_offset=9] = 1
        return -1
    var fb = Int64(p[unsafe_offset=start])
    var sb = Int64(p[unsafe_offset=start + 1])
    o[unsafe_offset=0] = (fb >> 7) & 1
    o[unsafe_offset=1] = (fb >> 6) & 1
    o[unsafe_offset=2] = (fb >> 5) & 1
    o[unsafe_offset=3] = (fb >> 4) & 1
    o[unsafe_offset=4] = fb & 0xF
    o[unsafe_offset=5] = (sb >> 7) & 1
    o[unsafe_offset=6] = sb & 0x7F
    var pos = start + 2
    var plen = o[unsafe_offset=6]
    var hlen = 2
    if o[unsafe_offset=6] == 126:
        if pos + 2 > blen:
            o[unsafe_offset=9] = 1
            return -1
        plen = (Int64(p[unsafe_offset=pos]) << 8) | Int64(p[unsafe_offset=pos + 1])
        pos += 2
        hlen += 2
    elif o[unsafe_offset=6] > 126:
        if pos + 8 > blen:
            o[unsafe_offset=9] = 1
            return -1
        var wide = Int64(0)
        for k in range(8):
            wide = (wide << 8) | Int64(p[unsafe_offset=pos + k])
        plen = wide
        pos += 8
        hlen += 8
    o[unsafe_offset=7] = plen
    if o[unsafe_offset=5] == 1:
        if pos + 4 > blen:
            o[unsafe_offset=9] = 1
            return -1
        o[unsafe_offset=10] = Int64(pos)
        pos += 4
        hlen += 4
    o[unsafe_offset=8] = Int64(hlen)
    o[unsafe_offset=9] = 0
    o[unsafe_offset=11] = Int64(pos)
    return pos


# ---------------------------------------------------------------------------
# Hexadecimal chunk-size decoding (RFC 9112 section 7.1)
# ---------------------------------------------------------------------------


@export("ah_hex_to_u64")
def ah_hex_to_u64(buf_addr: Int, start: Int, n: Int) abi("C") -> Int64:
    """Parse `n` hex digits at `start` as an unsigned integer.

    Returns -1 if any digit is not `[0-9a-fA-F]`, matching the `HEXDIGITS`
    guard aiohttp applies before calling `int(size_b, 16)`.
    """
    var p = uptr(buf_addr)
    if n <= 0:
        return -1
    var v = Int64(0)
    for k in range(n):
        var c = Int64(p[unsafe_offset=start + k])
        var d = Int64(-1)
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        if d < 0:
            return -1
        v = v * 16 + d
    return v


# ---------------------------------------------------------------------------
# RFC 9112 section 7.1 -- chunked transfer decoding
# ---------------------------------------------------------------------------

# out slots after the max_chunks size slots:
#   [m]     n_chunks
#   [m + 1] decoded length
#   [m + 2] trailer section start offset
#   [m + 3] status
#   [m + 4] end of chunked body (first unconsumed byte)


@export("ah_chunked_parse")
def ah_chunked_parse(
    buf_addr: Int,
    blen: Int,
    out_addr: Int,
    max_chunks: Int,
    max_line_size: Int,
    max_field_size: Int,
    max_trailers: Int,
) abi("C") -> Int:
    """Walk a complete chunked body, recording every chunk size.

    Returns the decoded payload length, or -1 on error. `out` must hold
    `max_chunks + 5` Int64 slots.
    """
    var p = uptr(buf_addr)
    var o = lptr(out_addr)
    var base = Int64(max_chunks)
    o[unsafe_offset=base + 0] = 0
    o[unsafe_offset=base + 1] = 0
    o[unsafe_offset=base + 2] = 0
    o[unsafe_offset=base + 3] = 1
    o[unsafe_offset=base + 4] = 0
    var pos = 0
    var total = Int64(0)
    var nch = 0
    while True:
        # --- chunk-size line -------------------------------------------------
        var eol = -1
        var i = pos
        while i < blen:
            if p[unsafe_offset=i] == 13 and i + 1 < blen and p[unsafe_offset=i + 1] == 10:
                eol = i
                break
            i += 1
        if eol < 0:
            o[unsafe_offset=base + 3] = 1
            return -1
        if eol - pos > max_line_size:
            o[unsafe_offset=base + 3] = -2
            return -1
        # strip chunk-extensions at the first ';'
        var stop = eol
        var k = pos
        while k < eol:
            if p[unsafe_offset=k] == 59:
                stop = k
                break
            k += 1
        while stop > pos and (
            p[unsafe_offset=stop - 1] == 32 or p[unsafe_offset=stop - 1] == 9
        ):
            stop -= 1
        var size = ah_hex_to_u64(buf_addr, pos, stop - pos)
        if size < 0:
            o[unsafe_offset=base + 3] = -1
            return -1
        pos = eol + 2
        if size == 0:
            break
        if nch >= max_chunks:
            o[unsafe_offset=base + 3] = -4
            return -1
        if pos + Int(size) > blen:
            o[unsafe_offset=base + 3] = 1
            return -1
        o[unsafe_offset=Int64(nch)] = size
        nch += 1
        total += size
        pos += Int(size)
        # --- CRLF that terminates the chunk data ----------------------------
        if pos + 2 > blen:
            o[unsafe_offset=base + 3] = 1
            return -1
        if p[unsafe_offset=pos] != 13 or p[unsafe_offset=pos + 1] != 10:
            o[unsafe_offset=base + 3] = -3
            return -1
        pos += 2
    # --- trailer section ---------------------------------------------------
    o[unsafe_offset=base + 0] = Int64(nch)
    o[unsafe_offset=base + 1] = total
    o[unsafe_offset=base + 2] = Int64(pos)
    var ntrail = 0
    while True:
        var eol = -1
        var i = pos
        while i < blen:
            if p[unsafe_offset=i] == 13 and i + 1 < blen and p[unsafe_offset=i + 1] == 10:
                eol = i
                break
            i += 1
        if eol < 0:
            o[unsafe_offset=base + 3] = 1
            return -1
        if eol - pos > max_field_size:
            o[unsafe_offset=base + 3] = -2
            return -1
        var line_len = eol - pos
        pos = eol + 2
        if line_len == 0:
            break
        ntrail += 1
        if ntrail > max_trailers:
            o[unsafe_offset=base + 3] = -4
            return -1
    o[unsafe_offset=base + 3] = 0
    o[unsafe_offset=base + 4] = Int64(pos)
    return Int(total)


# ---------------------------------------------------------------------------
# RFC 9110 section 5 -- field parsing (aiohttp HeadersParser rules)
# ---------------------------------------------------------------------------


@export("ah_is_tchar")
def ah_is_tchar(c: Int) abi("C") -> Int:
    """RFC 9110 tchar: DIGIT / ALPHA / ``!#$%&'*+-.^_`|~``."""
    if c >= 48 and c <= 57:
        return 1
    if c >= 65 and c <= 90:
        return 1
    if c >= 97 and c <= 122:
        return 1
    if c == 33 or c == 35 or c == 36 or c == 37 or c == 38 or c == 39:
        return 1
    if c == 42 or c == 43 or c == 45 or c == 46:
        return 1
    if c == 94 or c == 95 or c == 96 or c == 124 or c == 126:
        return 1
    return 0


@export("ah_headers_parse")
def ah_headers_parse(
    buf_addr: Int,
    blen: Int,
    out_addr: Int,
    max_headers: Int,
    lax: Int,
    max_field_size: Int,
    vbuf_addr: Int,
    vbuf_cap: Int,
) abi("C") -> Int:
    """Parse an RFC 9110 field block terminated by an empty line.

    `buf` is the raw field block, CRLF separated, with the terminating empty
    line present -- exactly the `list[bytes]` that aiohttp's `HeadersParser`
    receives.

    `out` holds four Int64 slots per header (name offset into `buf`, name
    length, value offset into `vbuf`, value length) followed by four summary
    slots: count, status, obs-fold count, value bytes written.

    `vbuf` receives the assembled field values. An obs-folded value is
    discontinuous in `buf` -- its segments are separated by CRLF -- so it is
    concatenated into `vbuf` instead of being read back as one slice.

    Returns the header count, or -1 on error.
    """
    var p = uptr(buf_addr)
    var o = lptr(out_addr)
    var v = uptr(vbuf_addr)
    var stride = Int64(max_headers) * 4
    o[unsafe_offset=stride + 0] = 0
    o[unsafe_offset=stride + 1] = -1
    o[unsafe_offset=stride + 2] = 0
    o[unsafe_offset=stride + 3] = 0
    var pos = 0
    var count = 0
    var folds = 0
    var vused = 0
    while True:
        if pos >= blen:
            o[unsafe_offset=stride + 1] = 1
            return -1
        # read one CRLF-terminated line
        var eol = -1
        var i = pos
        while i < blen:
            if p[unsafe_offset=i] == 13 and i + 1 < blen and p[unsafe_offset=i + 1] == 10:
                eol = i
                break
            i += 1
        if eol < 0:
            o[unsafe_offset=stride + 1] = 1
            return -1
        var line_len = eol - pos
        var line_start = pos
        pos = eol + 2
        if line_len == 0:
            o[unsafe_offset=stride + 1] = 0
            break
        if count >= max_headers:
            o[unsafe_offset=stride + 1] = -4
            return -1
        # split on the first ':'
        var colon = -1
        var k = line_start
        while k < eol:
            if p[unsafe_offset=k] == 58:
                colon = k
                break
            k += 1
        if colon < 0:
            o[unsafe_offset=stride + 1] = -1
            return -1
        var name_off = line_start
        var name_len = colon - line_start
        if name_len == 0:
            o[unsafe_offset=stride + 1] = -1
            return -1
        # RFC 9112 5.1: no whitespace is allowed around the field name
        var c0 = Int64(p[unsafe_offset=name_off])
        var cN = Int64(p[unsafe_offset=colon - 1])
        if c0 == 32 or c0 == 9 or cN == 32 or cN == 9:
            o[unsafe_offset=stride + 1] = -1
            return -1
        # the name must be 1*tchar. The test is inlined rather than calling
        # ah_is_tchar: a call per byte costs more than the comparisons.
        for k in range(name_off, colon):
            var tc = Int64(p[unsafe_offset=k])
            var ok = (
                (tc >= 48 and tc <= 57)
                or (tc >= 65 and tc <= 90)
                or (tc >= 97 and tc <= 122)
                or tc == 33 or tc == 35 or tc == 36 or tc == 37
                or tc == 38 or tc == 39 or tc == 42 or tc == 43
                or tc == 45 or tc == 46 or tc == 94 or tc == 95
                or tc == 96 or tc == 124 or tc == 126
            )
            if not ok:
                o[unsafe_offset=stride + 1] = -1
                return -1
        # first value segment: everything after the colon, leading OWS stripped
        var vstart = colon + 1
        while vstart < eol:
            var cv = Int64(p[unsafe_offset=vstart])
            if cv != 32 and cv != 9:
                break
            vstart += 1
        var accum = eol - vstart
        var val_off = vused
        if vused + (eol - vstart) > vbuf_cap:
            o[unsafe_offset=stride + 1] = -5
            return -1
        for k in range(vstart, eol):
            v[unsafe_offset=vused] = p[unsafe_offset=k]
            vused += 1
        # obs-fold continuation lines (deprecated, lax mode only)
        while lax == 1 and pos < blen:
            var fc = Int64(p[unsafe_offset=pos])
            if fc != 32 and fc != 9:
                break
            var neol = -1
            var j = pos
            while j < blen:
                if p[unsafe_offset=j] == 13 and j + 1 < blen and p[unsafe_offset=j + 1] == 10:
                    neol = j
                    break
                j += 1
            if neol < 0:
                o[unsafe_offset=stride + 1] = 1
                return -1
            accum += neol - pos
            if accum > max_field_size:
                o[unsafe_offset=stride + 1] = -2
                return -1
            var fend = neol
            while fend > pos:
                var cv2 = Int64(p[unsafe_offset=fend - 1])
                if cv2 != 32 and cv2 != 9:
                    break
                fend -= 1
            if vused + (fend - pos) > vbuf_cap:
                o[unsafe_offset=stride + 1] = -5
                return -1
            for k in range(pos, fend):
                v[unsafe_offset=vused] = p[unsafe_offset=k]
                vused += 1
            pos = neol + 2
            folds += 1
        # RFC 9110 5.2: strip trailing OWS from the assembled value
        var vend = vused
        while vend > val_off:
            var cv = Int64(v[unsafe_offset=vend - 1])
            if cv != 32 and cv != 9:
                break
            vend -= 1
        # RFC 9110 5.5: forbidden control characters in the field value
        for k in range(val_off, vend):
            var cv = Int64(v[unsafe_offset=k])
            if lax == 1:
                if cv == 10 or cv == 13 or cv == 0:
                    o[unsafe_offset=stride + 1] = -1
                    return -1
            else:
                if cv == 9:
                    continue
                if cv < 32 or cv == 127:
                    o[unsafe_offset=stride + 1] = -1
                    return -1
        var slot = Int64(count) * 4
        o[unsafe_offset=slot + 0] = Int64(name_off)
        o[unsafe_offset=slot + 1] = Int64(name_len)
        o[unsafe_offset=slot + 2] = Int64(val_off)
        o[unsafe_offset=slot + 3] = Int64(vend - val_off)
        count += 1
    o[unsafe_offset=stride + 0] = Int64(count)
    o[unsafe_offset=stride + 1] = 0
    o[unsafe_offset=stride + 2] = Int64(folds)
    o[unsafe_offset=stride + 3] = Int64(vused)
    return count


# ---------------------------------------------------------------------------
# Content-Length payload bookkeeping (aiohttp PARSE_LENGTH)
# ---------------------------------------------------------------------------


@export("ah_content_length_plan")
def ah_content_length_plan(
    total: Int, feed_addr: Int, n: Int, out_addr: Int
) abi("C") -> Int:
    """Replay `HttpPayloadParser.feed_data` in PARSE_LENGTH mode.

    For each feed of `feed[i]` bytes, records how many bytes the parser hands
    to the payload in `out[i]`, and accumulates in `out[n]` the bytes the
    parser carries over to the next message. `out` must hold `n + 1` Int64
    slots. Returns the bytes still outstanding when the input ran out.
    """
    var f = lptr(feed_addr)
    var o = lptr(out_addr)
    var remaining = Int64(total)
    var tail = Int64(0)
    for i in range(n):
        var size = f[unsafe_offset=i]
        var required = remaining
        if size < required:
            remaining = required - size
            o[unsafe_offset=i] = size
        else:
            remaining = 0
            tail += size - required
            o[unsafe_offset=i] = required
    o[unsafe_offset=Int64(n)] = tail
    return Int(remaining)


# ---------------------------------------------------------------------------
# WebSocket max_msg_size guard arithmetic
# ---------------------------------------------------------------------------


@export("ah_bytes_until_limit")
def ah_bytes_until_limit(
    limit: Int, have: Int, chunk_addr: Int, n: Int
) abi("C") -> Int:
    """Total projected message size after feeding `n` more chunks.

    Mirrors the `max_msg_size` guard aiohttp applies before buffering payload
    bytes: `projected = have + chunk`, and the frame is rejected at
    `projected >= limit`. `limit` is accepted so the call reads like the guard
    it implements; the caller compares the returned total against it.
    """
    var c = lptr(chunk_addr)
    var total = Int64(have)
    for i in range(n):
        total += c[unsafe_offset=i]
    return Int(total)
