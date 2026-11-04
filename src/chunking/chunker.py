"""Recursive character chunker that tracks character offsets itself.

A chunk is a (start, end) range of the document text; its text is always
text[start:end]. Sizes are in characters.
"""

DEFAULT_SEPS = ["\n\n", "\n", ". ", " "]


def _atoms(text, s, e, seps, size):
    """Contiguous (start, end) ranges covering text[s:e], each <= size chars."""
    if e - s <= size:
        return [(s, e)]
    if not seps:  # last resort: hard cut
        return [(i, min(i + size, e)) for i in range(s, e, size)]
    sep, rest = seps[0], seps[1:]
    out, pos = [], s
    while pos < e:
        i = text.find(sep, pos, e)
        end = e if i == -1 else i + len(sep)   # separator stays with the piece before it
        out.extend(_atoms(text, pos, end, rest, size))
        pos = end
    return out


def _trim(text, s, e):
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def chunk_text(text, size=512, overlap=64, seps=None):
    """Return a list of (start, end) chunk ranges, in order, whitespace-trimmed."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    atoms = _atoms(text, 0, len(text), list(seps or DEFAULT_SEPS), size)

    spans, window, wlen = [], [], 0
    for a in atoms:
        alen = a[1] - a[0]
        if window and wlen + alen > size:
            spans.append((window[0][0], window[-1][1]))
            tail, tlen = [], 0                      # carry over up to `overlap` chars
            for b in reversed(window):
                blen = b[1] - b[0]
                if tlen + blen > overlap:
                    break
                tail.insert(0, b)
                tlen += blen
            while tail and tlen + alen > size:      # never let overlap + new atom exceed size
                tlen -= tail[0][1] - tail[0][0]
                tail.pop(0)
            window, wlen = tail, tlen
        window.append(a)
        wlen += alen
    if window:
        spans.append((window[0][0], window[-1][1]))

    out = []
    for s, e in spans:
        s, e = _trim(text, s, e)
        if e > s:
            out.append((s, e))
    return out


def make_chunks(docs, size=512, overlap=64):
    """docs: {doc_key: text}. Returns a list of chunk dicts."""
    chunks = []
    for key in sorted(docs):
        text = docs[key]
        for s, e in chunk_text(text, size, overlap):
            chunks.append({"cid": len(chunks), "doc": key,
                           "start": s, "end": e, "text": text[s:e]})
    return chunks