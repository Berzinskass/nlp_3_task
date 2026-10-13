"""Extract WAV files from one EMO archive part (e.g. emo-wavs.zip.009).

A single part of a split ZIP has no usable central directory, so `zipfile`
cannot open it. Instead we walk the local file headers: every complete file
stored in the part is recovered, and the partial files at its start and end
are skipped. A regular (complete) .zip file is handled with `zipfile`.
"""

import mmap
import os
import struct
import zipfile
import zlib

LOCAL_HEADER = b"PK\x03\x04"
DATA_DESCRIPTOR = b"PK\x07\x08"
_HEADER = struct.Struct("<4sHHHHHIIIHH")  # 30 bytes


def _parse_header(buf, pos):
    """Return entry metadata for a local header at `pos`, or None if invalid."""
    if pos + _HEADER.size > len(buf):
        return None
    (sig, _ver, flags, method, _t, _d, crc, csize, usize,
     name_len, extra_len) = _HEADER.unpack_from(buf, pos)
    if sig != LOCAL_HEADER or method not in (0, 8) or not 0 < name_len < 1024:
        return None
    name_start = pos + _HEADER.size
    data_start = name_start + name_len + extra_len
    if data_start > len(buf):
        return None
    raw_name = bytes(buf[name_start:name_start + name_len])
    try:
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
    except UnicodeDecodeError:
        return None
    if any(ord(c) < 32 for c in name):
        return None
    extra = bytes(buf[name_start + name_len:data_start])
    if 0xFFFFFFFF in (csize, usize):  # ZIP64 sizes live in the extra field
        csize, usize = _zip64_sizes(extra, csize, usize)
    return {"name": name, "flags": flags, "method": method, "crc": crc,
            "csize": csize, "usize": usize, "data_start": data_start}


def _zip64_sizes(extra, csize, usize):
    i = 0
    while i + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, i)
        if tag == 0x0001:
            fields = extra[i + 4:i + 4 + size]
            values = list(struct.unpack_from(f"<{len(fields) // 8}Q", fields))
            if usize == 0xFFFFFFFF and values:
                usize = values.pop(0)
            if csize == 0xFFFFFFFF and values:
                csize = values.pop(0)
            break
        i += 4 + size
    return csize, usize


def _read_entry(buf, entry):
    """Return (data, end_position) or (None, end_position) if unrecoverable."""
    start = entry["data_start"]
    has_descriptor = entry["flags"] & 0x08
    if entry["flags"] & 0x01:  # encrypted
        return None, start + entry["csize"]

    if entry["method"] == 8:
        d = zlib.decompressobj(-15)
        chunk = entry["csize"] + 64 if entry["csize"] else 1 << 20
        pos = start
        out = []
        while not d.eof and pos < len(buf):
            out.append(d.decompress(bytes(buf[pos:pos + chunk])))
            pos += chunk
        if not d.eof:  # truncated at the end of the part
            return None, len(buf)
        end = pos - len(d.unused_data)
        data = b"".join(out)
    else:
        if has_descriptor and entry["csize"] == 0:
            end = buf.find(DATA_DESCRIPTOR, start)
            if end < 0:
                return None, len(buf)
        else:
            end = start + entry["csize"]
        if end > len(buf):
            return None, len(buf)
        data = bytes(buf[start:end])

    crc = entry["crc"]
    if has_descriptor:
        desc_pos = end
        if bytes(buf[end:end + 4]) == DATA_DESCRIPTOR:
            desc_pos += 4
        if desc_pos + 12 <= len(buf):
            crc = struct.unpack_from("<I", buf, desc_pos)[0]
        end = desc_pos + 12
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        return None, end
    return data, end


def iter_split_part(path):
    """Yield (name, bytes) for every complete file inside an archive part."""
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        pos = buf.find(LOCAL_HEADER)
        while 0 <= pos < len(buf):
            entry = _parse_header(buf, pos)
            if entry is None:
                pos = buf.find(LOCAL_HEADER, pos + 1)
                continue
            data, end = _read_entry(buf, entry)
            if data is not None and not entry["name"].endswith("/"):
                yield entry["name"], data
            # The next header normally starts right at `end`; if not (we
            # resynced onto a false signature), keep searching.
            nxt = end if bytes(buf[end:end + 4]) == LOCAL_HEADER else \
                buf.find(LOCAL_HEADER, max(end, pos + 1))
            pos = nxt


def extract_wavs(archive, out_dir):
    """Extract all WAV files from `archive` into `out_dir` (flat, by basename).

    Returns the list of extracted file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    extracted = []

    def keep(name):
        return name.lower().endswith(".wav")

    def write(name, data):
        path = os.path.join(out_dir, os.path.basename(name))
        with open(path, "wb") as f:
            f.write(data)
        extracted.append(path)

    try:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if keep(info.filename):
                    write(info.filename, zf.read(info))
        return extracted
    except (zipfile.BadZipFile, NotImplementedError, OSError) as err:
        print(f"zipfile cannot open {os.path.basename(archive)} ({err}); "
              "reading it as one part of a split archive")

    for name, data in iter_split_part(archive):
        if keep(name):
            write(name, data)
    return extracted

