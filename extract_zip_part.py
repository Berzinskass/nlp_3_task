# Only needed if a normal unzip tool cannot open your archive part
# (e.g. emo-wavs.zip.009 is one piece of a big split zip).
# It finds every complete wav file inside the piece and saves it to emo_wavs/.
#
#   python extract_zip_part.py emo-wavs.zip.009

import mmap
import os
import struct
import sys
import zlib

archive = sys.argv[1]
os.makedirs("emo_wavs", exist_ok=True)

with open(archive, "rb") as f:
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

saved = 0
pos = data.find(b"PK\x03\x04")  # every file in a zip starts with these 4 bytes
while pos != -1:
    method = struct.unpack("<H", data[pos + 8:pos + 10])[0]  # 0 = stored, 8 = compressed
    size = struct.unpack("<I", data[pos + 18:pos + 22])[0]
    name_len, extra_len = struct.unpack("<HH", data[pos + 26:pos + 30])
    name = data[pos + 30:pos + 30 + name_len].decode("utf-8", "ignore")
    start = pos + 30 + name_len + extra_len

    if name.endswith(".wav"):
        content = None
        if method == 8:
            d = zlib.decompressobj(-15)
            try:
                out = d.decompress(data[start:start + (size or 100_000_000)])
                if d.eof:  # False if the file is cut off at the end of the piece
                    content = out
            except zlib.error:
                pass
        elif method == 0 and start + size <= len(data):
            content = data[start:start + size]
        if content:
            with open(os.path.join("emo_wavs", os.path.basename(name)), "wb") as out_file:
                out_file.write(content)
            saved += 1

    pos = data.find(b"PK\x03\x04", pos + 4)

print(f"Saved {saved} wav files to emo_wavs/")
