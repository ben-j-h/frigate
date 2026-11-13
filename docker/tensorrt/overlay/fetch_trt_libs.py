"""Fetch only the TensorRT libraries onnxruntime-gpu's TensorRT provider loads, straight out of NVIDIA's wheel.

The tensorrt-cu12-libs wheel is ~4.3 GB (builder resources for every GPU generation, Windows copies, lean/dispatch
runtimes). The ONNX detector needs libnvinfer, libnvonnxparser and libnvinfer_plugin, plus the builder resource for each
GPU it builds engines on. This reads the wheel's zip directory over HTTP range requests and pulls only those members
(~1.3 GB for sm86 + sm120), so a build host needs no room for the full wheel. Every member's CRC is checked by zipfile.

usage: fetch_trt_libs.py <tensorrt version> <out dir> <sm> [<sm> ...]   e.g. 10.16.1.11 /trt 86 120
"""
import io
import os
import re
import shutil
import sys
import urllib.request
import zipfile

INDEX = "https://pypi.nvidia.com/tensorrt-cu12-libs/"
LIBS = ("libnvinfer.so.10", "libnvonnxparser.so.10", "libnvinfer_plugin.so.10")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file (the server must answer Range requests with 206)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            if r.status != 206:
                raise IOError(f"server ignored the Range request ({r.status})")
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main() -> None:
    version, out, sms = sys.argv[1], sys.argv[2], sys.argv[3:]
    index = urllib.request.urlopen(INDEX).read().decode()
    name = re.search(rf'href="(tensorrt_cu12_libs-{re.escape(version)}-py3-none-manylinux_[^"#]*x86_64\.whl)', index)
    if not name:
        sys.exit(f"tensorrt-cu12-libs {version} (x86_64) not found on {INDEX}")
    want = set(LIBS) | {f"libnvinfer_builder_resource_sm{sm}.so.{version.rsplit('.', 1)[0]}" for sm in sms}
    os.makedirs(out, exist_ok=True)
    # big sequential reads: zipfile asks for small chunks, so buffer them
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(INDEX + name.group(1)), buffer_size=16 << 20)) as z:
        members = {os.path.basename(i.filename): i for i in z.infolist() if "/" in i.filename}
        missing = want - members.keys()
        if missing:
            sys.exit(f"not in the wheel: {sorted(missing)}; it has {sorted(m for m in members if m.startswith('lib'))}")
        for m in sorted(want):
            with z.open(members[m]) as src, open(os.path.join(out, m), "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            print(f"{m}: {members[m].file_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
