import numpy as np, subprocess, os, struct

BASE = os.path.expanduser('~/workspace/pherc1667/path4_ink')
ZARR = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
TW, TH = 30097, 2061
PX0, PY0, PS = 5858, 193, 400
SU, SV = 30097/9205, 2061/709
U0, V0 = int(PX0*SU), int(PY0*SV)   # 19153, 561
US, VS = int(PS*SU), int(PS*SV)     # 1308, 1162

def read_tif_pixel(f, u, v):
    # uncompressed float32 TIFF, single strip: header then pixels row-major
    # parse header to find strip offset
    with open(f,'rb') as fh:
        fh.seek(0); head = fh.read(8)
        bo = '<' if head[:2]==b'II' else '>'
        off = struct.unpack(bo+'I', head[4:8])[0]
        fh.seek(off)
        n = struct.unpack(bo+'H', fh.read(2))[0]
        strip = None
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo+'HHI4s', fh.read(12))
            if tag == 273:  # StripOffsets
                strip = struct.unpack(bo+'I', val)[0]
        assert strip is not None
        fh.seek(strip + (v*TW + u)*4)
        return struct.unpack('<f', fh.read(4))[0]

# patch center + 4 corners
pts = [(U0+US//2, V0+VS//2), (U0, V0), (U0+US-1, V0), (U0, V0+VS-1), (U0+US-1, V0+VS-1)]
for (u,v) in pts:
    x = read_tif_pixel(os.path.join(BASE,'tifxyz_2399_x.tif'), u, v)
    y = read_tif_pixel(os.path.join(BASE,'tifxyz_2399_y.tif'), u, v)
    z = read_tif_pixel(os.path.join(BASE,'tifxyz_2399_z.tif'), u, v)
    print(f"tifxyz({u},{v}) -> vol({x:.1f},{y:.1f},{z:.1f}) chunk({int(z)//128},{int(y)//128},{int(x)//128})")
