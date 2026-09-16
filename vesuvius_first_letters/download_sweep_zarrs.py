"""Download PHerc1447 segment surface-volume zarrs from the public S3 bucket.

Usage: venv/bin/python download_sweep_zarrs.py [seg_id ...]
With no args, downloads all 14 sweep segments (everything except the main
20250703034159 segment, already local).
"""
import os
import sys
import s3fs

BUCKET = "vesuvius-challenge-open-data"
VOL = "8.64um-1.2m-116keV-volume-20250521151220.zarr"

SEGMENTS = [
    ("20250502183421", "auto_grown_20250502161744358"),
    ("20251105093211", "z_dbg_gen_00320"),
    ("20250502182142", "auto_grown_20250502161324419"),
    ("20250502185519", "auto_grown_20250502164303733"),
    ("20250702235910", "auto_grown_20250702235910292"),
    ("20250703025628", "auto_grown_20250703025628283"),
    ("20250502205333", "auto_grown_20250502181030065"),
    ("20250502182456", "auto_grown_20250502161202782"),
    ("20250502183138", "auto_grown_20250502162038685"),
    ("20250502184845", "auto_grown_20250502164121265"),
    ("20250502180748", "auto_grown_20250502160748721"),
    # ("20250703034159", "auto_grown_20250703034159599"),  # main segment, local
    ("20250502184658", "auto_grown_20250502163923577"),
    ("20250502184201", "auto_grown_20250502163549332"),
    ("20250502180708", "auto_grown_20250502160708188"),
]

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters/PHerc1447/sweep")


def main():
    only = set(sys.argv[1:])
    fs = s3fs.S3FileSystem(anon=True)
    for seg_id, suffix in SEGMENTS:
        if only and seg_id not in only:
            continue
        long_id = f"{seg_id}-{suffix}"
        remote = f"{BUCKET}/PHerc1447/segments/{long_id}/surface-volumes/{VOL}"
        local = os.path.join(BASE, f"seg_{seg_id}.zarr")
        try:
            import zarr
            g = zarr.open_group(local, mode="r")
            if g["0"].shape[0] > 0:
                print(f"[skip] {seg_id} already local", flush=True)
                continue
        except Exception:
            pass
        print(f"[get] {seg_id} -> {local}", flush=True)
        try:
            fs.get(remote, local, recursive=True)
            # verify it opens (multiscale group: level '0' is full-res)
            import zarr
            g = zarr.open_group(local, mode="r")
            a = g["0"]
            print(f"  ok: level0 shape={a.shape} chunks={a.chunks} dtype={a.dtype}", flush=True)
        except Exception as e:
            print(f"  FAILED {seg_id}: {e}", flush=True)


if __name__ == "__main__":
    main()
