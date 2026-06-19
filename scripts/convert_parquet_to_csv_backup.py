#!/usr/bin/env python3
"""
convert_parquet_to_csv.py — Konversi NYC Taxi parquet → CSV
Jalankan: python3 scripts/convert_parquet_to_csv.py data/ data/nyc_taxi_2023.csv
"""

import sys
import os
import glob
import csv
import struct
import io

def convert_with_pandas(input_dir: str, output_csv: str):
    """Konversi menggunakan pandas + pyarrow (metode utama)."""
    import pandas as pd

    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print(f"[ERROR] Tidak ada file .parquet di {input_dir}")
        sys.exit(1)

    print(f"Ditemukan {len(files)} file parquet:")
    for f in files:
        print(f"  {f} ({os.path.getsize(f)/1024/1024:.1f} MB)")

    total_rows = 0
    with open(output_csv, "w", newline="") as out:
        writer = None

        for i, pfile in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] Membaca {os.path.basename(pfile)}...")
            df = pd.read_parquet(pfile)
            print(f"  Kolom: {list(df.columns)}")
            print(f"  Baris: {len(df):,}")

            # Deteksi nama kolom longitude/latitude
            col_lon = next((c for c in df.columns if "lon" in c.lower()
                            or c in ["pickup_longitude", "start_lon"]), None)
            col_lat = next((c for c in df.columns if "lat" in c.lower()
                            or c in ["pickup_latitude", "start_lat"]), None)

            if not col_lon or not col_lat:
                print(f"  [WARN] Kolom lon/lat tidak ditemukan, skip.")
                continue

            # Buat DataFrame minimal
            sub = df[[col_lon, col_lat]].copy()
            sub.columns = ["pickup_longitude", "pickup_latitude"]

            # Dummy kolom agar kompatibel dengan C++ reader (col 5, 6)
            for idx in range(5):
                if idx not in sub.columns:
                    sub.insert(idx, f"col_{idx}", 0)

            # Filter nilai valid
            sub = sub[
                (sub["pickup_longitude"].between(-75.0, -73.0)) &
                (sub["pickup_latitude"].between(40.0, 41.5))
            ].dropna()

            print(f"  Baris valid: {len(sub):,}")

            if writer is None:
                # Tulis header hanya satu kali
                out.write(",".join(sub.columns) + "\n")

            sub.to_csv(out, header=False, index=False)
            total_rows += len(sub)
            print(f"  ✓ Ditulis ke CSV")

    size_mb = os.path.getsize(output_csv) / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"Output   : {output_csv}")
    print(f"Total baris: {total_rows:,}")
    print(f"Ukuran   : {size_mb:.1f} MB")
    if size_mb < 5000:
        print(f"[WARN] Dataset < 5 GB ({size_mb:.0f} MB). Download lebih banyak bulan.")
    else:
        print(f"✓ Dataset >= 5 GB siap digunakan!")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output.csv>")
        sys.exit(1)

    input_dir  = sys.argv[1]
    output_csv = sys.argv[2]

    try:
        convert_with_pandas(input_dir, output_csv)
    except ImportError:
        print("[ERROR] pandas/pyarrow tidak tersedia.")
        print("  Install: pip install pandas pyarrow")
        sys.exit(1)


if __name__ == "__main__":
    main()
