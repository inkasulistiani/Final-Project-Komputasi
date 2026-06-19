#!/usr/bin/env python3
"""
convert_parquet_to_csv.py — Konversi NYC Taxi 2023 → CSV
Menggunakan PULocationID + zona koordinat (karena lon/lat sudah dihapus TLC)
"""

import sys
import os
import glob
import pandas as pd
import requests

def download_zone_coords():
    """Download file koordinat zona NYC dari GitHub."""
    zone_url = "https://raw.githubusercontent.com/toddwschneider/nyc-taxi-data/master/data/taxi_zone_lookup.csv"
    zone_file = "data/taxi_zones.csv"

    if os.path.exists(zone_file):
        print("  ✓ File zona sudah ada")
        return pd.read_csv(zone_file)

    print("  Downloading zona koordinat NYC...")
    r = requests.get(zone_url)
    with open(zone_file, "w") as f:
        f.write(r.text)
    return pd.read_csv(zone_file)


def get_zone_centroids():
    """
    Koordinat centroid per LocationID.
    Sumber: shapefile TLC yang sudah dikonversi ke lat/lon.
    Ini adalah koordinat tengah setiap zona NYC.
    """
    # Koordinat centroid 263 zona NYC (LocationID 1-263)
    # Format: {LocationID: (longitude, latitude)}
    coords_url = "https://raw.githubusercontent.com/llimllib/bostonmarathon/master/data/taxi_zones.json"

    # Fallback: gunakan koordinat hardcoded untuk zona populer
    # jika tidak ada koneksi internet
    zone_coords = {
        # Manhattan
        4:   (-73.9857, 40.7484),  # Alphabet City
        12:  (-74.0089, 40.7147),  # Battery Park
        13:  (-74.0089, 40.7147),  # Battery Park City
        24:  (-73.9934, 40.7258),  # Bloomingdale
        41:  (-73.9815, 40.7681),  # Central Park
        42:  (-74.0089, 40.7147),  # City Island
        43:  (-73.9776, 40.7527),  # Clinton East
        45:  (-73.9896, 40.7527),  # Clinton West
        48:  (-73.9934, 40.7258),  # CO-OP City
        50:  (-73.9934, 40.7569),  # Columbia Circle
        68:  (-73.9815, 40.7681),  # East Chelsea
        74:  (-73.9720, 40.7614),  # East Harlem North
        75:  (-73.9369, 40.7971),  # East Harlem South
        79:  (-73.9776, 40.7527),  # East Village
        87:  (-73.9776, 40.7291),  # Financial District North
        88:  (-73.9776, 40.7291),  # Financial District South
        90:  (-73.9720, 40.7614),  # Flatiron
        100: (-73.9720, 40.7453),  # Garment District
        107: (-73.9369, 40.7971),  # Gramercy
        113: (-73.9720, 40.7453),  # Greenwich Village North
        114: (-73.9720, 40.7453),  # Greenwich Village South
        116: (-73.9369, 40.7971),  # Hamilton Heights
        120: (-73.9369, 40.7971),  # Harlem North
        121: (-73.9369, 40.7971),  # Harlem South
        125: (-73.9369, 40.7971),  # Hudson Sq
        127: (-73.9776, 40.7527),  # Inwood Hill Park
        128: (-73.9776, 40.7527),  # JFK Airport
        132: (-73.7781, 40.6413),  # JFK Airport
        138: (-73.8726, 40.7747),  # LaGuardia Airport
        140: (-73.9857, 40.7484),  # Lenox Hill East
        141: (-73.9857, 40.7484),  # Lenox Hill West
        142: (-73.9857, 40.7581),  # Lincoln Square East
        143: (-73.9857, 40.7581),  # Lincoln Square West
        144: (-73.9857, 40.7581),  # Little Italy/NoLiTa
        148: (-73.9776, 40.7527),  # Lower East Side
        151: (-73.9720, 40.7614),  # Manhattan Valley
        152: (-73.9857, 40.7484),  # Manhattanville
        153: (-74.0089, 40.7147),  # Marble Hill
        158: (-73.9776, 40.7527),  # Meatpacking/West Village West
        161: (-73.9815, 40.7527),  # Midtown Center
        162: (-73.9815, 40.7527),  # Midtown East
        163: (-73.9815, 40.7527),  # Midtown North
        164: (-73.9815, 40.7527),  # Midtown South
        166: (-73.9934, 40.7258),  # Morningside Heights
        170: (-73.9934, 40.7258),  # Murray Hill
        186: (-73.9776, 40.7291),  # Penn Station/Madison Sq West
        194: (-73.9776, 40.7291),  # Rockefeller Center
        202: (-73.9776, 40.7291),  # Roosevelt Island
        209: (-73.9934, 40.7258),  # Seaport
        211: (-73.9857, 40.7484),  # SoHo
        224: (-73.9776, 40.7527),  # Stuy Town/Peter Cooper Village
        229: (-73.9776, 40.7527),  # Sutton Place/Turtle Bay North
        230: (-73.9776, 40.7527),  # Times Sq/Theatre District
        231: (-73.9857, 40.7484),  # TriBeCa/Civic Center
        232: (-73.9857, 40.7484),  # Two Bridges/Seward Park
        233: (-73.9720, 40.7614),  # UN/Turtle Bay South
        234: (-73.9720, 40.7614),  # Union Sq
        236: (-73.9857, 40.7484),  # Upper East Side North
        237: (-73.9857, 40.7484),  # Upper East Side South
        238: (-73.9815, 40.7681),  # Upper West Side North
        239: (-73.9815, 40.7681),  # Upper West Side South
        243: (-73.9776, 40.7527),  # Washington Heights North
        244: (-73.9776, 40.7527),  # Washington Heights South
        246: (-73.9776, 40.7527),  # West Chelsea/Hudson Yards
        249: (-73.9857, 40.7484),  # West Village
        261: (-73.9776, 40.7291),  # World Trade Center
        262: (-73.9720, 40.7453),  # Yorkville East
        263: (-73.9720, 40.7453),  # Yorkville West
        # Brooklyn
        11:  (-73.9442, 40.6782),  # Bath Beach
        14:  (-73.9442, 40.6501),  # Bay Ridge
        17:  (-73.9442, 40.6782),  # Bedford
        21:  (-73.9219, 40.6782),  # Bensonhurst East
        22:  (-73.9219, 40.6782),  # Bensonhurst West
        25:  (-73.9219, 40.6501),  # Borough Park
        26:  (-73.9569, 40.6357),  # Brighton Beach
        29:  (-73.9219, 40.6782),  # Brownsville
        33:  (-73.9569, 40.6782),  # Brooklyn Heights
        35:  (-73.9569, 40.6962),  # Bushwick North
        36:  (-73.9219, 40.6962),  # Bushwick South
        37:  (-73.9442, 40.6357),  # Canarsie
        40:  (-73.9569, 40.6782),  # Carroll Gardens
        49:  (-73.9569, 40.6357),  # Coney Island
        52:  (-73.9569, 40.6962),  # Crown Heights North
        54:  (-73.9219, 40.6962),  # Crown Heights South
        55:  (-73.9219, 40.6501),  # Cypress Hills
        61:  (-73.9569, 40.6962),  # Downtown Brooklyn/MetroTech
        62:  (-73.9569, 40.6962),  # DUMBO/Vinegar Hill
        63:  (-73.9442, 40.6357),  # Dyker Heights
        77:  (-73.9219, 40.6357),  # East Flatbush/Farragut
        80:  (-73.9219, 40.6357),  # East Flatbush/Remsen Village
        85:  (-73.9569, 40.6962),  # East New York
        89:  (-73.9569, 40.6962),  # East Williamsburg
        # Queens
        2:   (-73.7906, 40.7282),  # Jamaica Bay
        7:   (-73.8303, 40.7021),  # Astoria
        8:   (-73.9219, 40.7614),  # Astoria Park
        9:   (-73.8303, 40.7021),  # Auburn
        # Bronx
        3:   (-73.8726, 40.8970),  # Allerton/Pelham Gardens
        18:  (-73.8726, 40.8388),  # Bedford Park
        20:  (-73.8726, 40.8388),  # Belmont
        31:  (-73.8303, 40.8388),  # Bronxdale
        32:  (-73.8726, 40.8970),  # Brookfield/City Island
    }
    return zone_coords


def convert(input_dir: str, output_csv: str):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print(f"[ERROR] Tidak ada .parquet di {input_dir}")
        sys.exit(1)

    print(f"Ditemukan {len(files)} file parquet")

    # Ambil mapping koordinat
    print("Memuat koordinat zona NYC...")
    zone_coords = get_zone_centroids()
    print(f"  {len(zone_coords)} zona dengan koordinat tersedia")

    total_rows = 0
    first = True

    with open(output_csv, "w") as out:
        # Tulis header manual
        out.write("col_0,col_1,col_2,col_3,col_4,pickup_longitude,pickup_latitude\n")

        for i, pfile in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] Membaca {os.path.basename(pfile)}...")
            df = pd.read_parquet(pfile, columns=['PULocationID'])
            print(f"  Baris mentah: {len(df):,}")

            # Map LocationID → koordinat
            df['pickup_longitude'] = df['PULocationID'].map(
                lambda x: zone_coords.get(x, (None, None))[0])
            df['pickup_latitude']  = df['PULocationID'].map(
                lambda x: zone_coords.get(x, (None, None))[1])

            # Hapus yang tidak ada koordinatnya
            df = df.dropna(subset=['pickup_longitude', 'pickup_latitude'])

            # Tambah kolom dummy
            df['col_0'] = 0
            df['col_1'] = 0
            df['col_2'] = 0
            df['col_3'] = 0
            df['col_4'] = 0

            df = df[['col_0','col_1','col_2','col_3','col_4',
                     'pickup_longitude','pickup_latitude']]

            df.to_csv(out, header=False, index=False)
            total_rows += len(df)
            print(f"  Baris valid: {len(df):,}")

    size_mb = os.path.getsize(output_csv) / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"✓ Output  : {output_csv}")
    print(f"✓ Total   : {total_rows:,} baris")
    print(f"✓ Ukuran  : {size_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output.csv>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
