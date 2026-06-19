# K-Means Clustering Terdistribusi — NYC Taxi Trips

Implementasi K-Means Clustering pada dataset NYC Taxi Trips (≥5 GB) dengan
empat pendekatan komputasi: **Serial**, **OpenMP**, **CUDA**, dan **Distributed (Spark)**.

---

## Struktur Repository

```
kmeans-distributed/
├── README.md                    ← Dokumentasi ini
├── Makefile                     ← Build system utama
│
├── src/
│   ├── serial/
│   │   └── kmeans_serial.cpp    ← Implementasi baseline single-thread
│   ├── openmp/
│   │   └── kmeans_omp.cpp       ← OpenMP: parallel for, reduction, atomic, barrier
│   ├── cuda/
│   │   └── kmeans_cuda.cu       ← CUDA: shared memory kernel, occupancy analysis
│   └── distributed/
│       └── kmeans_spark.py      ← PySpark: 3+ node, broadcast, adaptive QE
│
├── infra/
│   ├── Dockerfile.cpu           ← Multi-stage build: serial + openmp
│   ├── Dockerfile.cuda          ← Multi-stage build: CUDA sm_75/80/86/89
│   ├── docker-compose.yml       ← 3-node Spark + Prometheus + Grafana lokal
│   └── monitoring/
│       └── prometheus.yml       ← Scrape config: node, JMX, GPU, app metrics
│
├── scripts/
│   ├── convert_parquet_to_csv.py ← Preprocess dataset NYC Taxi
│   ├── analyze_results.py        ← Hitung speedup, efficiency, buat grafik
│   ├── entrypoint_cpu.sh         ← Docker entrypoint CPU container
│   └── entrypoint_cuda.sh        ← Docker entrypoint CUDA container
│
├── data/                        ← Dataset (tidak di-commit ke Git)
└── bin/                         ← Binary hasil kompilasi
```

---

## Cara Build & Menjalankan

### Prasyarat

```bash
# Untuk Serial & OpenMP
sudo apt-get install -y build-essential g++-12 libomp-dev

# Untuk CUDA
# Install CUDA Toolkit 12.x dari https://developer.nvidia.com/cuda-downloads

# Untuk PySpark
pip install pyspark pandas pyarrow

# Untuk visualisasi hasil
pip install matplotlib
```

### 1. Download Dataset

Dataset: NYC Taxi Trip Records 2023 (~6 GB setelah digabung)

```bash
make download_data
# Mengunduh 12 bulan Parquet dari TLC Trip Record Data
# Mengkonversi ke CSV: data/nyc_taxi_2023.csv
```

Manual (jika curl tidak bisa):
```
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Download: yellow_tripdata_2023-01.parquet s/d yellow_tripdata_2023-12.parquet
Simpan di folder: data/
Lalu jalankan: python3 scripts/convert_parquet_to_csv.py data/ data/nyc_taxi_2023.csv
```

### 2. Build Semua Implementasi

```bash
# Build serial + openmp (tidak butuh CUDA)
make all

# Build CUDA (butuh nvcc)
make cuda

# Build semua Docker image
make docker
```

### 3. Jalankan Satu per Satu

```bash
# Serial baseline
./bin/kmeans_serial data/nyc_taxi_2023.csv 8 100

# OpenMP (4 thread)
OMP_NUM_THREADS=4 ./bin/kmeans_omp data/nyc_taxi_2023.csv 8 100

# OpenMP (16 thread)
OMP_NUM_THREADS=16 ./bin/kmeans_omp data/nyc_taxi_2023.csv 8 100

# CUDA
./bin/kmeans_cuda data/nyc_taxi_2023.csv 8 100

# PySpark lokal (3 thread simulasi)
spark-submit --master local[3] src/distributed/kmeans_spark.py \
  --input data/nyc_taxi_2023.csv --k 8 --max_iter 100
```

---

## Reproducing Eksperimen

### Eksperimen 1: Strong Scaling (OpenMP)

**Konsep**: Ukuran data tetap, jumlah thread bertambah (1 → 16).
Speedup = T₁ / Tₙ

```bash
make experiment_strong
```

Atau manual:
```bash
DATA=data/nyc_taxi_2023.csv
# Serial baseline
./bin/kmeans_serial $DATA 8 50
# Thread 1, 2, 4, 8, 16
for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=$T ./bin/kmeans_omp $DATA 8 50
done
```
### Eksperimen 2: Weak Scaling (OpenMP)

**Konsep**: Ukuran data DAN thread bertambah proporsional.
Efficiency = T₁ / Tₙ (idealnya tetap 100%)

```bash
make experiment_weak
```

### Eksperimen 3: GPU vs CPU

Jika tidak punya GPU — gunakan Google Colab (gratis):
1. Buka: https://colab.research.google.com
2. Runtime → Change runtime type → GPU (T4)
3. Upload file kmeans_cuda.cu, kmean_serial.cpp, dan nyc_taxi_2023.csv
4. Di cell baru, ketik:
   ```bash 
   !nvcc -O3 -arch=sm_75 -o kmeans_cuda kmeans_cuda.cu
   !./kmeans_cuda nyc_taxi_2023.csv 8 100
   ```
5. Bandingkan GPU vs CPU (Serial)
   
 ```bash  
!g++ -O3 -std=c++17 -o kmeans_serial kmeans_serial.cpp

import time, subprocess

def run_and_time(cmd):
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    return elapsed, result.stdout

DATA   = "nyc_taxi_2023.csv"
K      = 8
ITERS  = 50
ROWS   = 1_000_000  # pakai 1 juta baris agar serial tidak terlalu lama

print("=" * 50)
print("Perbandingan Serial vs CUDA")
print("=" * 50)

print("\n[1/2] Menjalankan Serial...")
t_serial, out_serial = run_and_time(
    f"./kmeans_serial {DATA} {K} {ITERS} {ROWS}")
print(out_serial[-500:])  # tampilkan 500 char terakhir
print(f"  → Waktu total: {t_serial:.2f}s")

print("\n[2/2] Menjalankan CUDA...")
t_cuda, out_cuda = run_and_time(
    f"./kmeans_cuda {DATA} {K} {ITERS} {ROWS}")
print(out_cuda[-500:])
print(f"  → Waktu total: {t_cuda:.2f}s")

print("\n" + "=" * 50)
print(f"  Serial : {t_serial:.2f}s")
print(f"  CUDA   : {t_cuda:.2f}s")
print(f"  Speedup: {t_serial/t_cuda:.1f}x")
print("=" * 50)
```

### Eksperimen 4: Multi-Node Spark

**Lokal** (simulasi 3 node via docker-compose):
```bash
cd infra
docker-compose up -d
# Tunggu ~60 detik sampai semua service ready
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --num-executors 3 \
  --executor-memory 3g \
  /app/src/distributed/kmeans_spark.py \
  --input /data/nyc_taxi_2023.csv \
  --k 8 --max_iter 50

# Lihat Spark UI: http://localhost:8080
# Lihat Grafana: http://localhost:3000 (admin/kmeans123)
```

## Monitoring

Dashboard Grafana tersedia di `http://localhost:3000` setelah `docker-compose up`.

**Panel yang tersedia**:
- CPU Utilization (%) per core
- Memory usage (GB)
- Spark Executor Active Tasks
- Network throughput antar node
- K-Means Throughput (pts/sec) real-time
- WCSS per iterasi

**Akses Prometheus**: `http://localhost:9090`
**Akses Spark UI**: `http://localhost:8080`

---

## Penjelasan Pola Paralel

### OpenMP
| Pola | Lokasi | Tujuan |
|------|--------|--------|
| `#pragma omp parallel for` | Assignment loop | Paralel hitung jarak setiap titik |
| `reduction(+:moved)` | Counter konvergensi | Jumlahkan moved tanpa race condition |
| Thread-private buffer | Update centroid | Hindari race pada array akumulasi |
| `#pragma omp barrier` | Antara fase | Sinkronisasi assignment → update |

### CUDA
| Teknik | Kernel | Keuntungan |
|--------|--------|------------|
| Shared memory centroid | `kernel_assignment` | ~4x lebih cepat dari global mem |
| SoA layout | Semua kernel | Coalesced memory access |
| Partial sum per block | `kernel_partial_sum` | Kurangi atomic ke global |
| CUDA Events | Timing | Akurasi sub-ms |

### Spark
| Fitur | Penggunaan | Keuntungan |
|-------|-----------|------------|
| Broadcast variable | Centroid | Kirim sekali ke semua executor |
| Adaptive QE | `spark.sql.adaptive` | Auto load-balancing partisi |
| Persist/cache | Dataset | Hindari re-read dari disk tiap iterasi |
| MapReduce groupBy | Update centroid | Otomatis terdistribusi |

---

## Troubleshooting

**Error: OpenMP tidak tersedia**
```bash
sudo apt-get install libomp-dev
# Atau: conda install -c conda-forge openmp
```

**Error: nvcc not found**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: CUDA out of memory**
```bash
# Kurangi jumlah titik dengan max_rows
./bin/kmeans_cuda data/nyc_taxi_2023.csv 8 100 5000000  # 5 juta baris
```

**Spark executor OOM**
```bash
# Tambah memory di docker-compose.yml:
# SPARK_WORKER_MEMORY=8g
```
