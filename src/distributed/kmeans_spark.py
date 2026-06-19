"""
================================================================
K-Means Clustering — Versi Distributed (Apache Spark)
================================================================
Platform : Apache Spark (PySpark) + GCP Dataproc / AWS EMR
Dataset  : NYC Taxi Trips (parquet/csv di GCS/S3)
Algoritma: K-Means custom (bukan MLlib) agar bisa kita ukur

Cara jalankan lokal (3 worker simulasi):
  spark-submit --master local[3] kmeans_spark.py \
               --input gs://bucket/nyc_taxi/ \
               --k 8 --max_iter 100 --output gs://bucket/results/

Cara jalankan di cluster (Dataproc):
  gcloud dataproc jobs submit pyspark kmeans_spark.py \
    --cluster=kmeans-cluster \
    --region=us-central1 \
    -- --input gs://bucket/nyc_taxi/ --k 8
================================================================
"""

import argparse
import time
import json
import logging
from typing import List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark import StorageLevel
import numpy as np

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("kmeans_spark")


# ══════════════════════════════════════════════════════════════
# Fungsi jarak Euclidean kuadrat (berjalan di executor, bukan driver)
# ══════════════════════════════════════════════════════════════
def sq_dist(px: float, py: float, cx: float, cy: float) -> float:
    """Hitung jarak kuadrat antara titik (px,py) dan centroid (cx,cy)."""
    return (px - cx) ** 2 + (py - cy) ** 2


# ══════════════════════════════════════════════════════════════
# Assignment: temukan cluster terdekat untuk setiap titik
# Ini berjalan di SETIAP EXECUTOR (node worker) secara paralel.
# centroids di-broadcast agar setiap node punya salinan lokal.
# ══════════════════════════════════════════════════════════════
def assign_cluster(row, centroids_bc):
    """
    Fungsi yang dijalankan di executor untuk setiap baris data.
    centroids_bc: Broadcast variable — dikirim satu kali ke semua executor,
                  disimpan di memory lokal tiap executor (efisien).
    """
    centroids = centroids_bc.value   # ambil dari broadcast
    px, py    = row["pickup_longitude"], row["pickup_latitude"]

    best_dist = float("inf")
    best_k    = 0
    for k, (cx, cy) in enumerate(centroids):
        d = sq_dist(px, py, cx, cy)
        if d < best_dist:
            best_dist = d
            best_k    = k

    return (best_k, px, py, 1)


# ══════════════════════════════════════════════════════════════
# Update centroid: hitung rata-rata per cluster
# Ini menggunakan Spark aggregation — otomatis terdistribusi.
# ══════════════════════════════════════════════════════════════
def update_centroids_spark(assigned_df, K: int) -> List[Tuple[float, float]]:
    """
    Gunakan Spark SQL aggregation untuk hitung centroid baru.
    Spark otomatis melakukan:
      1. Map phase  : setiap executor hitung partial sum lokal
      2. Shuffle    : data dikelompokkan per cluster_id
      3. Reduce phase: jumlahkan semua partial sum
    Ini adalah implementasi MapReduce yang sesungguhnya.
    """
    centroids_df = (
        assigned_df
        .groupBy("cluster_id")
        .agg(
            F.avg("pickup_longitude").alias("cx"),
            F.avg("pickup_latitude").alias("cy"),
            F.count("*").alias("cnt")
        )
        .orderBy("cluster_id")
    )

    rows = centroids_df.collect()
    new_centroids = [(float(r["cx"]), float(r["cy"])) for r in rows]
    counts        = {int(r["cluster_id"]): int(r["cnt"]) for r in rows}
    return new_centroids, counts


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Distributed K-Means via Spark")
    parser.add_argument("--input",    required=True,  help="Path ke dataset (GCS/S3/lokal)")
    parser.add_argument("--output",   default="output_kmeans", help="Path output")
    parser.add_argument("--k",        type=int, default=8,   help="Jumlah cluster")
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterasi")
    parser.add_argument("--n_parts",  type=int, default=None,
                        help="Jumlah partisi (default: auto dari Spark)")
    parser.add_argument("--format",   default="csv", choices=["csv","parquet"],
                        help="Format input")
    args = parser.parse_args()

    # ── Init Spark Session ─────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("KMeans-NYC-Taxi")
        .config("spark.sql.shuffle.partitions", "200")
        # Aktifkan Adaptive Query Execution (load balancing otomatis):
        # Spark akan menyeimbangkan ukuran partisi setelah shuffle,
        # menghindari skew (satu node kebanjiran data, node lain idle)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Logging metrik per stage
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/tmp/spark-events")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    sc = spark.sparkContext
    log.info(f"Spark version: {spark.version}")
    log.info(f"Executors: {sc._jsc.sc().getExecutorMemoryStatus().size()}")

    # ── Load Dataset ────────────────────────────────────────────
    log.info(f"[1/5] Memuat dataset dari: {args.input}")
    t_load = time.time()

    if args.format == "parquet":
        df = spark.read.parquet(args.input)
    else:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(args.input)

    # Pilih & bersihkan kolom yang dibutuhkan
    df = (
        df.select(
            F.col("pickup_longitude").cast(DoubleType()),
            F.col("pickup_latitude").cast(DoubleType())
        )
        .filter(
            (F.col("pickup_longitude").between(-75.0, -73.0)) &
            (F.col("pickup_latitude").between(40.0, 41.5))
        )
        .dropna()
    )

    # Repartisi untuk load balancing yang merata di semua node
    # Aturan: ~128 MB per partisi, atau n_parts jika ditentukan
    if args.n_parts:
        df = df.repartition(args.n_parts)

    # Cache dataset: agar tidak dibaca ulang dari disk di setiap iterasi.
    # StorageLevel.MEMORY_AND_DISK_SER: compress di memory, spill ke disk jika penuh
    df.persist(StorageLevel.MEMORY_AND_DISK_SER)

    n_points = df.count()   # trigger persist
    load_sec = time.time() - t_load
    n_parts  = df.rdd.getNumPartitions()

    log.info(f"  {n_points:,} titik dimuat dalam {load_sec:.1f}s")
    log.info(f"  Jumlah partisi: {n_parts} (≈ {n_points//n_parts:,} titik/partisi)")

    # ── Inisialisasi Centroid (K-Means++ di driver) ────────────
    log.info("[2/5] Inisialisasi centroid K-Means++...")
    # Sampling efisien dari distributed dataset
    sample_fraction = min(1.0, 10_000 / n_points)
    sample = df.sample(fraction=sample_fraction, seed=42).collect()
    sample_pts = [(float(r["pickup_longitude"]), float(r["pickup_latitude"]))
                  for r in sample]

    rng = np.random.default_rng(42)
    K   = args.k

    # Pilih centroid pertama
    idx = rng.integers(0, len(sample_pts))
    centroids: List[Tuple[float, float]] = [sample_pts[idx]]

    # K-Means++ sampling (di driver, pada sample kecil)
    for _ in range(1, K):
        dists = np.array([
            min(sq_dist(px, py, cx, cy) for cx, cy in centroids)
            for px, py in sample_pts
        ])
        dists /= dists.sum()
        chosen = rng.choice(len(sample_pts), p=dists)
        centroids.append(sample_pts[chosen])

    log.info(f"  Centroid awal: {centroids}")

    # ── Iterasi K-Means Terdistribusi ──────────────────────────
    log.info(f"[3/5] Iterasi K-Means ({K} cluster, max {args.max_iter} iter)...")
    metrics = {
        "iterations": [], "wcss": [],
        "iter_time_sec": [], "n_moved": []
    }

    for it in range(args.max_iter):
        t_iter = time.time()

        # ── Broadcast centroid ke semua executor ────────────────
        # Tanpa broadcast: Spark kirim centroid lewat task serialization
        # (sangat lambat untuk K besar atau task banyak).
        # Dengan broadcast: dikirim sekali ke setiap executor, disimpan
        # di memory lokal → tidak ada overhead jaringan per-task.
        centroids_bc = sc.broadcast(centroids)

        # ── Assignment via UDF (User Defined Function) ──────────
        # Spark UDF berjalan di executor secara paralel
        @F.udf(returnType=IntegerType())
        def assign_udf(px, py):
            c = centroids_bc.value
            bd, bk = float("inf"), 0
            for k, (cx, cy) in enumerate(c):
                d = (px-cx)**2 + (py-cy)**2
                if d < bd: bd, bk = d, k
            return bk

        assigned_df = df.withColumn("cluster_id", assign_udf(
            F.col("pickup_longitude"), F.col("pickup_latitude")
        ))

        # Cache assigned untuk reuse di update + WCSS
        assigned_df.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # ── Update centroid (MapReduce aggregation) ─────────────
        new_centroids, counts = update_centroids_spark(assigned_df, K)

        # ── Hitung WCSS (convergence metric) ───────────────────
        @F.udf(returnType=DoubleType())
        def wcss_udf(px, py, k):
            c = centroids_bc.value
            cx, cy = c[k]
            return (px-cx)**2 + (py-cy)**2

        wcss = (
            assigned_df
            .withColumn("sq_dist", wcss_udf(
                F.col("pickup_longitude"),
                F.col("pickup_latitude"),
                F.col("cluster_id")
            ))
            .agg(F.sum("sq_dist"))
            .collect()[0][0]
        )

        # Hitung titik yang berpindah cluster
        if it > 0:
            prev_bc = sc.broadcast(prev_centroids)

            @F.udf(returnType=IntegerType())
            def prev_assign_udf(px, py):
                c = prev_bc.value
                bd, bk = float("inf"), 0
                for k, (cx, cy) in enumerate(c):
                    d = (px-cx)**2 + (py-cy)**2
                    if d < bd: bd, bk = d, k
                return bk

            moved_df = assigned_df.withColumn(
                "prev_cluster", prev_assign_udf(
                    F.col("pickup_longitude"), F.col("pickup_latitude"))
            ).filter(F.col("cluster_id") != F.col("prev_cluster"))
            n_moved = moved_df.count()
        else:
            n_moved = n_points

        iter_sec = time.time() - t_iter
        metrics["iterations"].append(it)
        metrics["wcss"].append(float(wcss) if wcss else 0)
        metrics["iter_time_sec"].append(iter_sec)
        metrics["n_moved"].append(n_moved)

        log.info(f"  Iter {it:3d} | moved={n_moved:8,} | "
                 f"WCSS={wcss:.4e} | {iter_sec:.1f}s")

        assigned_df.unpersist()
        centroids_bc.unpersist()

        # Cek konvergensi
        prev_centroids = centroids
        centroids = new_centroids
        if n_moved == 0:
            log.info(f"  Konvergen pada iterasi {it}!")
            break

    # ── Simpan Hasil ────────────────────────────────────────────
    log.info("[4/5] Menyimpan hasil...")
    centroids_bc_final = sc.broadcast(centroids)

    @F.udf(returnType=IntegerType())
    def final_assign_udf(px, py):
        c = centroids_bc_final.value
        bd, bk = float("inf"), 0
        for k, (cx, cy) in enumerate(c):
            d = (px-cx)**2 + (py-cy)**2
            if d < bd: bd, bk = d, k
        return bk

    result_df = df.withColumn("cluster_id", final_assign_udf(
        F.col("pickup_longitude"), F.col("pickup_latitude")
    ))

    # Simpan sebagai parquet (kompresi snappy, efisien untuk data besar)
    result_df.write.mode("overwrite").parquet(f"{args.output}/assignments")

    # Simpan centroid sebagai JSON
    centroid_data = {
        "k": K,
        "centroids": [{"id": k, "x": cx, "y": cy, "count": counts.get(k, 0)}
                      for k, (cx, cy) in enumerate(centroids)]
    }

    # ── Simpan Metrik Eksperimen ─────────────────────────────────
    log.info("[5/5] Menyimpan metrik...")
    n_nodes = sc._jsc.sc().getExecutorMemoryStatus().size()
    exp_metrics = {
        "impl": "spark",
        "n_nodes": n_nodes,
        "n_partitions": n_parts,
        "K": K,
        "n_points": n_points,
        "iterations": len(metrics["iterations"]),
        "total_time_sec": sum(metrics["iter_time_sec"]),
        "load_time_sec": load_sec,
        "final_wcss": metrics["wcss"][-1] if metrics["wcss"] else None,
        "throughput_pts_sec": n_points * len(metrics["iterations"]) /
                              max(sum(metrics["iter_time_sec"]), 1e-9),
        "per_iter_metrics": metrics,
        "centroids": centroid_data["centroids"]
    }

    print("\n" + "="*50)
    print("HASIL AKHIR K-MEANS SPARK")
    print("="*50)
    print(f"Nodes      : {n_nodes}")
    print(f"Partisi    : {n_parts}")
    print(f"Titik      : {n_points:,}")
    print(f"Iterasi    : {len(metrics['iterations'])}")
    print(f"Total waktu: {sum(metrics['iter_time_sec']):.1f}s")
    print(f"Throughput : {exp_metrics['throughput_pts_sec']:.0f} pts/sec")
    print(f"WCSS final : {metrics['wcss'][-1]:.4e}")
    print("\nCentroid:")
    for c in centroid_data["centroids"]:
        print(f"  Cluster {c['id']}: ({c['x']:.4f}, {c['y']:.4f}) | {c['count']:,} titik")

    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
