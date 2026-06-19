#!/usr/bin/env python3
"""
================================================================
analyze_results.py — Analisis & Visualisasi Hasil Eksperimen
================================================================
Baca metrics.csv, hitung speedup, efficiency, dan buat grafik.

Jalankan: python3 scripts/analyze_results.py metrics.csv
================================================================
"""

import sys
import csv
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib tidak tersedia. Install: pip install matplotlib")
    print("       Analisis teks tetap berjalan.\n")


def load_metrics(path: str):
    """Baca metrics.csv dan kelompokkan per implementasi."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "impl":       row["impl"],
                "K":          int(row["K"]),
                "threads":    row["threads"],
                "n_points":   int(row["n_points"]),
                "iterations": int(row["iterations"]),
                "elapsed_ms": float(row["elapsed_ms"]),
                "wcss":       float(row["wcss"]),
                "throughput": float(row["throughput_pts_sec"]),
            })
    return rows


def compute_speedup(baseline_ms: float, times: list) -> list:
    """Hitung speedup = T_baseline / T_parallel."""
    return [baseline_ms / t if t > 0 else 0 for t in times]


def print_table(title: str, headers: list, rows: list):
    """Cetak tabel teks yang rapi."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-"*w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:{w}}}" for w in col_widths) + " |"

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def analyze_strong_scaling(rows: list):
    """Analisis strong scaling: speedup saat N tetap, thread bertambah."""
    serial_rows = [r for r in rows if r["impl"] == "serial"]
    omp_rows    = [r for r in rows if r["impl"] == "openmp"]

    if not serial_rows or not omp_rows:
        print("[INFO] Data serial/openmp belum ada. Jalankan: make experiment_strong")
        return

    # Baseline: serial 1 thread, N terbesar
    serial_row   = max(serial_rows, key=lambda r: r["n_points"])
    baseline_ms  = serial_row["elapsed_ms"]
    N            = serial_row["n_points"]

    # Kelompokkan OMP berdasarkan jumlah thread
    by_thread = defaultdict(list)
    for r in omp_rows:
        if r["n_points"] == N:
            by_thread[r["threads"]].append(r["elapsed_ms"])

    table_rows = []
    speedups   = []
    for t in sorted(by_thread.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        avg_ms   = sum(by_thread[t]) / len(by_thread[t])
        speedup  = baseline_ms / avg_ms
        efficiency = speedup / int(t) * 100 if t.isdigit() else 0
        tp       = N * serial_row["iterations"] / (avg_ms / 1000)
        speedups.append((int(t) if t.isdigit() else 0, speedup))
        table_rows.append([
            t,
            f"{avg_ms:.1f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%",
            f"{tp:.0f}"
        ])

    # Serial sebagai baris pertama
    table_rows.insert(0, [
        "1 (serial)",
        f"{baseline_ms:.1f}",
        "1.00x",
        "100.0%",
        f"{N * serial_row['iterations'] / (baseline_ms/1000):.0f}"
    ])

    print_table(
        f"STRONG SCALING — N={N:,} titik",
        ["Threads", "Waktu(ms)", "Speedup", "Efficiency", "Throughput(pts/s)"],
        table_rows
    )

    # Hukum Amdahl: estimasi bagian serial
    if len(speedups) >= 2:
        # Gunakan speedup tertinggi untuk estimasi p (bagian paralel)
        max_speedup = max(s for _, s in speedups)
        max_p       = max(t for t, _ in speedups)
        # Amdahl: S = 1 / (1-p + p/n) → p = (1 - 1/S) / (1 - 1/n)
        p_est = (1 - 1/max_speedup) / (1 - 1/max_p) if max_p > 1 else 0
        print(f"\n  Estimasi bagian paralel (Hukum Amdahl): {p_est*100:.1f}%")
        print(f"  Speedup teoritis maksimal: {1/(1-p_est):.1f}x" if p_est < 1 else "")

    return speedups


def analyze_weak_scaling(rows: list):
    """Analisis weak scaling: efficiency saat N & thread bertambah proporsional."""
    omp_rows = [r for r in rows if r["impl"] == "openmp"]
    if not omp_rows:
        return

    # Baseline: 1 thread, N terkecil
    single = [r for r in omp_rows if r["threads"] == "1"]
    if not single:
        return
    baseline_ms = min(r["elapsed_ms"] for r in single)

    table_rows = []
    for r in sorted(omp_rows, key=lambda x: x["n_points"]):
        t   = int(r["threads"]) if r["threads"].isdigit() else 1
        eff = baseline_ms / r["elapsed_ms"] * 100
        table_rows.append([
            r["threads"],
            f"{r['n_points']:,}",
            f"{r['elapsed_ms']:.1f}",
            f"{eff:.1f}%"
        ])

    print_table(
        "WEAK SCALING — N bertambah proporsional dengan thread",
        ["Threads", "N (titik)", "Waktu(ms)", "Efficiency"],
        table_rows
    )

def plot_results(rows: list, out_dir: str = "experiments/results"):
    """Buat grafik PNG jika matplotlib tersedia."""
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(out_dir, exist_ok=True)

    # ── Grafik 1: Strong Scaling ─────────────────────────────
    serial_rows = [r for r in rows if r["impl"] == "serial"]
    omp_rows    = [r for r in rows if r["impl"] == "openmp"]

    if serial_rows and omp_rows:
        baseline_ms = max(serial_rows, key=lambda r: r["n_points"])["elapsed_ms"]
        by_thread = {}
        for r in omp_rows:
            t = r["threads"]
            if t not in by_thread:
                by_thread[t] = []
            by_thread[t].append(r["elapsed_ms"])

        threads  = sorted([int(t) for t in by_thread if t.isdigit()])
        speedups = [baseline_ms / (sum(by_thread[str(t)]) / len(by_thread[str(t)]))
                    for t in threads]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Strong Scaling — K-Means OpenMP", fontsize=14, fontweight="bold")

        # Speedup chart
        ax = axes[0]
        ax.plot([1] + threads, [1.0] + speedups, "b-o", linewidth=2,
                markersize=8, label="OpenMP actual")
        ax.plot([1] + threads, [1.0] + [float(t) for t in threads],
                "r--", alpha=0.5, label="Ideal (linear)")
        ax.set_xlabel("Jumlah Thread")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1] + threads)

        # Efficiency chart
        ax = axes[1]
        efficiencies = [s / t * 100 for s, t in zip(speedups, threads)]
        ax.plot(threads, efficiencies, "g-s", linewidth=2, markersize=8)
        ax.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="Ideal (100%)")
        ax.set_xlabel("Jumlah Thread")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Parallel Efficiency")
        ax.set_ylim(0, 120)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(threads)

        plt.tight_layout()
        path = f"{out_dir}/strong_scaling.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Grafik: {path}")

    # ── Grafik 2: Throughput Comparison ─────────────────────
    impls      = list(set(r["impl"] for r in rows))
    throughputs = [max(r["throughput"] for r in rows if r["impl"] == impl)
                   for impl in impls]

    if len(impls) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"serial": "#888", "openmp": "#2196F3", "cuda": "#4CAF50",
                  "spark": "#FF9800"}
        bars = ax.bar(impls, [tp/1e6 for tp in throughputs],
                      color=[colors.get(i, "#999") for i in impls])
        ax.set_ylabel("Throughput (Juta titik/detik)")
        ax.set_title("Perbandingan Throughput Semua Implementasi")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, tp in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{tp/1e6:.2f}M", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        path = f"{out_dir}/throughput_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Grafik: {path}")

    print(f"\n  Semua grafik disimpan di: {out_dir}/")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"

    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        print("  Jalankan eksperimen dulu: make experiment")
        sys.exit(1)

    rows = load_metrics(path)
    print(f"Total run: {len(rows)}")
    print(f"Implementasi: {set(r['impl'] for r in rows)}")

    analyze_strong_scaling(rows)
    analyze_weak_scaling(rows)

    if HAS_MATPLOTLIB:
        print("\n=== Membuat Grafik ===")
        plot_results(rows)
    else:
        print("\n[INFO] Install matplotlib untuk grafik: pip install matplotlib pandas")


if __name__ == "__main__":
    main()
