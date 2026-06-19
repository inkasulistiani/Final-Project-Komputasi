#!/bin/bash
# ============================================================
# entrypoint_cpu.sh — Entry point container CPU/OpenMP
# ============================================================
set -euo pipefail

echo "============================================"
echo "  K-Means CPU/OpenMP Container"
echo "============================================"
echo "OMP_NUM_THREADS : ${OMP_NUM_THREADS:-4}"
echo "K               : ${KMEANS_K:-8}"
echo "Max Iter        : ${KMEANS_MAX_ITER:-100}"
echo "Data            : ${DATA_PATH:-/data/nyc_taxi.csv}"
echo "--------------------------------------------"

# Tunggu data tersedia
until [ -f "${DATA_PATH:-/data/nyc_taxi.csv}" ]; do
    echo "[WAIT] Menunggu dataset di ${DATA_PATH}..."
    sleep 5
done

# Jalankan Serial (baseline)
echo ""
echo "=== [1/2] Serial Baseline ==="
kmeans_serial "${DATA_PATH}" "${KMEANS_K:-8}" "${KMEANS_MAX_ITER:-100}" \
    2>&1 | tee /results/serial_output.txt

# Jalankan OpenMP dengan jumlah thread dari env
echo ""
echo "=== [2/2] OpenMP (${OMP_NUM_THREADS:-4} thread) ==="
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
kmeans_omp "${DATA_PATH}" "${KMEANS_K:-8}" "${KMEANS_MAX_ITER:-100}" \
    2>&1 | tee /results/omp_output.txt

echo ""
echo "✓ Selesai. Hasil di /results/"
