/**
 * ============================================================
 * K-Means Clustering — Versi OpenMP (Paralel CPU)
 * ============================================================
 * Kompilasi: g++ -O3 -std=c++17 -fopenmp -o kmeans_omp kmeans_omp.cpp
 * Jalankan : OMP_NUM_THREADS=8 ./kmeans_omp <data.csv> <K> <max_iter>
 *
 * Pola OpenMP yang digunakan:
 *   1. #pragma omp parallel for  — paralel loop assignment & update
 *   2. reduction(+:...)          — akumulasi sum centroid tanpa race condition
 *   3. #pragma omp atomic        — increment counter cluster (sinkronisasi)
 *   4. #pragma omp barrier       — sinkronisasi antar fase assignment/update
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <omp.h>   // ← header OpenMP

// ─── Struktur Data ────────────────────────────────────────────
struct Point    { double x, y; };
struct Centroid { double x, y; int count; };

inline double dist_sq(const Point& p, const Centroid& c) {
    double dx = p.x - c.x, dy = p.y - c.y;
    return dx*dx + dy*dy;
}

// ─── Load CSV (sama dengan serial) ───────────────────────────
std::vector<Point> load_csv(const std::string& path, int col_x=5,
                             int col_y=6, long max_rows=-1)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Tidak bisa membuka file: " + path);

    std::vector<Point> points;
    std::string line;
    std::getline(file, line); // skip header

    long row = 0;
    while (std::getline(file, line)) {
        if (max_rows > 0 && row >= max_rows) break;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);

        int mc = std::max(col_x, col_y);
        if ((int)cols.size() <= mc) continue;
        try {
            double px = std::stod(cols[col_x]);
            double py = std::stod(cols[col_y]);
            if (px < -75.0 || px > -73.0) continue;
            if (py <  40.0 || py >  41.5) continue;
            points.push_back({px, py});
        } catch (...) { continue; }
        ++row;
    }
    return points;
}

// ─── Inisialisasi K-Means++ ───────────────────────────────────
std::vector<Centroid> init_centroids_pp(const std::vector<Point>& pts, int K)
{
    std::mt19937_64 rng(42);
    std::vector<Centroid> centroids;
    centroids.reserve(K);

    std::uniform_int_distribution<size_t> idx_dist(0, pts.size()-1);
    size_t first = idx_dist(rng);
    centroids.push_back({pts[first].x, pts[first].y, 0});

    for (int k = 1; k < K; ++k) {
        std::vector<double> d2(pts.size());
        double total = 0.0;

        // ══ OpenMP Pola 1: parallel for + reduction ══════════
        // Hitung jarak ke centroid terdekat untuk setiap titik secara paralel.
        // 'reduction(+:total)' memastikan akumulasi total aman dari race condition:
        // setiap thread punya salinan lokal 'total', lalu dijumlahkan di akhir.
        #pragma omp parallel for reduction(+:total) schedule(static)
        for (size_t i = 0; i < pts.size(); ++i) {
            double min_d2 = std::numeric_limits<double>::max();
            for (const auto& c : centroids) {
                double d = dist_sq(pts[i], c);
                min_d2 = std::min(min_d2, d);
            }
            d2[i]  = min_d2;
            total += min_d2;
        }

        // Sampling sekuensial (bergantung pada total kumulatif)
        std::uniform_real_distribution<double> u(0.0, total);
        double target = u(rng), cumul = 0.0;
        size_t chosen = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            cumul += d2[i];
            if (cumul >= target) { chosen = i; break; }
        }
        centroids.push_back({pts[chosen].x, pts[chosen].y, 0});
    }
    return centroids;
}

// ─── Iterasi K-Means Paralel ──────────────────────────────────
int kmeans_iteration_omp(const std::vector<Point>&  pts,
                          std::vector<Centroid>&      centroids,
                          std::vector<int>&           assignments)
{
    int K  = (int)centroids.size();
    int N  = (int)pts.size();

    // Buffer akumulasi per centroid
    std::vector<double> sum_x(K, 0.0);
    std::vector<double> sum_y(K, 0.0);
    std::vector<int>    cnt(K, 0);

    int moved = 0;

    // ══ OpenMP Pola 1 + 2 + 3: parallel for, reduction, atomic ═
    //
    // - 'reduction(+:moved)' : tiap thread hitung moved lokal, dijumlah akhir
    // - Di dalam loop, kita TIDAK bisa pakai reduction untuk array sum_x/sum_y
    //   karena OpenMP reduction standar tidak support array.
    //   Solusi: kita gunakan pendekatan thread-private buffer.
    //
    // Thread-private buffer: setiap thread punya salinan local_sum_x,
    // local_sum_y, local_cnt → tidak ada race condition pada array.
    // Di akhir parallel region, kita merge semua buffer.

    int n_threads = omp_get_max_threads();

    // Buffer per-thread: [thread_id][cluster_id]
    std::vector<std::vector<double>> thr_sum_x(n_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> thr_sum_y(n_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<int>>    thr_cnt  (n_threads, std::vector<int>   (K, 0));

    // ── ASSIGNMENT STEP (Paralel) ─────────────────────────────
    #pragma omp parallel reduction(+:moved)
    {
        int tid = omp_get_thread_num();

        // schedule(static) = bagi range [0,N) merata ke semua thread
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            double best_dist = std::numeric_limits<double>::max();
            int    best_k    = 0;

            for (int k = 0; k < K; ++k) {
                double d = dist_sq(pts[i], centroids[k]);
                if (d < best_dist) { best_dist = d; best_k = k; }
            }

            if (assignments[i] != best_k) {
                assignments[i] = best_k;
                ++moved;   // ← aman: reduction variable
            }

            // Akumulasi ke buffer thread-private (tanpa kunci)
            thr_sum_x[tid][best_k] += pts[i].x;
            thr_sum_y[tid][best_k] += pts[i].y;
            thr_cnt  [tid][best_k] += 1;
        }

        // ══ OpenMP Pola 4: barrier ════════════════════════════
        // Semua thread harus selesai assignment sebelum merge buffer.
        // Barrier implisit ada di akhir #pragma omp for, tapi kita
        // tulis eksplisit untuk kejelasan dokumentasi.
        #pragma omp barrier

        // ── MERGE BUFFER ke sum_x/sum_y/cnt (Paralel per cluster) ─
        // Bagi K cluster ke thread-thread yang tersedia untuk merge.
        #pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) {
            double sx = 0.0, sy = 0.0;
            int    sc = 0;
            for (int t = 0; t < n_threads; ++t) {
                sx += thr_sum_x[t][k];
                sy += thr_sum_y[t][k];
                sc += thr_cnt  [t][k];
            }
            sum_x[k] = sx;
            sum_y[k] = sy;
            cnt  [k] = sc;
        }
    } // ← akhir parallel region

    // ── UPDATE STEP (Sekuensial, K kecil) ────────────────────
    for (int k = 0; k < K; ++k) {
        if (cnt[k] > 0) {
            centroids[k].x     = sum_x[k] / cnt[k];
            centroids[k].y     = sum_y[k] / cnt[k];
            centroids[k].count = cnt[k];
        }
    }

    return moved;
}

// ─── WCSS ─────────────────────────────────────────────────────
double compute_wcss_omp(const std::vector<Point>&   pts,
                         const std::vector<Centroid>& centroids,
                         const std::vector<int>&      assignments)
{
    double wcss = 0.0;
    // ══ reduction untuk sum WCSS ══════════════════════════════
    #pragma omp parallel for reduction(+:wcss) schedule(static)
    for (size_t i = 0; i < pts.size(); ++i) {
        wcss += dist_sq(pts[i], centroids[assignments[i]]);
    }
    return wcss;
}

// ─── Simpan Metrik ────────────────────────────────────────────
void save_metrics(const std::string& path, int K, int n_threads,
                  long n_pts, int iters, double ms,
                  double wcss, double throughput)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "impl,K,threads,n_points,iterations,elapsed_ms,wcss,throughput_pts_sec\n";
    f << "openmp," << K << "," << n_threads << "," << n_pts << ","
      << iters << "," << std::fixed << std::setprecision(2) << ms << ","
      << std::scientific << std::setprecision(4) << wcss << ","
      << std::fixed << std::setprecision(0) << throughput << "\n";
}

// ─── MAIN ─────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.csv> [K=8] [max_iter=100] [max_rows=-1]\n";
        return 1;
    }

    std::string path = argv[1];
    int  K        = (argc > 2) ? std::stoi(argv[2]) : 8;
    int  max_iter = (argc > 3) ? std::stoi(argv[3]) : 100;
    long max_rows = (argc > 4) ? std::stol(argv[4]) : -1;

    int n_threads = omp_get_max_threads();

    std::cout << "========================================\n";
    std::cout << "  K-Means Clustering — OpenMP\n";
    std::cout << "========================================\n";
    std::cout << "Dataset  : " << path << "\n";
    std::cout << "K        : " << K << "\n";
    std::cout << "Threads  : " << n_threads << "\n";
    std::cout << "Max iter : " << max_iter << "\n";

    // ── Load data ──────────────────────────────────────────────
    std::cout << "\n[1/4] Memuat dataset...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Point> pts;
    try { pts = load_csv(path, 5, 6, max_rows); }
    catch (const std::exception& e) { std::cerr << e.what(); return 1; }

    double load_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  " << pts.size() << " titik dimuat (" << load_ms << " ms)\n";

    // ── Init centroid ──────────────────────────────────────────
    std::cout << "[2/4] Inisialisasi K-Means++...\n";
    auto centroids  = init_centroids_pp(pts, K);
    std::vector<int> assignments(pts.size(), 0);

    // ── Iterasi ────────────────────────────────────────────────
    std::cout << "[3/4] Iterasi K-Means (OpenMP, " << n_threads << " thread)...\n";
    auto t_km = std::chrono::high_resolution_clock::now();

    int iter = 0, moved = (int)pts.size();
    while (iter < max_iter && moved > 0) {
        moved = kmeans_iteration_omp(pts, centroids, assignments);
        if (iter % 10 == 0 || moved == 0) {
            double w = compute_wcss_omp(pts, centroids, assignments);
            std::cout << "  Iter " << std::setw(3) << iter
                      << " | moved=" << std::setw(7) << moved
                      << " | WCSS=" << std::scientific << std::setprecision(4) << w << "\n";
        }
        ++iter;
    }

    double km_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_km).count();

    double wcss = compute_wcss_omp(pts, centroids, assignments);
    double tp   = (double)pts.size() * iter / (km_ms / 1000.0);

    std::cout << "----------------------------------------\n";
    std::cout << "[4/4] Selesai!\n";
    std::cout << "  Iterasi     : " << iter << "\n";
    std::cout << "  Waktu       : " << std::fixed << std::setprecision(1) << km_ms << " ms\n";
    std::cout << "  Throughput  : " << std::fixed << std::setprecision(0) << tp << " pts/sec\n";
    std::cout << "  WCSS final  : " << std::scientific << std::setprecision(4) << wcss << "\n";
    std::cout << "\n  Ukuran cluster:\n";
    for (int k = 0; k < K; ++k)
        std::cout << "    Cluster " << k << ": " << centroids[k].count << " titik\n";

    save_metrics("metrics.csv", K, n_threads, (long)pts.size(), iter,
                 km_ms, wcss, tp);
    return 0;
}
