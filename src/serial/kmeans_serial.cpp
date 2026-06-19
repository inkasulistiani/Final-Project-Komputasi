/**
 * ============================================================
 * K-Means Clustering — Versi Serial (Baseline)
 * ============================================================
 * Dataset  : NYC Taxi Trips (latitude/longitude pickup points)
 * Algoritma: Lloyd's K-Means (iterasi hingga konvergensi)
 * Kompilasi: g++ -O3 -std=c++17 -o kmeans_serial kmeans_serial.cpp
 * Jalankan : ./kmeans_serial <data.csv> <K> <max_iter>
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdexcept>

// ─── Struktur Data ───────────────────────────────────────────
struct Point {
    double x;   // pickup_longitude
    double y;   // pickup_latitude
};

struct Centroid {
    double x;
    double y;
    int    count;  // jumlah titik yang di-assign ke centroid ini
};

// ─── Utility: hitung jarak Euclidean kuadrat ─────────────────
// Kita pakai kuadrat agar tidak perlu sqrt (lebih cepat, hasil ranking sama)
inline double dist_sq(const Point& p, const Centroid& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx * dx + dy * dy;
}

// ─── Load Data dari CSV ──────────────────────────────────────
// Format CSV: trip_id, ..., pickup_longitude, pickup_latitude, ...
// Kita ambil kolom 5 (longitude) dan 6 (latitude) dari NYC TLC dataset
std::vector<Point> load_csv(const std::string& path,
                             int col_x = 5,
                             int col_y = 6,
                             long max_rows = -1)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Tidak bisa membuka file: " + path);
    }

    std::vector<Point> points;
    std::string line;

    // skip header
    std::getline(file, line);

    long row_count = 0;
    while (std::getline(file, line)) {
        if (max_rows > 0 && row_count >= max_rows) break;

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> cols;

        while (std::getline(ss, token, ',')) {
            cols.push_back(token);
        }

        // validasi ukuran kolom
        int max_col = std::max(col_x, col_y);
        if ((int)cols.size() <= max_col) continue;

        try {
            double px = std::stod(cols[col_x]);
            double py = std::stod(cols[col_y]);

            // filter nilai tidak valid (NYC approx bounds)
            if (px < -75.0 || px > -73.0) continue;
            if (py <  40.0 || py >  41.5) continue;

            points.push_back({px, py});
        } catch (...) {
            // skip baris yang tidak bisa di-parse
            continue;
        }

        ++row_count;
    }

    return points;
}

// ─── Inisialisasi Centroid dengan K-Means++ ──────────────────
// K-Means++ memilih centroid awal secara cerdas agar konvergensi lebih cepat
// daripada inisialisasi acak biasa
std::vector<Centroid> init_centroids_pp(const std::vector<Point>& points, int K)
{
    std::mt19937_64 rng(42);  // seed tetap agar reproducible
    std::vector<Centroid> centroids;
    centroids.reserve(K);

    // Pilih centroid pertama secara acak
    std::uniform_int_distribution<size_t> idx_dist(0, points.size() - 1);
    size_t first = idx_dist(rng);
    centroids.push_back({points[first].x, points[first].y, 0});

    // Pilih centroid berikutnya dengan probabilitas proporsional terhadap
    // jarak kuadrat ke centroid terdekat yang sudah dipilih
    for (int k = 1; k < K; ++k) {
        std::vector<double> d2(points.size());
        double total = 0.0;

        for (size_t i = 0; i < points.size(); ++i) {
            double min_d2 = std::numeric_limits<double>::max();
            for (const auto& c : centroids) {
                double d = dist_sq(points[i], c);
                min_d2 = std::min(min_d2, d);
            }
            d2[i] = min_d2;
            total += min_d2;
        }

        // Sampling berdasarkan distribusi jarak kuadrat
        std::uniform_real_distribution<double> u(0.0, total);
        double target = u(rng);
        double cumul  = 0.0;
        size_t chosen = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            cumul += d2[i];
            if (cumul >= target) { chosen = i; break; }
        }
        centroids.push_back({points[chosen].x, points[chosen].y, 0});
    }

    return centroids;
}

// ─── Satu Iterasi K-Means ─────────────────────────────────────
// Kembalikan jumlah titik yang berpindah cluster (untuk cek konvergensi)
int kmeans_iteration(const std::vector<Point>&   points,
                     std::vector<Centroid>&        centroids,
                     std::vector<int>&             assignments)
{
    int K = (int)centroids.size();
    int moved = 0;

    // Akumulator sementara: sum_x, sum_y, count per centroid
    std::vector<double> sum_x(K, 0.0);
    std::vector<double> sum_y(K, 0.0);
    std::vector<int>    cnt(K, 0);

    // ── ASSIGNMENT STEP ──────────────────────────────────────
    // Untuk setiap titik, cari centroid terdekat
    for (size_t i = 0; i < points.size(); ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int    best_k    = 0;

        for (int k = 0; k < K; ++k) {
            double d = dist_sq(points[i], centroids[k]);
            if (d < best_dist) {
                best_dist = d;
                best_k    = k;
            }
        }

        if (assignments[i] != best_k) {
            assignments[i] = best_k;
            ++moved;
        }

        // Akumulasi untuk update centroid
        sum_x[best_k] += points[i].x;
        sum_y[best_k] += points[i].y;
        cnt[best_k]   += 1;
    }

    // ── UPDATE STEP ──────────────────────────────────────────
    // Hitung posisi centroid baru sebagai rata-rata cluster
    for (int k = 0; k < K; ++k) {
        if (cnt[k] > 0) {
            centroids[k].x     = sum_x[k] / cnt[k];
            centroids[k].y     = sum_y[k] / cnt[k];
            centroids[k].count = cnt[k];
        }
        // Jika cluster kosong, biarkan centroid di tempat (edge case)
    }

    return moved;
}

// ─── Hitung Within-Cluster Sum of Squares (WCSS) ─────────────
// Metrik kualitas clustering: semakin kecil semakin baik
double compute_wcss(const std::vector<Point>&   points,
                    const std::vector<Centroid>& centroids,
                    const std::vector<int>&      assignments)
{
    double wcss = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        wcss += dist_sq(points[i], centroids[assignments[i]]);
    }
    return wcss;
}

// ─── Simpan Hasil ke CSV ──────────────────────────────────────
void save_results(const std::string&           out_path,
                  const std::vector<Point>&    points,
                  const std::vector<int>&      assignments,
                  const std::vector<Centroid>& centroids)
{
    std::ofstream f(out_path);
    f << "x,y,cluster,centroid_x,centroid_y\n";
    for (size_t i = 0; i < points.size(); ++i) {
        int k = assignments[i];
        f << std::fixed << std::setprecision(6)
          << points[i].x << ","
          << points[i].y << ","
          << k << ","
          << centroids[k].x << ","
          << centroids[k].y << "\n";
    }
    std::cout << "[INFO] Hasil disimpan ke: " << out_path << "\n";
}

// ─── Simpan Metrik Eksperimen ─────────────────────────────────
void save_metrics(const std::string& metrics_path,
                  int    K,
                  int    n_threads,
                  long   n_points,
                  int    iterations,
                  double elapsed_ms,
                  double wcss,
                  double throughput_pts_sec)
{
    std::ofstream f(metrics_path, std::ios::app);
    // Tulis header jika file baru
    if (f.tellp() == 0) {
        f << "impl,K,threads,n_points,iterations,elapsed_ms,wcss,throughput_pts_sec\n";
    }
    f << "serial,"
      << K << ","
      << n_threads << ","
      << n_points << ","
      << iterations << ","
      << std::fixed << std::setprecision(2) << elapsed_ms << ","
      << std::scientific << std::setprecision(4) << wcss << ","
      << std::fixed << std::setprecision(0) << throughput_pts_sec << "\n";
}

// ─── MAIN ─────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse argumen ──────────────────────────────────────────
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.csv> [K=8] [max_iter=100] [max_rows=-1]\n";
        return 1;
    }

    std::string data_path = argv[1];
    int    K        = (argc > 2) ? std::stoi(argv[2]) : 8;
    int    max_iter = (argc > 3) ? std::stoi(argv[3]) : 100;
    long   max_rows = (argc > 4) ? std::stol(argv[4]) : -1;

    std::cout << "========================================\n";
    std::cout << "  K-Means Clustering — Serial Baseline\n";
    std::cout << "========================================\n";
    std::cout << "Dataset  : " << data_path << "\n";
    std::cout << "K        : " << K << "\n";
    std::cout << "Max iter : " << max_iter << "\n";
    if (max_rows > 0) std::cout << "Max rows : " << max_rows << "\n";
    std::cout << "----------------------------------------\n";

    // ── Load data ──────────────────────────────────────────────
    std::cout << "[1/4] Memuat dataset...\n";
    auto t_load_start = std::chrono::high_resolution_clock::now();

    std::vector<Point> points;
    try {
        points = load_csv(data_path, 5, 6, max_rows);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(
                         t_load_end - t_load_start).count();

    std::cout << "[1/4] " << points.size()
              << " titik dimuat dalam " << std::fixed << std::setprecision(1)
              << load_ms << " ms\n";

    if (points.empty()) {
        std::cerr << "[ERROR] Tidak ada data valid. Periksa format CSV.\n";
        return 1;
    }

    // ── Inisialisasi centroid ──────────────────────────────────
    std::cout << "[2/4] Inisialisasi centroid (K-Means++)...\n";
    auto centroids = init_centroids_pp(points, K);

    // Assignment awal
    std::vector<int> assignments(points.size(), 0);

    // ── Iterasi K-Means ────────────────────────────────────────
    std::cout << "[3/4] Menjalankan K-Means (" << K << " cluster)...\n";
    auto t_kmeans_start = std::chrono::high_resolution_clock::now();

    int iter   = 0;
    int moved  = (int)points.size();  // paksa masuk loop pertama

    while (iter < max_iter && moved > 0) {
        moved = kmeans_iteration(points, centroids, assignments);

        if (iter % 10 == 0 || moved == 0) {
            double wcss_now = compute_wcss(points, centroids, assignments);
            std::cout << "  Iterasi " << std::setw(3) << iter
                      << " | moved=" << std::setw(7) << moved
                      << " | WCSS=" << std::scientific << std::setprecision(4)
                      << wcss_now << "\n";
        }
        ++iter;
    }

    auto t_kmeans_end = std::chrono::high_resolution_clock::now();
    double kmeans_ms = std::chrono::duration<double, std::milli>(
                           t_kmeans_end - t_kmeans_start).count();

    // ── Hitung metrik final ────────────────────────────────────
    double wcss = compute_wcss(points, centroids, assignments);
    double throughput = (double)points.size() * iter / (kmeans_ms / 1000.0);

    std::cout << "----------------------------------------\n";
    std::cout << "[4/4] Selesai!\n";
    std::cout << "  Iterasi total   : " << iter << "\n";
    std::cout << "  Konvergensi     : " << (moved == 0 ? "Ya" : "Tidak (max iter)") << "\n";
    std::cout << "  Waktu K-Means   : " << std::fixed << std::setprecision(1)
              << kmeans_ms << " ms\n";
    std::cout << "  Waktu total     : " << std::fixed << std::setprecision(1)
              << (load_ms + kmeans_ms) << " ms\n";
    std::cout << "  Throughput      : " << std::fixed << std::setprecision(0)
              << throughput << " pts/sec\n";
    std::cout << "  WCSS final      : " << std::scientific << std::setprecision(4)
              << wcss << "\n";

    std::cout << "\n  Ukuran cluster:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "    Cluster " << k << ": " << centroids[k].count
                  << " titik  @ (" << std::fixed << std::setprecision(4)
                  << centroids[k].x << ", " << centroids[k].y << ")\n";
    }

    // ── Simpan output ──────────────────────────────────────────
    save_results("results_serial.csv", points, assignments, centroids);
    save_metrics("metrics.csv", K, 1, (long)points.size(), iter,
                 kmeans_ms, wcss, throughput);

    return 0;
}
