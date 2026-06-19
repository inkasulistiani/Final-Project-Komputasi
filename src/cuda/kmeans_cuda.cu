/**
 * ============================================================
 * K-Means Clustering — Versi CUDA (GPU)
 * ============================================================
 * Kompilasi: nvcc -O3 -arch=sm_75 --ptxas-options=-v \
 *                 -o kmeans_cuda kmeans_cuda.cu
 * Jalankan : ./kmeans_cuda <data.csv> <K> <max_iter>
 *
 * Optimasi CUDA:
 *   1. Shared Memory  : centroid di-load ke __shared__ agar akses cepat
 *   2. Coalesced Access: data titik disimpan sebagai SoA (struct of arrays)
 *   3. Warp Reduction : digunakan saat merge partial sums
 *   4. Occupancy     : dianalisis dengan --ptxas-options=-v
 * ============================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <stdexcept>

// ─── Makro error-check CUDA ───────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ─── Konstanta ────────────────────────────────────────────────
static const int BLOCK_SIZE    = 256;   // thread per block (optimal untuk sm_75)
static const int MAX_K         = 64;    // batas K untuk shared memory centroid

// ─── Struct data di host ──────────────────────────────────────
struct Point    { double x, y; };
struct Centroid { double x, y; int count; };

// ═══════════════════════════════════════════════════════════════
// KERNEL 1: Assignment — cari centroid terdekat untuk setiap titik
// ═══════════════════════════════════════════════════════════════
//
// Strategi Shared Memory:
// - Setiap block GPU load SEMUA centroid ke __shared__ memory satu kali
// - Setiap thread lalu baca centroid dari shared (cepat, ~100 cycle)
//   bukan dari global memory (lambat, ~400-800 cycle)
// - Ini mengurangi global memory bandwidth drastis ketika K besar
//
// Layout memori (SoA — Structure of Arrays, bukan AoS):
// - d_px[i] = koordinat x titik ke-i   → coalesced access
// - d_py[i] = koordinat y titik ke-i   → coalesced access
// - Jika AoS (x,y,x,y,...), thread 0 baca byte 0, thread 1 baca byte 16
//   (stride 2), tidak coalesced. SoA: thread 0 baca byte 0, thread 1 byte 8
//   → coalesced, DRAM burst bisa dipakai penuh
// ═══════════════════════════════════════════════════════════════
__global__ void kernel_assignment(
    const double* __restrict__ d_px,      // koordinat x semua titik
    const double* __restrict__ d_py,      // koordinat y semua titik
    int*          __restrict__ d_assign,  // hasil assignment
    const double* __restrict__ d_cx,      // centroid x (di global mem)
    const double* __restrict__ d_cy,      // centroid y (di global mem)
    int N,  // jumlah titik
    int K   // jumlah centroid
)
{
    // ── Load centroid ke shared memory ────────────────────────
    // Setiap thread dalam block berpartisipasi mengisi shared memory.
    // __shared__ berarti satu buffer per block (bukan per thread).
    __shared__ double s_cx[MAX_K];
    __shared__ double s_cy[MAX_K];

    // threadIdx.x = indeks thread dalam block
    // blockDim.x  = jumlah thread per block (= BLOCK_SIZE = 256)
    // Jika K <= blockDim.x, setiap thread mengisi satu entri:
    if (threadIdx.x < K) {
        s_cx[threadIdx.x] = d_cx[threadIdx.x];
        s_cy[threadIdx.x] = d_cy[threadIdx.x];
    }

    // __syncthreads(): barrier — tunggu sampai SEMUA thread dalam block
    // selesai menulis ke shared memory sebelum ada yang membaca.
    __syncthreads();

    // ── Cari centroid terdekat ─────────────────────────────────
    // Global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;  // guard: thread ekstra di luar range data

    double px = d_px[i];
    double py = d_py[i];

    double best_dist = 1e300;
    int    best_k    = 0;

    // Bandingkan dengan semua centroid — baca dari shared memory (cepat)
    for (int k = 0; k < K; ++k) {
        double dx = px - s_cx[k];
        double dy = py - s_cy[k];
        double d  = dx*dx + dy*dy;
        if (d < best_dist) { best_dist = d; best_k = k; }
    }

    d_assign[i] = best_k;
}

// ═══════════════════════════════════════════════════════════════
// KERNEL 2: Partial Sum — hitung partial sum per block
// ═══════════════════════════════════════════════════════════════
//
// Mengapa partial sum, bukan langsung atomic global?
// - Jika N = 10M titik dan K = 8, ada 80M atomic operasi → bottleneck
// - Solusi: tiap block hitung partial sum-nya sendiri (di shared mem),
//   lalu tulis 1 entri per block ke global. Atomic hanya K * gridDim kali.
//
// Warp-level reduction menggunakan __shfl_down_sync (shuffle instruksi):
// - Komunikasi langsung antar thread dalam warp (32 thread) via register
// - Tidak perlu shared memory, tidak perlu __syncthreads dalam warp
// ═══════════════════════════════════════════════════════════════
__global__ void kernel_partial_sum(
    const double* __restrict__ d_px,
    const double* __restrict__ d_py,
    const int*    __restrict__ d_assign,
    double*       d_partial_sx,   // output: partial sum x [K * gridDim.x]
    double*       d_partial_sy,   // output: partial sum y [K * gridDim.x]
    int*          d_partial_cnt,  // output: partial count  [K * gridDim.x]
    int N,
    int K
)
{
    // Shared memory untuk akumulasi per-cluster dalam satu block
    // Ukuran dinamis ditentukan saat launch: <<<grid, block, shm_bytes>>>
    extern __shared__ double s_buf[];
    // Layout: [K] sx, [K] sy, [K] cnt (sebagai double)
    double* s_sx  = s_buf;
    double* s_sy  = s_buf + K;
    double* s_cnt = s_buf + 2*K;

    // Inisialisasi shared memory ke 0
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s_sx [k] = 0.0;
        s_sy [k] = 0.0;
        s_cnt[k] = 0.0;
    }
    __syncthreads();

    // Setiap thread akumulasi titik-titiknya ke shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int k = d_assign[i];
        // atomicAdd pada shared memory (tersedia sejak sm_60)
        atomicAdd(&s_sx [k], d_px[i]);
        atomicAdd(&s_sy [k], d_py[i]);
        atomicAdd(&s_cnt[k], 1.0);
    }
    __syncthreads();

    // Tulis partial sum block ini ke global memory
    // blockIdx.x = nomor block ini
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int out_idx = blockIdx.x * K + k;
        d_partial_sx [out_idx] = s_sx [k];
        d_partial_sy [out_idx] = s_sy [k];
        d_partial_cnt[out_idx] = (int)s_cnt[k];
    }
}

// ═══════════════════════════════════════════════════════════════
// KERNEL 3: Update Centroid — merge partial sums → centroid baru
// ═══════════════════════════════════════════════════════════════
__global__ void kernel_update_centroid(
    const double* d_partial_sx,
    const double* d_partial_sy,
    const int*    d_partial_cnt,
    double*       d_cx,
    double*       d_cy,
    int*          d_cnt,
    int K,
    int n_blocks  // = gridDim.x dari kernel sebelumnya
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    double sx = 0.0, sy = 0.0;
    int    sc = 0;

    // Merge semua partial sums untuk cluster k
    for (int b = 0; b < n_blocks; ++b) {
        sx += d_partial_sx [b * K + k];
        sy += d_partial_sy [b * K + k];
        sc += d_partial_cnt[b * K + k];
    }

    if (sc > 0) {
        d_cx [k] = sx / sc;
        d_cy [k] = sy / sc;
        d_cnt[k] = sc;
    }
}

// ─── Load CSV (host) ──────────────────────────────────────────
std::vector<Point> load_csv(const std::string& path, int cx=5,
                             int cy=6, long max_rows=-1)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Tidak bisa membuka: " + path);

    std::vector<Point> pts;
    std::string line;
    std::getline(file, line); // skip header

    long row = 0;
    while (std::getline(file, line)) {
        if (max_rows > 0 && row >= max_rows) break;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);

        if ((int)cols.size() <= std::max(cx,cy)) continue;
        try {
            double px = std::stod(cols[cx]), py = std::stod(cols[cy]);
            if (px < -75.0 || px > -73.0) continue;
            if (py <  40.0 || py >  41.5) continue;
            pts.push_back({px, py});
        } catch (...) { continue; }
        ++row;
    }
    return pts;
}

// ─── Init K-Means++ (host) ────────────────────────────────────
std::vector<Centroid> init_centroids_pp(const std::vector<Point>& pts, int K)
{
    std::mt19937_64 rng(42);
    std::vector<Centroid> cs;
    cs.reserve(K);
    std::uniform_int_distribution<size_t> idx(0, pts.size()-1);
    size_t first = idx(rng);
    cs.push_back({pts[first].x, pts[first].y, 0});

    for (int k = 1; k < K; ++k) {
        std::vector<double> d2(pts.size());
        double total = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            double md = 1e300;
            for (auto& c : cs) {
                double dx=pts[i].x-c.x, dy=pts[i].y-c.y;
                md = std::min(md, dx*dx+dy*dy);
            }
            d2[i] = md; total += md;
        }
        std::uniform_real_distribution<double> u(0, total);
        double t=u(rng), cu=0; size_t chosen=0;
        for (size_t i=0;i<pts.size();++i) { cu+=d2[i]; if(cu>=t){chosen=i;break;} }
        cs.push_back({pts[chosen].x, pts[chosen].y, 0});
    }
    return cs;
}

// ─── Cetak info GPU ───────────────────────────────────────────
void print_gpu_info() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "  GPU       : " << prop.name << "\n";
    std::cout << "  SM count  : " << prop.multiProcessorCount << "\n";
    std::cout << "  Shared/SM : " << prop.sharedMemPerBlock/1024 << " KB\n";
    std::cout << "  Warp size : " << prop.warpSize << "\n";
    std::cout << "  Max thread/block: " << prop.maxThreadsPerBlock << "\n";

    // Hitung occupancy teoritis kernel_assignment
    int block_size = BLOCK_SIZE;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size,
        kernel_assignment, 0, 0);
    std::cout << "  Suggested block size (occupancy): " << block_size << "\n";
    std::cout << "  Min grid size untuk full occupancy: " << min_grid_size << "\n";

    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, kernel_assignment, BLOCK_SIZE, 0);
    float occupancy = (float)(max_active_blocks * BLOCK_SIZE) /
                      prop.maxThreadsPerMultiProcessor;
    std::cout << "  Theoretical occupancy kernel_assignment: "
              << std::fixed << std::setprecision(1)
              << occupancy * 100.0f << "%\n";
}

// ─── Simpan metrik ────────────────────────────────────────────
void save_metrics(const std::string& path, int K, long N, int iters,
                  double h2d_ms, double km_ms, double d2h_ms,
                  double wcss, double throughput)
{
    std::ofstream f(path, std::ios::app);
    if (f.tellp() == 0)
        f << "impl,K,threads,n_points,iterations,"
          << "h2d_ms,kmeans_ms,d2h_ms,elapsed_ms,wcss,throughput_pts_sec\n";
    f << "cuda," << K << ",GPU," << N << "," << iters << ","
      << std::fixed << std::setprecision(2)
      << h2d_ms << "," << km_ms << "," << d2h_ms << ","
      << (h2d_ms+km_ms+d2h_ms) << ","
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

    std::cout << "========================================\n";
    std::cout << "  K-Means Clustering — CUDA GPU\n";
    std::cout << "========================================\n";
    print_gpu_info();
    std::cout << "K=" << K << "  max_iter=" << max_iter << "\n";
    std::cout << "----------------------------------------\n";

    if (K > MAX_K) {
        std::cerr << "[ERROR] K > MAX_K (" << MAX_K << "). Edit MAX_K di source.\n";
        return 1;
    }

    // ── Load data ──────────────────────────────────────────────
    std::cout << "[1/5] Memuat dataset...\n";
    auto pts = load_csv(path, 5, 6, max_rows);
    int  N   = (int)pts.size();
    std::cout << "  " << N << " titik dimuat\n";

    // Ubah AoS → SoA untuk coalesced GPU access
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; ++i) { h_px[i] = pts[i].x; h_py[i] = pts[i].y; }

    // ── Init centroid ──────────────────────────────────────────
    auto centroids = init_centroids_pp(pts, K);
    std::vector<double> h_cx(K), h_cy(K);
    std::vector<int>    h_cnt(K, 0);
    for (int k = 0; k < K; ++k) { h_cx[k] = centroids[k].x; h_cy[k] = centroids[k].y; }

    // ── Alokasi GPU memory ─────────────────────────────────────
    std::cout << "[2/5] Mengalokasi GPU memory...\n";
    double *d_px, *d_py, *d_cx, *d_cy;
    int    *d_assign, *d_cnt;

    CUDA_CHECK(cudaMalloc(&d_px,     N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py,     N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cx,     K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cy,     K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt,    K * sizeof(int)));

    // Partial sum buffers
    int n_blocks_ps = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *d_psx, *d_psy;
    int    *d_pcnt;
    CUDA_CHECK(cudaMalloc(&d_psx,  n_blocks_ps * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psy,  n_blocks_ps * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pcnt, n_blocks_ps * K * sizeof(int)));

    // ── Host → Device (H2D) ────────────────────────────────────
    std::cout << "[3/5] Transfer data Host → GPU...\n";
    cudaEvent_t ev_h2d_start, ev_h2d_end;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_end));

    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d_end));
    CUDA_CHECK(cudaEventSynchronize(ev_h2d_end));

    float h2d_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_end));
    std::cout << "  H2D transfer: " << h2d_ms << " ms ("
              << std::fixed << std::setprecision(1)
              << (2.0 * N * sizeof(double)) / (h2d_ms/1000.0) / 1e9
              << " GB/s)\n";

    // ── Iterasi K-Means di GPU ─────────────────────────────────
    std::cout << "[4/5] Iterasi K-Means di GPU...\n";
    cudaEvent_t ev_km_start, ev_km_end;
    CUDA_CHECK(cudaEventCreate(&ev_km_start));
    CUDA_CHECK(cudaEventCreate(&ev_km_end));

    int grid_assign = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_update = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shm_ps   = 3 * K * sizeof(double); // shared mem untuk kernel_partial_sum

    int iter = 0;
    CUDA_CHECK(cudaEventRecord(ev_km_start));

    // Buffer host sementara untuk cek konvergensi (polling setiap 5 iterasi)
    std::vector<int> h_assign_prev(N, -1);
    std::vector<int> h_assign_curr(N, 0);

    while (iter < max_iter) {
        // Upload centroid terbaru ke GPU
        CUDA_CHECK(cudaMemcpy(d_cx, h_cx.data(), K*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cy, h_cy.data(), K*sizeof(double), cudaMemcpyHostToDevice));

        // Kernel 1: Assignment (shared memory centroid)
        kernel_assignment<<<grid_assign, BLOCK_SIZE>>>(
            d_px, d_py, d_assign, d_cx, d_cy, N, K);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2: Partial sum (shared memory akumulasi per block)
        CUDA_CHECK(cudaMemset(d_psx, 0, n_blocks_ps*K*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_psy, 0, n_blocks_ps*K*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_pcnt, 0, n_blocks_ps*K*sizeof(int)));
        kernel_partial_sum<<<n_blocks_ps, BLOCK_SIZE, shm_ps>>>(
            d_px, d_py, d_assign, d_psx, d_psy, d_pcnt, N, K);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 3: Update centroid
        CUDA_CHECK(cudaMemset(d_cnt, 0, K*sizeof(int)));
        kernel_update_centroid<<<grid_update, BLOCK_SIZE>>>(
            d_psx, d_psy, d_pcnt, d_cx, d_cy, d_cnt, K, n_blocks_ps);
        CUDA_CHECK(cudaGetLastError());

        // Download centroid baru ke host untuk iterasi berikutnya
        CUDA_CHECK(cudaMemcpy(h_cx.data(), d_cx, K*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cy.data(), d_cy, K*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cnt.data(), d_cnt, K*sizeof(int), cudaMemcpyDeviceToHost));

        // Cek konvergensi setiap 5 iterasi (download assignment mahal)
        if (iter % 5 == 0) {
            CUDA_CHECK(cudaMemcpy(h_assign_curr.data(), d_assign, N*sizeof(int),
                                  cudaMemcpyDeviceToHost));
            int moved = 0;
            for (int i = 0; i < N; ++i) moved += (h_assign_curr[i] != h_assign_prev[i]);
            std::cout << "  Iter " << std::setw(3) << iter
                      << " | moved=" << std::setw(8) << moved << "\n";
            if (moved == 0) break;
            h_assign_prev = h_assign_curr;
        }
        ++iter;
    }

    CUDA_CHECK(cudaEventRecord(ev_km_end));
    CUDA_CHECK(cudaEventSynchronize(ev_km_end));
    float km_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&km_ms, ev_km_start, ev_km_end));

    // ── Device → Host (D2H) ────────────────────────────────────
    std::cout << "[5/5] Transfer hasil GPU → Host...\n";
    cudaEvent_t ev_d2h_start, ev_d2h_end;
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_end));

    std::vector<int> h_assign_final(N);
    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(h_assign_final.data(), d_assign, N*sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_end));
    float d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_end));

    // ── Hitung WCSS (host) ─────────────────────────────────────
    double wcss = 0;
    for (int i = 0; i < N; ++i) {
        int k = h_assign_final[i];
        double dx = pts[i].x - h_cx[k], dy = pts[i].y - h_cy[k];
        wcss += dx*dx + dy*dy;
    }
    double tp = (double)N * iter / (km_ms / 1000.0);

    std::cout << "----------------------------------------\n";
    std::cout << "  Iterasi     : " << iter << "\n";
    std::cout << "  H2D         : " << std::fixed << std::setprecision(2) << h2d_ms << " ms\n";
    std::cout << "  K-Means GPU : " << std::fixed << std::setprecision(2) << km_ms  << " ms\n";
    std::cout << "  D2H         : " << std::fixed << std::setprecision(2) << d2h_ms << " ms\n";
    std::cout << "  Total       : " << std::fixed << std::setprecision(2)
              << (h2d_ms+km_ms+d2h_ms) << " ms\n";
    std::cout << "  Throughput  : " << std::fixed << std::setprecision(0)
              << tp << " pts/sec\n";
    std::cout << "  WCSS final  : " << std::scientific << std::setprecision(4)
              << wcss << "\n";
    std::cout << "\n  Ukuran cluster:\n";
    for (int k = 0; k < K; ++k)
        std::cout << "    Cluster " << k << ": " << h_cnt[k] << " titik\n";

    // Bersihkan GPU memory
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cx); cudaFree(d_cy);
    cudaFree(d_assign); cudaFree(d_cnt);
    cudaFree(d_psx); cudaFree(d_psy); cudaFree(d_pcnt);

    save_metrics("metrics.csv", K, N, iter, h2d_ms, km_ms, d2h_ms, wcss, tp);
    return 0;
}
