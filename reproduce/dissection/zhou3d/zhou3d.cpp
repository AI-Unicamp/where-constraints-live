/*
 * Lattice dissection search of Zhou & Wang (2012), in 2D and (below) 3D.
 *
 * Our C++ implementation of "A Computational Algorithm for Creating Geometric
 * Dissection Puzzles" (UM-CS-2011-028 / SIGGRAPH 2012) by Yahan Zhou and
 * Rui Wang, ported from our earlier Python version.
 *
 * Elements taken from the paper:
 * - Discrete lattice (square), binary figures with equal area.
 * - Distance D: 8 isometries (4 rotations x flip), center alignment + 4-snap; counts bidirectional mismatches.
 * - Matching: min-cost bipartite (Bellman-Ford style) on per-piece mismatch costs.
 * - Objective: E = EM + λ(Eα(A)+Eα(B)) + split-penalty; λ = (λ0/2) * decay^(level-1).
 * - Level 1: many random trials (N), each with two-seed grow + R back-and-forth refinement; keep Nb best.
 * - Level >=2: random exploration (N trials), each trial either:
 * (H#1) split the worst-matched parent pair using Level-1 on the parent masks (paper’s “optimal split”),
 * (H#2) create a new cluster from mismatched pixels (seed + flood-fill), then run refinement.
 * - Refinement (both directions): copy-paste, random flood-fill of holes (as in the paper), then random label switching
 * over pixels that would be OVERWRITTEN (overlap conflicts), not XOR mismatches.
 *
 * Determinism:
 * - Random choices use a std::mt19937 generator seeded per trial/level.
 * - No global RNG state is shared across threads.
 *
 * Dependencies (header-only, public domain, included in this folder):
 *   stb_image.h and stb_image_write.h by Sean Barrett (image I/O).
 *
 * Build (MSVC): build_zhou3d.bat
 * Build (g++):  g++ -std=c++17 -O3 -pthread zhou3d.cpp -o zhou3d
 */

// --- C++ Standard Library Includes ---
#include <iostream>     // For std::cout, std::cerr
#include <vector>       // For std::vector
#include <string>       // For std::string
#include <map>          // For std::map
#include <set>          // For std::set
#include <deque>        // For std::deque
#include <cmath>        // For std::cmath, M_PI
#include <limits>       // For std::numeric_limits
#include <random>       // For std::mt19937, std::uniform_real_distribution
#include <numeric>      // For std::iota
#include <algorithm>    // For std::sort, std::shuffle, std::max_element
#include <thread>       // For std::thread
#include <mutex>        // For std::mutex
#include <future>       // For std::future, std::promise
#include <condition_variable> // For std::condition_variable
#include <filesystem>   // For std::filesystem::path
#include <chrono>       // For timing (optional)
#include <iomanip>      // For std::setw, std::setprecision
#include <tuple>        // For std::tuple, std::pair
#include <stdexcept>    // For std::runtime_error
#include <functional>   // For std::function (used by ThreadPool)
#include <climits>      // For INT_MAX, INT_MIN
#include <cstdint>      // For uint8_t, uint32_t, uint64_t
#include <array>        // For std::array (used in four_snaps)
#include <fstream>      // For checkpoint metadata

// --- External Image Libraries (STB) ---
// Define these *before* including, only in this one .cpp file
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- Dense 2D matrix ---
template <typename T>
class Matrix {
public:
    int m_rows, m_cols;
    std::vector<T> m_data;

    Matrix() : m_rows(0), m_cols(0) {}

    Matrix(int rows, int cols, T val = T())
        : m_rows(rows), m_cols(cols), m_data(rows * cols, val) {}

    T& at(int r, int c) {
        return m_data[static_cast<size_t>(r) * m_cols + c];
    }

    const T& at(int r, int c) const {
        return m_data[static_cast<size_t>(r) * m_cols + c];
    }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    size_t size() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }

    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }
};

// --- Type aliases ---
using Mask = Matrix<uint8_t>;
using Labels = Matrix<int>;
using Point = std::pair<int, int>;
using Coords = std::vector<Point>; // (y, x) coordinates
using BBox = std::tuple<int, int, int, int>; // y_min, x_min, y_max, x_max

// Struct for pair_mismatch return value
struct MismatchParams {
    int rot = 0;
    bool flip = false;
    Point shift = {0, 0};
};
struct MismatchResult {
    double mismatch = std::numeric_limits<double>::infinity();
    MismatchParams params;
};

// Struct for match_labels return value
using MatchMap = std::map<int, int>;
using TransformMap = std::map<std::pair<int, int>, MismatchResult>;
struct MatchResult {
    MatchMap M;
    double totalMismatch = 0.0;
    TransformMap T;
};

// Struct for refinement results
struct RefineResult {
    Labels LA;
    Labels LB;
    double em = 0.0;
};

// Struct for Objective function results
struct ObjectiveResult {
    double total, mismatch, area, split;
};

// Struct for MCMF Edge
struct Edge {
    int to;
    size_t rev; // Index of the reverse edge in g[to]
    int cap;
    double cost;
};

// --- Simple Thread Pool ---
// Fixed-size worker pool for parallel trials
class ThreadPool {
public:
    ThreadPool(size_t numThreads) : m_stop(false) {
        for(size_t i = 0; i < numThreads; ++i) {
            m_workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->m_queueMutex);
                        this->m_condition.wait(lock, [this]{ return this->m_stop || !this->m_tasks.empty(); });
                        if(this->m_stop && this->m_tasks.empty()) return;
                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop_front();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_stop = true;
        }
        m_condition.notify_all();
        for(std::thread &worker : m_workers) {
            worker.join();
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            if(m_stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            m_tasks.emplace_back([task](){ (*task)(); });
        }
        m_condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> m_workers;
    std::deque<std::function<void()>> m_tasks;
    std::mutex m_queueMutex;
    std::condition_variable m_condition;
    bool m_stop;
};

// --- Global Defaults ---
const int R_DEFAULT = 100;
const int N_DEFAULT = 400;
const int NB_DEFAULT = 30;
const int MAX_LEVEL_DEFAULT = 15;
const double LAMBDA0_DEFAULT = 1.0;
const double DECAY_DEFAULT = 0.8;
const double SPLIT_PENALTY_DEFAULT = 1e6;
const uint64_t SEED_DEFAULT = 2012;
const double CONVERGENCE_EPS = 1e-9;

// --- Struct to hold all runtime arguments ---
struct Args {
    std::string shapeA_path;
    std::string shapeB_path;
    std::string outdir;
    int R = R_DEFAULT;
    int N = N_DEFAULT;
    int Nb = NB_DEFAULT;
    int max_level = MAX_LEVEL_DEFAULT;
    double lambda0 = LAMBDA0_DEFAULT;
    double decay = DECAY_DEFAULT;
    double split_penalty = SPLIT_PENALTY_DEFAULT;
    uint64_t seed = SEED_DEFAULT;
    int jobs = -1;
};


// ----------------- I/O -----------------
Mask load_binary(const std::string& path, int thr = 128) {
    int w = 0, h = 0, comp = 0;
    // Force 1-channel grayscale load
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &comp, 1);
    if (!img) {
        throw std::runtime_error("stbi_load failed for: " + path);
    }
    Mask out(h, w, uint8_t(0));
    // As in the paper's data: foreground is "dark" (value < thr) -> 1, else 0
    for (int y = 0; y < h; ++y) {
        const unsigned char* row = img + y * w;
        for (int x = 0; x < w; ++x) {
            out.at(y, x) = (row[x] < thr) ? uint8_t(1) : uint8_t(0);
        }
    }
    stbi_image_free(img);
    return out;
}

// Simple deterministic hash -> RGB for label visualization
static inline std::tuple<uint8_t,uint8_t,uint8_t> _label_color(int id) {
    if (id <= 0) return {0,0,0};
    // xorshift-ish hashing of the label id
    uint32_t v = static_cast<uint32_t>(id) * 0x9E3779B1u; // Knuth
    v ^= v >> 16; v *= 0x85EBCA6Bu; v ^= v >> 13; v *= 0xC2B2AE35u; v ^= v >> 16;
    uint8_t r = static_cast<uint8_t>( (v      ) & 0xFF );
    uint8_t g = static_cast<uint8_t>( (v >>  8) & 0xFF );
    uint8_t b = static_cast<uint8_t>( (v >> 16) & 0xFF );
    // Avoid very dark colors for readability
    if (r < 24 && g < 24 && b < 24) { r ^= 0x55; g ^= 0xAA; b ^= 0xCC; }
    return {r,g,b};
}

void save_labels_png(const std::string& path, const Labels& labels) {
    const int H = labels.rows();
    const int W = labels.cols();
    if (H <= 0 || W <= 0) {
        throw std::runtime_error("save_labels_png: empty labels.");
    }

    // Scan once to detect max label and decide encoding
    int maxlab = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (labels.at(y,x) > maxlab) maxlab = labels.at(y,x);

    // Always save as RGB for easy visual debugging
    std::vector<uint8_t> rgb(static_cast<size_t>(H) * W * 3, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int id = labels.at(y,x);
            auto [r,g,b] = _label_color(id);
            size_t idx = static_cast<size_t>(y) * W * 3 + static_cast<size_t>(x) * 3;
            rgb[idx+0] = r; rgb[idx+1] = g; rgb[idx+2] = b;
        }
    }
    if (!stbi_write_png(path.c_str(), W, H, 3, rgb.data(), W * 3)) {
        throw std::runtime_error("stbi_write_png failed for: " + path);
    }
}
// ----------------- Label utils -----------------
std::vector<int> cluster_ids(const Labels& L) {
    std::set<int> s;
    const int H = L.rows(), W = L.cols();
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int v = L.at(y,x);
            if (v > 0) s.insert(v);
        }
    return std::vector<int>(s.begin(), s.end());
}

Mask mask_of(const Labels& L, int cid) {
    const int H = L.rows(), W = L.cols();
    Mask m(H, W, uint8_t(0));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            m.at(y,x) = (L.at(y,x) == cid) ? uint8_t(1) : uint8_t(0);
    return m;
}

std::map<int, int> areas(const Labels& L) {
    std::map<int,int> A;
    const int H = L.rows(), W = L.cols();
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int v = L.at(y,x);
            if (v > 0) ++A[v];
        }
    return A;
}

BBox bbox(const Mask& mask) {
    const int H = mask.rows(), W = mask.cols();
    int ymin = H, xmin = W, ymax = -1, xmax = -1;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (mask.at(y,x)) {
                if (y < ymin) ymin = y;
                if (x < xmin) xmin = x;
                if (y > ymax) ymax = y;
                if (x > xmax) xmax = x;
            }
        }
    }
    if (ymax < 0) { // empty
        return {0, 0, -1, -1};
    }
    return {ymin, xmin, ymax, xmax};
}

int components4(const Mask& mask) {
    const int H = mask.rows(), W = mask.cols();
    if (H == 0 || W == 0) return 0;

    // Quick empty check
    bool any = false;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask.m_data[i]) { any = true; break; }
    }
    if (!any) return 0;

    std::vector<uint8_t> seen(static_cast<size_t>(H) * W, uint8_t(0));
    auto seen_at = [&](int y, int x) -> uint8_t& { return seen[static_cast<size_t>(y)*W + x]; };

    int comps = 0;
    std::deque<Point> q;

    for (int y0 = 0; y0 < H; ++y0) {
        for (int x0 = 0; x0 < W; ++x0) {
            if (!mask.at(y0,x0) || seen_at(y0,x0)) continue;
            ++comps;
            seen_at(y0,x0) = 1;
            q.clear();
            q.emplace_back(y0, x0);
            while (!q.empty()) {
                auto [y, x] = q.front(); q.pop_front();
                const int dy4[4] = {1,-1,0,0};
                const int dx4[4] = {0,0,1,-1};
                for (int k = 0; k < 4; ++k) {
                    int ny = y + dy4[k], nx = x + dx4[k];
                    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    if (!mask.at(ny,nx)) continue;
                    if (seen_at(ny,nx)) continue;
                    seen_at(ny,nx) = 1;
                    q.emplace_back(ny, nx);
                }
            }
        }
    }
    return comps;
}

// ----------------- Geometry -----------------
// Struct for rotflip return value
struct RotflipResult {
    int h = 0, w = 0;
    Coords coords;  // std::vector<std::pair<int,int>>
};

static inline std::pair<double,double> _centroid_f(const Coords& coords) {
    if (coords.empty()) return {-1.0, -1.0};
    long long sy = 0, sx = 0;
    for (auto &p : coords) { sy += p.first; sx += p.second; }
    double n = static_cast<double>(coords.size());
    return { sy / n, sx / n };
}

Point centroid(const Coords& coords) {
    auto [cy, cx] = _centroid_f(coords);
    if (cy < 0) return {-1, -1};
    // integer centroid (rounded)
    return { static_cast<int>(std::lround(cy)), static_cast<int>(std::lround(cx)) };
}

RotflipResult rotflip(const Coords& coords, int h, int w, int k, bool flip) {
    // k in {0,1,2,3} corresponding to 0/90/180/270 degrees
    RotflipResult R;
    int hh = h, ww = w;
    if (k == 0)       { hh = h; ww = w; }
    else if (k == 1)  { hh = w; ww = h; }
    else if (k == 2)  { hh = h; ww = w; }
    else              { hh = w; ww = h; }

    R.h = hh; R.w = ww;
    R.coords.reserve(coords.size());

    for (auto &p : coords) {
        int y = p.first, x = p.second;
        int y2, x2;
        switch (k & 3) {
            case 0: y2 = y;          x2 = x;          break;             // 0°
            case 1: y2 = x;          x2 = (h - 1) - y;break;             // 90°
            case 2: y2 = (h - 1)- y; x2 = (w - 1) - x;break;             // 180°
            default:y2 = (w - 1)- x; x2 = y;          break;             // 270°
        }
        // horizontal flip (over x-axis direction) in the rotated frame
        if (flip) x2 = (ww - 1) - x2;
        R.coords.emplace_back(y2, x2);
    }
    return R;
}

std::array<Point, 4> four_snaps(double dy, double dx) {
    int fy = static_cast<int>(std::floor(dy));
    int cy = static_cast<int>(std::ceil(dy));
    int fx = static_cast<int>(std::floor(dx));
    int cx = static_cast<int>(std::ceil(dx));
    return { Point{fy, fx}, Point{fy, cx}, Point{cy, fx}, Point{cy, cx} };
}

// ----------------- Distance metric (Eq. 1) -----------------
class TargetCache {
public:
    explicit TargetCache(const Labels& LB) : m_LB(&LB) {}

    const Mask& get(int cid) {
        auto it = m_masks.find(cid);
        if (it != m_masks.end()) return it->second;
        const int H = m_LB->rows(), W = m_LB->cols();
        Mask m(H, W, uint8_t(0));
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (m_LB->at(y,x) == cid) m.at(y,x) = 1;
        auto [ins, _] = m_masks.emplace(cid, std::move(m));
        return ins->second;
    }

private:
    const Labels* m_LB;
    std::map<int, Mask> m_masks;
};

Coords cluster_coords_flat(const Labels& L, int cid) {
    Coords out;
    const int H = L.rows(), W = L.cols();
    out.reserve(static_cast<size_t>(H) * 4); // rough guess
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (L.at(y,x) == cid) out.emplace_back(y, x);
    return out;
}

MismatchResult pair_mismatch(const Coords& Ayx, const Coords& Byx, const Mask& Bmask) {
    MismatchResult res;
    const int Sa = static_cast<int>(Ayx.size());
    const int Sb = static_cast<int>(Byx.size());
    const int H = Bmask.rows(), W = Bmask.cols();

    if (Sa == 0 && Sb == 0) { res.mismatch = 0.0; res.params = {0,false,{0,0}}; return res; }
    if (Sa == 0 || Sb == 0) { res.mismatch = static_cast<double>(std::max(Sa, Sb)); res.params = {0,false,{0,0}}; return res; }

    // bbox for A
    int y0 = INT_MAX, x0 = INT_MAX, y1 = -1, x1 = -1;
    for (auto &p : Ayx) {
        int y = p.first, x = p.second;
        if (y < y0) y0 = y; if (x < x0) x0 = x;
        if (y > y1) y1 = y; if (x > x1) x1 = x;
    }
    const int h = y1 - y0 + 1, w = x1 - x0 + 1;

    // local coords of A inside its bbox
    Coords A_local; A_local.reserve(Ayx.size());
    for (auto &p : Ayx) A_local.emplace_back(p.first - y0, p.second - x0);

    // centroid of B (global, double)
    auto [cyB, cxB] = _centroid_f(Byx);
    double lb = std::abs(Sa - Sb); // lower bound

    double best_mis = std::numeric_limits<double>::infinity();
    MismatchParams best_par{0,false,{0,0}};

    for (int flip = 0; flip < 2; ++flip) {
        for (int rotk = 0; rotk < 4; ++rotk) {
            auto R = rotflip(A_local, h, w, rotk, flip != 0);
            if (R.coords.empty()) continue;

            // centroid of transformed A (local)
            auto [cyA_l, cxA_l] = _centroid_f(R.coords);
            if (cyA_l < 0) continue;
            double cyA = cyA_l + y0;
            double cxA = cxA_l + x0;

            auto snaps = four_snaps(cyB - cyA, cxB - cxA);
            for (auto s : snaps) {
                int dy = s.first, dx = s.second;
                // count overlap (transformed A into Bmask)
                int overlap = 0;
                for (auto &pt : R.coords) {
                    int gy = pt.first  + y0 + dy;
                    int gx = pt.second + x0 + dx;
                    if (0 <= gy && gy < H && 0 <= gx && gx < W) {
                        if (Bmask.at(gy, gx)) ++overlap;
                    }
                }
                double mis = static_cast<double>(Sa + Sb - 2 * overlap);
                if (mis < best_mis) {
                    best_mis = mis;
                    best_par.rot = rotk * 90;
                    best_par.flip = (flip != 0);
                    best_par.shift = {dy, dx};

                    if (best_mis <= lb + 1e-12) {
                        res.mismatch = best_mis; res.params = best_par; return res;
                    }
                    if (best_mis < CONVERGENCE_EPS) {
                        res.mismatch = 0.0; res.params = best_par; return res;
                    }
                }
            }
        }
    }

    res.mismatch = best_mis;
    res.params   = best_par;
    return res;
}

// ----------------- Assignment via optimized Bellman-Ford (Eq. 2) -----------------
void _add(std::vector<std::vector<Edge>>& g, int u, int v, int cap, double cost) {
    Edge f{v, g[v].size(), cap, cost};
    Edge r{u, g[u].size(), 0,   -cost};
    g[u].push_back(f);
    g[v].push_back(r);
}

std::pair<double, std::vector<Point>> _mcmf_bf(const Matrix<double>& cost_square, int maxflow) {
    const int n = cost_square.rows();
    const int S = 2*n, T = 2*n + 1;
    const double INF = 1e30;

    std::vector<std::vector<Edge>> g(2*n + 2);
    for (int i = 0; i < n; ++i) {
        _add(g, S, i,    1, 0.0);
        _add(g, n+i, T,  1, 0.0);
        for (int j = 0; j < n; ++j) {
            _add(g, i, n+j, 1, cost_square.at(i, j));
        }
    }

    auto spfa = [&](std::vector<double>& dist, std::vector<std::pair<int,size_t>>& parent)->bool {
        const int N = static_cast<int>(g.size());
        dist.assign(N, INF);
        parent.assign(N, {-1, 0});
        std::vector<uint8_t> inq(N, 0);
        std::deque<int> q;
        dist[S] = 0.0; q.push_back(S); inq[S] = 1;

        while (!q.empty()) {
            int u = q.front(); q.pop_front(); inq[u] = 0;
            for (size_t ei = 0; ei < g[u].size(); ++ei) {
                const Edge& e = g[u][ei];
                if (e.cap <= 0) continue;
                int v = e.to;
                double nd = dist[u] + e.cost;
                if (nd + 1e-18 < dist[v]) {
                    dist[v] = nd; parent[v] = {u, ei};
                    if (!inq[v]) { inq[v] = 1; q.push_back(v); }
                }
            }
        }
        return dist[T] < INF/2;
    };

    int flow = 0;
    double tot = 0.0;
    std::vector<double> dist;
    std::vector<std::pair<int,size_t>> parent;

    while (flow < maxflow && spfa(dist, parent)) {
        // augment 1 unit
        int v = T;
        while (v != S) {
            auto [u, ei] = parent[v];
            Edge &e = g[u][ei];
            Edge &rev = g[v][e.rev];
            e.cap -= 1; rev.cap += 1;
            v = u;
        }
        ++flow; tot += dist[T];
    }

    std::vector<Point> pairs;
    for (int i = 0; i < n; ++i) {
        for (auto &e : g[i]) {
            if (e.to >= n && e.to < 2*n) {
                int j = e.to - n;
                const Edge& rev = g[e.to][e.rev];
                // matched edge: forward cap consumed (0), reverse cap = 1
                if (e.cap == 0 && rev.cap == 1) {
                    pairs.emplace_back(i, j);
                }
            }
        }
    }
    return {tot, pairs};
}

std::vector<int> bellman_ford_assignment(const Matrix<double>& cost) {
    const int nr = cost.rows();
    const int nc = cost.cols();
    const int n  = std::max(nr, nc);
    const double INF = 1e20;

    Matrix<double> C(n, n, INF);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
            C.at(i, j) = std::isfinite(cost.at(i,j)) ? cost.at(i,j) : INF;

    auto mres = _mcmf_bf(C, nr);
    const auto& pairs = mres.second;
    std::vector<int> assign(nr, -1);
    for (auto &pr : pairs) {
        int i = pr.first, j = pr.second;
        if (i < nr && j < nc && C.at(i,j) < INF/2) assign[i] = j;
    }
    return assign;
}

// ----------------- Matching -----------------
MatchResult match_labels(const Labels& LA, const Labels& LB, const Mask& target_fg) {
    MatchResult out;
    const auto idsA = cluster_ids(LA);
    const auto idsB = cluster_ids(LB);
    if (idsA.empty() || idsB.empty()) return out;

    // Precompute coords for clusters
    std::map<int, Coords> coordsA, coordsB;
    for (int a : idsA) coordsA[a] = cluster_coords_flat(LA, a);
    for (int b : idsB) coordsB[b] = cluster_coords_flat(LB, b);

    TargetCache cache(LB);

    Matrix<double> cost(static_cast<int>(idsA.size()), static_cast<int>(idsB.size()),
                        std::numeric_limits<double>::infinity());

    for (int i = 0; i < static_cast<int>(idsA.size()); ++i) {
        int a = idsA[i];
        for (int j = 0; j < static_cast<int>(idsB.size()); ++j) {
            int b = idsB[j];
            const Mask& Bm = cache.get(b);
            MismatchResult mr = pair_mismatch(coordsA[a], coordsB[b], Bm);
            cost.at(i, j) = mr.mismatch;
            out.T[{a, b}] = mr;
        }
    }

    auto assign = bellman_ford_assignment(cost);
    double total = 0.0;
    for (int i = 0; i < static_cast<int>(idsA.size()); ++i) {
        int j = (i < static_cast<int>(assign.size())) ? assign[i] : -1;
        if (j >= 0 && std::isfinite(cost.at(i, j))) {
            int a = idsA[i], b = idsB[j];
            out.M[a] = b;
            total += cost.at(i, j);
        }
    }
    out.totalMismatch = total;
    return out;
}

// ----------------- Paste + masks -----------------
Mask apply_isometry(const Mask& mask, const MismatchParams& params, const Mask& target_fg) {
    const int H = target_fg.rows(), W = target_fg.cols();
    Mask out(H, W, uint8_t(0));
    // Quick empty
    bool any = false;
    for (auto v : mask.m_data) { if (v) { any = true; break; } }
    if (!any) return out;

    // Bounding box of source mask
    auto [y0, x0, y1, x1] = bbox(mask);
    if (y1 < y0 || x1 < x0) return out; // empty

    // Gather local coords (relative to bbox)
    Coords local;
    local.reserve(static_cast<size_t>((y1 - y0 + 1) * (x1 - x0 + 1)));
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (mask.at(y, x)) local.emplace_back(y - y0, x - x0);

    // Apply rotation+flip in local frame, then global shift
    int k = ((params.rot / 90) % 4 + 4) % 4;
    auto R = rotflip(local, y1 - y0 + 1, x1 - x0 + 1, k, params.flip);
    const int dy = params.shift.first;
    const int dx = params.shift.second;

    for (auto &pt : R.coords) {
        int gy = pt.first  + y0 + dy;
        int gx = pt.second + x0 + dx;
        if (0 <= gy && gy < H && 0 <= gx && gx < W && target_fg.at(gy, gx)) {
            out.at(gy, gx) = 1;
        }
    }
    return out;
}

Labels forward_copy(const Labels& LA, const Labels& LB, const MatchResult& match,
                    const Mask& shapeB, std::mt19937& rng) {
    const int H = LB.rows(), W = LB.cols();
    Labels out(H, W, 0);

    // Shuffle paste order for determinism across trials
    std::vector<std::pair<int,int>> items;
    items.reserve(match.M.size());
    for (auto &kv : match.M) items.emplace_back(kv.first, kv.second);
    std::shuffle(items.begin(), items.end(), rng);

    for (auto &ab : items) {
        int a_id = ab.first, b_id = ab.second;
        auto it = match.T.find({a_id, b_id});
        if (it == match.T.end()) continue;

        Mask a_mask = mask_of(LA, a_id);
        const MismatchParams& P = it->second.params;
        Mask placed = apply_isometry(a_mask, P, shapeB);

        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (placed.at(y, x)) out.at(y, x) = b_id;
    }

    // Ensure background stays zero
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (!shapeB.at(y, x)) out.at(y, x) = 0;

    return out;
}

// ----------------- Random flood fill for holes (as in the paper) -----------------
Labels fill_holes_random_flood(const Labels& L, const Mask& fg, std::mt19937& rng) {
    // Grow labels outward from the pasted frontier to fill any unlabeled holes (inside fg).
    const int H = L.rows(), W = L.cols();
    Labels out = L;

    // holes: inside fg but unlabeled
    auto hole_at = [&](int y, int x){ return fg.at(y,x) && (out.at(y,x) == 0); };

    bool any_hole = false;
    for (int y = 0; y < H && !any_hole; ++y)
        for (int x = 0; x < W && !any_hole; ++x)
            if (hole_at(y,x)) any_hole = true;
    if (!any_hole) return out;

    // Build initial frontier: labeled pixels adjacent to any hole
    std::vector<Point> frontier;
    frontier.reserve(static_cast<size_t>(H*W/8));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (!fg.at(y,x) || out.at(y,x) == 0) continue;
            bool is_front = false;
            const int dy4[4] = {1,-1,0,0};
            const int dx4[4] = {0,0,1,-1};
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy4[k], nx = x + dx4[k];
                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                if (hole_at(ny, nx)) { is_front = true; break; }
            }
            if (is_front) frontier.emplace_back(y, x);
        }
    }

    if (frontier.empty()) {
        // Fallback: assign each hole to any valid 4-neighbor label if possible
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!hole_at(y,x)) continue;
                const int dy4[4] = {1,-1,0,0};
                const int dx4[4] = {0,0,1,-1};
                for (int k = 0; k < 4; ++k) {
                    int ny = y + dy4[k], nx = x + dx4[k];
                    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    if (fg.at(ny,nx) && out.at(ny,nx) != 0) { out.at(y,x) = out.at(ny,nx); break; }
                }
            }
        }
        // Any stragglers -> label 1
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (fg.at(y,x) && out.at(y,x) == 0) out.at(y,x) = 1;
        return out;
    }

    // Shuffle frontier, then BFS growth
    std::shuffle(frontier.begin(), frontier.end(), rng);
    std::deque<Point> q(frontier.begin(), frontier.end());

    while (!q.empty()) {
        auto [y, x] = q.front(); q.pop_front();
        int lab = out.at(y, x);
        const int dy4[4] = {1,-1,0,0};
        const int dx4[4] = {0,0,1,-1};
        for (int k = 0; k < 4; ++k) {
            int ny = y + dy4[k], nx = x + dx4[k];
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            if (fg.at(ny,nx) && out.at(ny,nx) == 0) {
                out.at(ny,nx) = lab;
                q.emplace_back(ny, nx);
            }
        }
    }

    // Final safety: any leftover holes -> label 1
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (fg.at(y,x) && out.at(y,x) == 0) out.at(y,x) = 1;

    return out;
}

// ----------------- Conflict/overlap and mismatches -----------------
Mask overlap_mask_src_to_dst(const Labels& Lsrc, const Labels& Ldst, const Mask& fgdst) {
    const int H = fgdst.rows(), W = fgdst.cols();
    Mask ov(H, W, uint8_t(0));

    auto mr = match_labels(Lsrc, Ldst, fgdst);
    for (auto &kv : mr.M) {
        int s_id = kv.first, d_id = kv.second;
        const auto it = mr.T.find({s_id, d_id});
        if (it == mr.T.end()) continue;

        Mask s = mask_of(Lsrc, s_id);
        Mask placed = apply_isometry(s, it->second.params, fgdst);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!fgdst.at(y,x) || !placed.at(y,x)) continue;
                int lab = Ldst.at(y,x);
                if (lab != 0 && lab != d_id) ov.at(y,x) = 1;
            }
        }
    }
    return ov;
}

Mask mismatches_src_outside_dst(const Labels& Lsrc, const Labels& Ldst, const Mask& fgdst) {
    const int H = fgdst.rows(), W = fgdst.cols();
    Mask mis(H, W, uint8_t(0));

    auto mr = match_labels(Lsrc, Ldst, fgdst);
    for (auto &kv : mr.M) {
        int s_id = kv.first, d_id = kv.second;
        const auto it = mr.T.find({s_id, d_id});
        if (it == mr.T.end()) continue;

        Mask s = mask_of(Lsrc, s_id);
        Mask d(H, W, uint8_t(0));
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (Ldst.at(y,x) == d_id) d.at(y,x) = 1;

        Mask placed = apply_isometry(s, it->second.params, fgdst);

        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (fgdst.at(y,x) && placed.at(y,x) && !d.at(y,x))
                    mis.at(y,x) = 1;
    }
    return mis;
}

Labels random_switch(const Labels& L, const Mask& fg, const Mask& conflict, std::mt19937& rng) {
    const int H = L.rows(), W = L.cols();
    Labels out = L;

    // Collect conflict pixels
    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(H*W/16));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (conflict.at(y,x)) pts.emplace_back(y, x);

    // Shuffle visit order
    std::shuffle(pts.begin(), pts.end(), rng);

    for (auto &p : pts) {
        int y = p.first, x = p.second;
        std::vector<int> neigh;
        neigh.reserve(4);
        const int dy4[4] = {1,-1,0,0};
        const int dx4[4] = {0,0,1,-1};
        for (int k = 0; k < 4; ++k) {
            int ny = y + dy4[k], nx = x + dx4[k];
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            if (fg.at(ny,nx) && out.at(ny,nx) != 0) neigh.push_back(out.at(ny,nx));
        }
        if (!neigh.empty()) {
            std::uniform_int_distribution<int> d(0, static_cast<int>(neigh.size()) - 1);
            out.at(y,x) = neigh[d(rng)];
        }
    }
    return out;
}

// ----------------- Objective (Eq. 3/4) -----------------
double level_lambda(int level, double lambda0, double decay) {
    // Paper: λ = (λ0 / 2) * decay^(level-1), with floor at level=1
    int k = std::max(0, level - 1);
    return (lambda0 / 2.0) * std::pow(decay, static_cast<double>(k));
}

double area_penalty(const Labels& L, const Mask& fg, double lam) {
    if (lam <= 0.0) return 0.0;
    auto A = areas(L);
    if (A.empty()) return 0.0;

    // Total foreground area
    double total = 0.0;
    for (auto v : fg.m_data) total += (v ? 1.0 : 0.0);
    if (total <= 0.0) return 0.0;

    const int n = static_cast<int>(A.size());
    const double avg = total / static_cast<double>(n);
    if (avg <= 0.0) return 0.0;

    double s = 0.0;
    for (const auto& kv : A) {
        const double ai = static_cast<double>(kv.second);
        if (ai <= 0.0) continue;
        if (ai > 2.0 * avg) {
            s += (ai / avg) - 1.0;
        } else if (ai < 0.5 * avg) {
            s += (avg / ai) - 1.0;
        }
    }
    return lam * s;
}

double split_penalty(const Labels& L, double penalty) {
    if (penalty <= 0.0) return 0.0;
    double s = 0.0;
    for (int cid : cluster_ids(L)) {
        Mask m = mask_of(L, cid);
        int c = components4(m);            // 4-connected components
        if (c > 1) s += (c - 1) * penalty; // penalize splits beyond 1 piece
    }
    return s;
}

ObjectiveResult objective(const Labels& LA, const Labels& LB,
                          const Mask& Afg, const Mask& Bfg,
                          int level, const Args& args) {
    ObjectiveResult out{};
    MatchResult mr = match_labels(LA, LB, Bfg);
    const double lam = level_lambda(level, args.lambda0, args.decay);
    const double area = area_penalty(LA, Afg, lam) + area_penalty(LB, Bfg, lam);
    const double spl  = split_penalty(LA, args.split_penalty) + split_penalty(LB, args.split_penalty);
    out.mismatch = mr.totalMismatch;
    out.area     = area;
    out.split    = spl;
    out.total    = out.mismatch + area + spl;
    return out;
}

double em_only(const Labels& LA, const Labels& LB, const Mask& Bfg) {
    return match_labels(LA, LB, Bfg).totalMismatch;
}
// ----------------- Refinement loop -----------------
RefineResult refine(Labels LA, Labels LB,
                    const Mask& Afg, const Mask& Bfg,
                    int /*level*/, const Args& args, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));

    Labels curA = LA, curB = LB;
    Labels bestA = curA, bestB = curB;
    double best_em = em_only(curA, curB, Bfg);

    for (int it = 0; it < args.R; ++it) {
        // Forward: A -> B
        MatchResult mAB = match_labels(curA, curB, Bfg);
        if (mAB.M.empty()) break;

        Labels B1 = forward_copy(curA, curB, mAB, Bfg, rng);
        B1 = fill_holes_random_flood(B1, Bfg, rng);
        Mask confB = overlap_mask_src_to_dst(curA, B1, Bfg);
        B1 = random_switch(B1, Bfg, confB, rng);

        // Backward: B -> A
        MatchResult mBA = match_labels(B1, curA, Afg);
        if (mBA.M.empty()) break;

        Labels A2 = forward_copy(B1, curA, mBA, Afg, rng);
        A2 = fill_holes_random_flood(A2, Afg, rng);
        Mask confA = overlap_mask_src_to_dst(B1, A2, Afg);
        A2 = random_switch(A2, Afg, confA, rng);

        curA = std::move(A2);
        curB = std::move(B1);

        // Keep the best by EM (mismatch-only) as in the paper
        double em = em_only(curA, curB, Bfg);
        if (em + 1e-12 < best_em) {
            best_em = em;
            bestA = curA;
            bestB = curB;
        }
    }

    return {bestA, bestB, best_em};
}

// ----------------- Level 1 (serial per trial; trials parallelized) -----------------
Labels two_seed_grow(const Mask& fg, Point s1, Point s2) {
    const int H = fg.rows(), W = fg.cols();
    Labels L(H, W, 0);
    auto inb = [&](int y, int x){ return 0 <= y && y < H && 0 <= x && x < W; };
    if (!inb(s1.first, s1.second) || !inb(s2.first, s2.second)) return L;
    if (!fg.at(s1.first, s1.second) || !fg.at(s2.first, s2.second)) return L;

    std::deque<Point> q1, q2;
    std::vector<uint8_t> seen(static_cast<size_t>(H)*W, uint8_t(0));
    auto S = [&](int y, int x)->uint8_t& { return seen[static_cast<size_t>(y)*W + x]; };

    L.at(s1.first, s1.second) = 1; q1.emplace_back(s1);
    L.at(s2.first, s2.second) = 2; q2.emplace_back(s2);
    S(s1.first, s1.second) = 1; S(s2.first, s2.second) = 1;

    const int dy4[4] = {1,-1,0,0};
    const int dx4[4] = {0,0,1,-1};

    while (!q1.empty() || !q2.empty()) {
        if (!q1.empty()) {
            auto [y, x] = q1.front(); q1.pop_front();
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy4[k], nx = x + dx4[k];
                if (!inb(ny, nx) || !fg.at(ny, nx) || S(ny, nx)) continue;
                S(ny, nx) = 1; L.at(ny, nx) = 1; q1.emplace_back(ny, nx);
            }
        }
        if (!q2.empty()) {
            auto [y, x] = q2.front(); q2.pop_front();
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy4[k], nx = x + dx4[k];
                if (!inb(ny, nx) || !fg.at(ny, nx) || S(ny, nx)) continue;
                S(ny, nx) = 1; L.at(ny, nx) = 2; q2.emplace_back(ny, nx);
            }
        }
    }

    // Fill any leftover foreground to the larger cluster
    long long a1 = 0, a2 = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (!fg.at(y,x)) continue;
            if (L.at(y,x) == 1) ++a1;
            else if (L.at(y,x) == 2) ++a2;
        }
    int fill_lab = (a1 >= a2) ? 1 : 2;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (fg.at(y,x) && L.at(y,x) == 0) L.at(y,x) = fill_lab;

    return L;
}

static inline std::vector<Point> _fg_pixels(const Mask& fg) {
    std::vector<Point> pts;
    const int H = fg.rows(), W = fg.cols();
    pts.reserve(static_cast<size_t>(H)*W/4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (fg.at(y,x)) pts.emplace_back(y,x);
    return pts;
}

RefineResult _level1_trial(const Mask& Afg, const Mask& Bfg, const Args& args, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    auto Apts = _fg_pixels(Afg);
    auto Bpts = _fg_pixels(Bfg);
    if (Apts.size() < 2 || Bpts.size() < 2) return { Labels(), Labels(), std::numeric_limits<double>::infinity() };

    std::uniform_int_distribution<int> dA(0, static_cast<int>(Apts.size())-1);
    std::uniform_int_distribution<int> dB(0, static_cast<int>(Bpts.size())-1);

    Point a1 = Apts[dA(rng)], a2 = Apts[dA(rng)];
    while (a2 == a1) a2 = Apts[dA(rng)];
    Point b1 = Bpts[dB(rng)], b2 = Bpts[dB(rng)];
    while (b2 == b1) b2 = Bpts[dB(rng)];

    Labels LA = two_seed_grow(Afg, a1, a2);
    Labels LB = two_seed_grow(Bfg, b1, b2);

    // One full refinement chain (A->B, then B->A), repeating R times
    RefineResult rr = refine(LA, LB, Afg, Bfg, 1, args, seed + 13);
    return rr;
}

std::vector<RefineResult> level1(const Mask& Afg, const Mask& Bfg, const Args& args, ThreadPool& pool) {
    const int N = args.N;
    std::vector<uint64_t> seeds; seeds.reserve(N);
    for (int i = 0; i < N; ++i) seeds.push_back(args.seed + 1009ull*i + 17ull);

    std::vector<std::future<RefineResult>> futs;
    futs.reserve(N);

    // Parallelize trials; each trial has its own RNG seed
    for (auto s : seeds) {
        futs.emplace_back(pool.enqueue([&Afg, &Bfg, &args, s](){
            return _level1_trial(Afg, Bfg, args, s);
        }));
    }

    std::vector<RefineResult> outs;
    outs.reserve(N);
    for (auto &f : futs) {
        RefineResult r = f.get();
        if (r.LA.rows() > 0 && r.LB.rows() > 0) outs.push_back(std::move(r));
    }
    if (outs.empty()) return outs;

    // Sort: either by full objective (if λ or split penalty is active) or EM only
    const bool use_full = (args.lambda0 > 0.0) || (args.split_penalty > 0.0);
    auto score = [&](const RefineResult& r)->double {
        if (!use_full) return r.em;
        int pcs = static_cast<int>(cluster_ids(r.LA).size());
        int level = std::max(0, pcs - 1);
        auto obj = objective(r.LA, r.LB, Afg, Bfg, level, args);
        return obj.total;
    };

    std::sort(outs.begin(), outs.end(),
        [&](const RefineResult& a, const RefineResult& b){ return score(a) < score(b); });

    if (static_cast<int>(outs.size()) > args.Nb) outs.resize(args.Nb);
    return outs;
}

// ----------------- Level >=2 helpers -----------------
std::pair<Point, double> worst_pair(const Labels& LA, const Labels& LB, const Mask& Bfg) {
    MatchResult mr = match_labels(LA, LB, Bfg);
    Point ab = {0,0};
    double worst_cost = -1.0;
    for (auto &kv : mr.M) {
        int a = kv.first, b = kv.second;
        auto it = mr.T.find({a,b});
        if (it == mr.T.end()) continue;
        double c = it->second.mismatch;
        if (c > worst_cost) { worst_cost = c; ab = {a,b}; }
    }
    return {ab, worst_cost};
}

std::pair<Labels, Labels> best_2piece_split_via_level1(const Mask& mA, const Mask& mB,
                                                       const Args& args, uint64_t seed) {
    // Run a small, serial Level-1 on the parent masks to find the best 2-split
    const int N_sub = 60, R_sub = 60;
    Args sub = args;
    sub.N = N_sub; sub.R = R_sub; sub.Nb = 1;
    sub.seed = seed + 811;

    // We can't reuse the thread pool (risk of deadlock), so just run serially
    std::vector<RefineResult> outs;
    outs.reserve(N_sub);
    for (int i = 0; i < N_sub; ++i) {
        uint64_t s = sub.seed + 1009ull*i + 17ull;
        RefineResult r = _level1_trial(mA, mB, sub, s);
        if (r.LA.rows() > 0) outs.push_back(std::move(r));
    }
    if (outs.empty()) return { Labels(), Labels() };

    const bool use_full = (sub.lambda0 > 0.0) || (sub.split_penalty > 0.0);
    auto score = [&](const RefineResult& r)->double {
        if (!use_full) return r.em;
        int pcs = static_cast<int>(cluster_ids(r.LA).size());
        int level = std::max(0, pcs - 1);
        auto obj = objective(r.LA, r.LB, mA, mB, level, sub);
        return obj.total;
    };
    std::sort(outs.begin(), outs.end(),
              [&](const RefineResult& a, const RefineResult& b){ return score(a) < score(b); });

    return { outs.front().LA, outs.front().LB };
}

std::pair<Labels, Labels> local_split(const Labels& LA, const Labels& LB,
                                      const Mask& Afg, const Mask& Bfg,
                                      const Args& args, uint64_t seed) {
    // Paper heuristic #1: split the worst-matched parent pair via Level-1 on the parent masks,
    // with a randomized two-seed fallback if the Level-1 split fails.
    auto [ab, worst_cost] = worst_pair(LA, LB, Bfg);
    if (worst_cost < 0.0) return {LA, LB}; // no pairs

    int a = ab.first, b = ab.second;
    Mask mA = mask_of(LA, a);
    Mask mB = mask_of(LB, b);

    // If either mask is too small to split, abort.
    long long areaA = 0, areaB = 0;
    for (auto v : mA.m_data) areaA += (v ? 1 : 0);
    for (auto v : mB.m_data) areaB += (v ? 1 : 0);
    if (areaA < 2 || areaB < 2) return {LA, LB};

    // --- Try the paper's optimal split first (Level-1 on parent masks) ---
    auto [bestA, bestB] = best_2piece_split_via_level1(
        mA, mB, args, seed
    );

    // --- Fallback: randomized two-seed split inside the parent masks ---
    if (bestA.rows() == 0 || bestB.rows() == 0) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        auto ptsA = _fg_pixels(mA);
        auto ptsB = _fg_pixels(mB);
        if (ptsA.size() < 2 || ptsB.size() < 2) return {LA, LB};
        std::uniform_int_distribution<int> dA(0, static_cast<int>(ptsA.size())-1);
        std::uniform_int_distribution<int> dB(0, static_cast<int>(ptsB.size())-1);
        Point a1 = ptsA[dA(rng)], a2 = ptsA[dA(rng)];
        while (a2 == a1) a2 = ptsA[dA(rng)];
        Point b1 = ptsB[dB(rng)], b2 = ptsB[dB(rng)];
        while (b2 == b1) b2 = ptsB[dB(rng)];
        bestA = two_seed_grow(mA, a1, a2);
        bestB = two_seed_grow(mB, b1, b2);
    }

    // Splice new subclusters back into the global labels
    int max_id = 0;
    for (int y = 0; y < LA.rows(); ++y)
        for (int x = 0; x < LA.cols(); ++x)
            if (LA.at(y,x) > max_id) max_id = LA.at(y,x);
    for (int y = 0; y < LB.rows(); ++y)
        for (int x = 0; x < LB.cols(); ++x)
            if (LB.at(y,x) > max_id) max_id = std::max(max_id, LB.at(y,x));
    int new_id = max_id + 1;

    Labels LA2 = LA, LB2 = LB;
    for (int y = 0; y < LA.rows(); ++y)
        for (int x = 0; x < LA.cols(); ++x)
            if (mA.at(y,x) && bestA.at(y,x) == 2) LA2.at(y,x) = new_id;
    for (int y = 0; y < LB.rows(); ++y)
        for (int x = 0; x < LB.cols(); ++x)
            if (mB.at(y,x) && bestB.at(y,x) == 2) LB2.at(y,x) = new_id;

    return {LA2, LB2};
}


std::pair<Labels, Labels> new_cluster_from_mismatches(const Labels& LA, const Labels& LB,
                                                      const Mask& Afg, const Mask& Bfg,
                                                      std::mt19937& rng) {
    // Paper heuristic #2: build a new cluster from mismatched pixels (seed + flood-fill).
    Mask misB = mismatches_src_outside_dst(LA, LB, Bfg); // A->B mismatches
    Mask misA = mismatches_src_outside_dst(LB, LA, Afg); // B->A mismatches

    auto count_mask = [](const Mask& m)->long long {
        long long s = 0; for (auto v : m.m_data) s += (v ? 1 : 0); return s;
    };
    long long cntA = count_mask(misA), cntB = count_mask(misB);
    if (cntA == 0 && cntB == 0) return {LA, LB};

    // Choose which side to create on (weighted by counts if both nonzero)
    bool chooseA = false;
    if (cntA > 0 && cntB > 0) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double p = static_cast<double>(cntA) / static_cast<double>(cntA + cntB);
        chooseA = (U(rng) < p);
    } else if (cntA > 0) chooseA = true; else chooseA = false;

    auto flood_into_new = [&](const Labels& L, const Mask& mask, int new_id)->Labels {
        Labels out = L;
        const int H = L.rows(), W = L.cols();
        std::vector<Point> seeds;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (mask.at(y,x)) seeds.emplace_back(y,x);
        if (seeds.empty()) return out;
        std::uniform_int_distribution<int> d(0, static_cast<int>(seeds.size())-1);
        Point s = seeds[d(rng)];

        std::deque<Point> q; q.emplace_back(s);
        if (out.at(s.first, s.second) != new_id) out.at(s.first, s.second) = new_id;
        const int dy4[4] = {1,-1,0,0};
        const int dx4[4] = {0,0,1,-1};
        auto inb = [&](int y, int x){ return 0 <= y && y < H && 0 <= x && x < W; };

        while (!q.empty()) {
            auto [y, x] = q.front(); q.pop_front();
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy4[k], nx = x + dx4[k];
                if (!inb(ny, nx) || !mask.at(ny,nx)) continue;
                if (out.at(ny,nx) == new_id) continue;
                out.at(ny,nx) = new_id;
                q.emplace_back(ny, nx);
            }
        }
        return out;
    };

    // Compute next new label id
    int max_id = 0;
    for (int y = 0; y < LA.rows(); ++y)
        for (int x = 0; x < LA.cols(); ++x)
            if (LA.at(y,x) > max_id) max_id = LA.at(y,x);
    for (int y = 0; y < LB.rows(); ++y)
        for (int x = 0; x < LB.cols(); ++x)
            if (LB.at(y,x) > max_id) max_id = std::max(max_id, LB.at(y,x));
    int new_id = max_id + 1;

    Labels LA2 = LA, LB2 = LB;
    if (chooseA) {
        LA2 = flood_into_new(LA, misA, new_id);
    } else {
        LB2 = flood_into_new(LB, misB, new_id);
    }

    // Harmonize #clusters if they diverge by 1 (peel a pixel from the largest cluster)
    auto clusters = [&](const Labels& L)->std::vector<int> { return cluster_ids(L); };
    auto peel = [&](const Labels& L, int nid)->Labels {
        Labels out = L;
        auto A = areas(out);
        if (A.empty()) return out;
        int big = 0, bigsz = -1;
        for (auto &kv : A) if (kv.second > bigsz) { big = kv.first; bigsz = kv.second; }
        for (int y = 0; y < out.rows(); ++y) {
            bool done = false;
            for (int x = 0; x < out.cols(); ++x) {
                if (out.at(y,x) == big) { out.at(y,x) = nid; done = true; break; }
            }
            if (done) break;
        }
        return out;
    };

    auto ca = clusters(LA2), cb = clusters(LB2);
    if (ca.size() > cb.size()) {
        LB2 = peel(LB2, new_id);
    } else if (cb.size() > ca.size()) {
        LA2 = peel(LA2, new_id);
    }

    return {LA2, LB2};
}

// ----------------- Level >=2 random exploration -----------------
RefineResult _levelK_trial(const RefineResult& cand,
                           const Mask& Afg, const Mask& Bfg,
                           int level, const Args& args, uint64_t seed) {
    // One trial at Level >=2 using a local RNG seeded per trial (deterministic across threads).
    Labels LA = cand.LA;
    Labels LB = cand.LB;

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // Heuristic choice: 75% local split (H1), else mismatches (H2)
    if (U(rng) < 0.75) {
        auto pair = local_split(LA, LB, Afg, Bfg, args, seed + 37);
        LA = std::move(pair.first);
        LB = std::move(pair.second);
    } else {
        auto pair = new_cluster_from_mismatches(LA, LB, Afg, Bfg, rng);
        LA = std::move(pair.first);
        LB = std::move(pair.second);
    }

    return refine(LA, LB, Afg, Bfg, level, args, seed + 73);
}

std::vector<RefineResult> explore_next(const std::vector<RefineResult>& prev,
                                       const Mask& Afg, const Mask& Bfg,
                                       int level, const Args& args, ThreadPool& pool,
                                       uint64_t level_seed) {
    std::vector<RefineResult> outs;
    if (prev.empty()) return outs;

    const int N = args.N;
    std::vector<uint64_t> seeds; seeds.reserve(N);
    for (int i = 0; i < N; ++i) seeds.push_back(level_seed + 2003ull*i + 23ull);

    // Deterministic candidate pick per trial
    std::vector<const RefineResult*> picks; picks.reserve(N);
    std::mt19937 picks_rng(static_cast<uint32_t>(level_seed)); // Use the level_seed
    std::uniform_int_distribution<int> d(0, static_cast<int>(prev.size()) - 1);
    for (int i = 0; i < N; ++i) {
        picks.push_back(&prev[d(picks_rng)]);
    }

    // Parallel trials
    std::vector<std::future<RefineResult>> futs; futs.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto s = seeds[i];
        const RefineResult* c = picks[i];
        futs.emplace_back(pool.enqueue([=, &Afg, &Bfg, &args](){
            return _levelK_trial(*c, Afg, Bfg, level, args, s);
        }));
    }

    outs.reserve(N);
    for (auto &f : futs) outs.push_back(f.get());
    if (outs.empty()) return outs;

    // Sort: either full objective (if λ or split penalty > 0) or EM-only
    const bool use_full = (args.lambda0 > 0.0) || (args.split_penalty > 0.0);
    auto score = [&](const RefineResult& r)->double {
        if (!use_full) return r.em;
        int pcs = static_cast<int>(cluster_ids(r.LA).size());
        int lvl = std::max(0, pcs - 1);
        auto obj = objective(r.LA, r.LB, Afg, Bfg, lvl, args);
        return obj.total;
    };

    std::sort(outs.begin(), outs.end(),
              [&](const RefineResult& a, const RefineResult& b){ return score(a) < score(b); });
    if (static_cast<int>(outs.size()) > args.Nb) outs.resize(args.Nb);
    return outs;
}
// ----------------- Runner -----------------
int run(Args& args) {
    // --- Setup / I/O ---
    Mask A, B;
    try {
        A = load_binary(args.shapeA_path);
        B = load_binary(args.shapeB_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading images: " << e.what() << std::endl;
        return 1;
    }

    auto sum_mask = [](const Mask& m)->long long {
        long long s = 0; for (auto v : m.m_data) s += (v ? 1 : 0); return s;
    };

    long long a_area = sum_mask(A);
    long long b_area = sum_mask(B);
    if (a_area <= 0 || b_area <= 0) {
        std::cerr << "Error: empty shapes (no foreground pixels)." << std::endl;
        return 2;
    }
    if (a_area != b_area) {
        std::cerr << "Error: shapes must have equal area. A=" << a_area << " B=" << b_area << std::endl;
        return 3;
    }

    int num_threads = args.jobs;
    if (num_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        num_threads = (hc == 0 ? 1 : static_cast<int>(hc));
    }
    ThreadPool pool(static_cast<size_t>(num_threads));
    std::cout << "Using " << num_threads << " threads." << std::endl;

    // --- Level 1 ---
    std::cout << "\n=== Level 1 ===" << std::endl;
    auto L1_results = level1(A, B, args, pool);
    if (L1_results.empty()) {
        std::cerr << "Level 1 failed." << std::endl;
        return 4;
    }

    // Best after L1
    RefineResult best = L1_results.front();
    int pcs = static_cast<int>(cluster_ids(best.LA).size());
    int lvl = std::max(0, pcs - 1);
    auto obj1 = objective(best.LA, best.LB, A, B, lvl, args);
    std::cout << "Best L1: Obj=" << std::fixed << std::setprecision(3) << obj1.total
              << "  M=" << std::setprecision(1) << obj1.mismatch
              << "  Area=" << std::setprecision(1) << obj1.area
              << "  Split=" << std::setprecision(0) << obj1.split << std::endl;

    if (obj1.mismatch < CONVERGENCE_EPS && obj1.split < 1.0) {
        std::cout << "*** Perfect dissection at L1 ***" << std::endl;
        try {
            std::filesystem::create_directories(args.outdir);
            save_labels_png(args.outdir + "/result_A_" + std::to_string(pcs) + "pcs.png", best.LA);
            save_labels_png(args.outdir + "/result_B_" + std::to_string(pcs) + "pcs.png", best.LB);
        } catch (const std::exception& e) {
            std::cerr << "Save error: " << e.what() << std::endl;
        }
        return 0;
    }

    // --- Levels >=2 ---
    std::vector<RefineResult> cands = std::move(L1_results);
    RefineResult best_overall = best;

    for (int level = 2; level <= args.max_level; ++level) {
        std::cout << "\n=== Level " << level << " (" << (level+1) << " clusters) ===" << std::endl;
        // Use a different seed for each level's exploration
        uint64_t current_level_seed = args.seed + static_cast<uint64_t>(level) * 10007ull;
        auto next = explore_next(cands, A, B, level, args, pool, current_level_seed);
        if (next.empty()) {
            std::cout << "No candidates; stopping exploration." << std::endl;
            break;
        }
        cands = std::move(next);

        RefineResult cur = cands.front();
        int pcs_cur = static_cast<int>(cluster_ids(cur.LA).size());
        int lvl_cur = std::max(0, pcs_cur - 1);
        auto obj = objective(cur.LA, cur.LB, A, B, lvl_cur, args);

        std::cout << "Best L" << level << ": Obj=" << std::fixed << std::setprecision(3) << obj.total
                  << "  M=" << std::setprecision(1) << obj.mismatch
                  << "  Area=" << std::setprecision(1) << obj.area
                  << "  Split=" << std::setprecision(0) << obj.split << std::endl;

        if (obj.mismatch < CONVERGENCE_EPS && obj.split < 1.0) {
            std::cout << "*** Perfect dissection at L" << level << " ***" << std::endl;
            best_overall = cur;
            break;
        }

        // Keep best overall by objective total
        auto best_over_obj_lvl = std::max(0, static_cast<int>(cluster_ids(best_overall.LA).size()) - 1);
        auto best_over_obj = objective(best_overall.LA, best_overall.LB, A, B, best_over_obj_lvl, args);
        if (obj.total + 1e-12 < best_over_obj.total) best_overall = cur;
    }

    // --- Save final result ---
    pcs = static_cast<int>(cluster_ids(best_overall.LA).size());
    lvl = std::max(0, pcs - 1);
    auto objF = objective(best_overall.LA, best_overall.LB, A, B, lvl, args);

    std::cout << "\n=== Final ===" << std::endl;
    std::cout << "Pieces: " << pcs
              << "\nObjective: " << std::fixed << std::setprecision(3) << objF.total
              << "  Mismatch: " << std::setprecision(1) << objF.mismatch
              << "  Area: " << std::setprecision(1) << objF.area
              << "  Split: " << std::setprecision(0) << objF.split << std::endl;

    try {
        std::filesystem::create_directories(args.outdir);
        save_labels_png(args.outdir + "/result_A_" + std::to_string(pcs) + "pcs.png", best_overall.LA);
        save_labels_png(args.outdir + "/result_B_" + std::to_string(pcs) + "pcs.png", best_overall.LB);
    } catch (const std::exception& e) {
        std::cerr << "Save error: " << e.what() << std::endl;
    }

    return (objF.mismatch < CONVERGENCE_EPS && objF.split < 1.0) ? 0 : 1;
}

// ----------------- Main -----------------
int main2d_reference(int argc, char* argv[]) {
    // --- Argument parsing ---
    Args args;
    
    // Check for basic args
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <shapeA.png> <shapeB.png> [options...]" << std::endl;
        std::cerr << "Example: " << argv[0] << " shape_A2.png shape_B2.png -N 400 -j -1 -o out_dir" << std::endl;
        std::cerr << "\nNote: This is a demo. Full argument parsing is not implemented." << std::endl;
        return 1;
    }
    
    args.shapeA_path = argv[1];
    args.shapeB_path = argv[2];
    
    // Manual argument parser
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i+1 < argc) { args.outdir = argv[++i]; }
        else if (a == "-R" && i+1 < argc) { args.R = std::stoi(argv[++i]); }
        else if (a == "-N" && i+1 < argc) { args.N = std::stoi(argv[++i]); }
        else if (a == "-Nb" && i+1 < argc) { args.Nb = std::stoi(argv[++i]); }
        else if (a == "--max-level" && i+1 < argc) { args.max_level = std::stoi(argv[++i]); }
        else if (a == "--lambda0" && i+1 < argc) { args.lambda0 = std::stod(argv[++i]); }
        else if (a == "--decay" && i+1 < argc) { args.decay = std::stod(argv[++i]); }
        else if (a == "--split-penalty" && i+1 < argc) { args.split_penalty = std::stod(argv[++i]); }
        else if (a == "--seed" && i+1 < argc) { args.seed = std::stoull(argv[++i]); }
        else if (a == "-j" && i+1 < argc) { args.jobs = std::stoi(argv[++i]); }
    }
    
    if (args.outdir.empty()) {
        args.outdir = "out_dir_cpp";
    }
    
    std::cout << "Starting C++ dissection..." << std::endl;
    std::cout << "  Shape A: " << args.shapeA_path << std::endl;
    std::cout << "  Shape B: " << args.shapeB_path << std::endl;
    std::cout << "  Params: -N " << args.N << " -Nb " << args.Nb << " -R " << args.R 
              << " -j " << args.jobs << " --seed " << args.seed << std::endl;


    int return_code = 1;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        return_code = run(args);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Total runtime: " << std::fixed << std::setprecision(2) << diff.count() << "s" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An unhandled error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Finished." << std::endl;
    return return_code;
}

// ============================================================================
// 3D version. The 2D implementation above is kept as the reference; the
// code below follows the same stages:
// load binary shape, random two-seed growth, piece matching, copy-paste,
// random hole fill, random conflict switching, worst-pair split, mismatch
// cluster insertion, beam pruning, and final label export.
// ============================================================================

namespace zhou3d {

template <typename T>
class Volume {
public:
    int d = 0, h = 0, w = 0;
    std::vector<T> data;
    Volume() = default;
    Volume(int D, int H, int W, T val = T()) : d(D), h(H), w(W), data(static_cast<size_t>(D) * H * W, val) {}
    T& at(int z, int y, int x) { return data[(static_cast<size_t>(z) * h + y) * w + x]; }
    const T& at(int z, int y, int x) const { return data[(static_cast<size_t>(z) * h + y) * w + x]; }
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

using Mask = Volume<uint8_t>;
using Labels = Volume<int>;

struct Vec {
    int z = 0, y = 0, x = 0;
    bool operator==(const Vec& o) const { return z == o.z && y == o.y && x == o.x; }
};

struct Rot {
    std::array<int,3> axis{};
    std::array<int,3> sign{};
};

struct Params {
    int rot = 0;
    Vec shift{};
};

struct PairCost {
    double mismatch = std::numeric_limits<double>::infinity();
    Params params;
};

struct Match {
    std::map<int,int> M;
    std::map<std::pair<int,int>, PairCost> T;
    double total = 0.0;
};

struct Refine {
    Labels A;
    Labels B;
    double em = 0.0;
};

struct Args3 {
    std::string a_dir, b_dir, outdir = "out_zhou3d";
    std::string resume_a_dir, resume_b_dir;
    std::string resume_beam_dir;
    int R = R_DEFAULT, N = N_DEFAULT, Nb = NB_DEFAULT;
    int max_level = 5;          // level 5 -> 6 pieces
    int parent_N = 60, parent_R = 60;
    double lambda0 = LAMBDA0_DEFAULT, decay = DECAY_DEFAULT, split_penalty = SPLIT_PENALTY_DEFAULT;
    uint64_t seed = SEED_DEFAULT;
    int jobs = -1;
    bool fixed_final = true;
    bool require_connected = false;
    bool save_beam = false;
    int checkpoint_every = 10;
};

int flat(int z, int y, int x, int H, int W) { return (z * H + y) * W + x; }

std::vector<Rot> make_rots() {
    std::vector<Rot> out;
    std::array<int,3> p = {0,1,2};
    do {
        int inv = (p[0] > p[1]) + (p[0] > p[2]) + (p[1] > p[2]);
        int parity = (inv % 2) ? -1 : 1;
        for (int s0 : {-1,1}) for (int s1 : {-1,1}) for (int s2 : {-1,1}) {
            if (parity * s0 * s1 * s2 == 1) out.push_back({p, {s0,s1,s2}});
        }
    } while (std::next_permutation(p.begin(), p.end()));
    return out;
}

const std::vector<Rot>& rots() {
    static std::vector<Rot> r = make_rots();
    return r;
}

std::vector<std::filesystem::path> png_slices(const std::string& dir) {
    std::vector<std::filesystem::path> files;
    for (auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (ext == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("no PNG slices in " + dir);
    return files;
}

Mask load_stack(const std::string& dir, int thr = 128) {
    auto files = png_slices(dir);
    int W = -1, H = -1;
    Mask out;
    for (int z = 0; z < static_cast<int>(files.size()); ++z) {
        int w = 0, h = 0, comp = 0;
        unsigned char* img = stbi_load(files[z].string().c_str(), &w, &h, &comp, 1);
        if (!img) throw std::runtime_error("stbi_load failed: " + files[z].string());
        if (z == 0) {
            W = w; H = h; out = Mask(static_cast<int>(files.size()), H, W, uint8_t(0));
        } else if (w != W || h != H) {
            stbi_image_free(img);
            throw std::runtime_error("slice dimensions differ in " + dir);
        }
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            out.at(z,y,x) = (img[y * W + x] < thr) ? 1 : 0;
        }
        stbi_image_free(img);
    }
    return out;
}

long long area(const Mask& m) {
    long long s = 0;
    for (auto v : m.data) s += v ? 1 : 0;
    return s;
}

std::tuple<uint8_t,uint8_t,uint8_t> color(int id) {
    if (id <= 0) return {255,255,255};
    uint32_t v = static_cast<uint32_t>(id) * 0x9E3779B1u;
    v ^= v >> 16; v *= 0x85EBCA6Bu; v ^= v >> 13; v *= 0xC2B2AE35u; v ^= v >> 16;
    uint8_t r = static_cast<uint8_t>(v & 255), g = static_cast<uint8_t>((v >> 8) & 255), b = static_cast<uint8_t>((v >> 16) & 255);
    if (r < 24 && g < 24 && b < 24) { r ^= 0x55; g ^= 0xAA; b ^= 0xCC; }
    return {r,g,b};
}

void save_stack(const std::string& dir, const Labels& L) {
    std::filesystem::create_directories(dir);
    for (int z = 0; z < L.d; ++z) {
        std::vector<uint8_t> rgb(static_cast<size_t>(L.h) * L.w * 3, 255);
        for (int y = 0; y < L.h; ++y) for (int x = 0; x < L.w; ++x) {
            auto [r,g,b] = color(L.at(z,y,x));
            size_t i = (static_cast<size_t>(y) * L.w + x) * 3;
            rgb[i] = r; rgb[i+1] = g; rgb[i+2] = b;
        }
        char name[64];
        std::snprintf(name, sizeof(name), "z_%04d.png", z);
        auto path = std::filesystem::path(dir) / name;
        if (!stbi_write_png(path.string().c_str(), L.w, L.h, 3, rgb.data(), L.w * 3)) {
            throw std::runtime_error("stbi_write_png failed: " + path.string());
        }
    }
}

Labels load_label_stack(const std::string& dir, int max_label = 2048) {
    auto files = png_slices(dir);
    int W = -1, H = -1;
    Labels out;
    std::map<std::tuple<uint8_t,uint8_t,uint8_t>, int> inverse;
    for (int id = 1; id <= max_label; ++id) inverse[color(id)] = id;
    for (int z = 0; z < static_cast<int>(files.size()); ++z) {
        int w = 0, h = 0, comp = 0;
        unsigned char* img = stbi_load(files[z].string().c_str(), &w, &h, &comp, 3);
        if (!img) throw std::runtime_error("stbi_load failed: " + files[z].string());
        if (z == 0) {
            W = w; H = h; out = Labels(static_cast<int>(files.size()), H, W, 0);
        } else if (w != W || h != H) {
            stbi_image_free(img);
            throw std::runtime_error("slice dimensions differ in " + dir);
        }
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            size_t i = (static_cast<size_t>(y) * W + x) * 3;
            uint8_t r = img[i], g = img[i+1], b = img[i+2];
            if (r == 255 && g == 255 && b == 255) {
                out.at(z,y,x) = 0;
                continue;
            }
            auto it = inverse.find({r,g,b});
            if (it == inverse.end()) {
                stbi_image_free(img);
                throw std::runtime_error("unknown label color in " + files[z].string());
            }
            out.at(z,y,x) = it->second;
        }
        stbi_image_free(img);
    }
    return out;
}

std::vector<int> ids(const Labels& L) {
    std::set<int> s;
    for (int z = 0; z < L.d; ++z) for (int y = 0; y < L.h; ++y) for (int x = 0; x < L.w; ++x)
        if (L.at(z,y,x) > 0) s.insert(L.at(z,y,x));
    return {s.begin(), s.end()};
}

std::map<int,int> areas(const Labels& L) {
    std::map<int,int> a;
    for (int z = 0; z < L.d; ++z) for (int y = 0; y < L.h; ++y) for (int x = 0; x < L.w; ++x)
        if (L.at(z,y,x) > 0) ++a[L.at(z,y,x)];
    return a;
}

Mask mask_of(const Labels& L, int id) {
    Mask m(L.d, L.h, L.w, 0);
    for (int z = 0; z < L.d; ++z) for (int y = 0; y < L.h; ++y) for (int x = 0; x < L.w; ++x)
        if (L.at(z,y,x) == id) m.at(z,y,x) = 1;
    return m;
}

std::vector<Vec> coords(const Labels& L, int id) {
    std::vector<Vec> c;
    for (int z = 0; z < L.d; ++z) for (int y = 0; y < L.h; ++y) for (int x = 0; x < L.w; ++x)
        if (L.at(z,y,x) == id) c.push_back({z,y,x});
    return c;
}

std::vector<Vec> voxels(const Mask& M) {
    std::vector<Vec> c;
    for (int z = 0; z < M.d; ++z) for (int y = 0; y < M.h; ++y) for (int x = 0; x < M.w; ++x)
        if (M.at(z,y,x)) c.push_back({z,y,x});
    return c;
}

std::tuple<double,double,double> centroid(const std::vector<Vec>& c) {
    if (c.empty()) return {-1,-1,-1};
    long long sz = 0, sy = 0, sx = 0;
    for (auto& p : c) { sz += p.z; sy += p.y; sx += p.x; }
    double n = static_cast<double>(c.size());
    return {sz/n, sy/n, sx/n};
}

std::tuple<int,int,int,int,int,int> bbox(const Mask& M) {
    int z0=M.d, y0=M.h, x0=M.w, z1=-1, y1=-1, x1=-1;
    for (int z = 0; z < M.d; ++z) for (int y = 0; y < M.h; ++y) for (int x = 0; x < M.w; ++x) if (M.at(z,y,x)) {
        z0 = std::min(z0,z); y0 = std::min(y0,y); x0 = std::min(x0,x);
        z1 = std::max(z1,z); y1 = std::max(y1,y); x1 = std::max(x1,x);
    }
    if (z1 < z0) return {0,0,0,-1,-1,-1};
    return {z0,y0,x0,z1,y1,x1};
}

std::array<int,3> rotate_local(std::array<int,3> p, std::array<int,3> dims, const Rot& R) {
    std::array<int,3> q{};
    for (int i = 0; i < 3; ++i) {
        int a = R.axis[i];
        q[i] = R.sign[i] > 0 ? p[a] : dims[a] - 1 - p[a];
    }
    return q;
}

std::array<Vec,8> snaps(double dz, double dy, double dx) {
    int fz = static_cast<int>(std::floor(dz)), cz = static_cast<int>(std::ceil(dz));
    int fy = static_cast<int>(std::floor(dy)), cy = static_cast<int>(std::ceil(dy));
    int fx = static_cast<int>(std::floor(dx)), cx = static_cast<int>(std::ceil(dx));
    return {Vec{fz,fy,fx}, {fz,fy,cx}, {fz,cy,fx}, {fz,cy,cx}, {cz,fy,fx}, {cz,fy,cx}, {cz,cy,fx}, {cz,cy,cx}};
}

PairCost pair_mismatch(const std::vector<Vec>& A, const std::vector<Vec>& B, const Mask& Bm) {
    PairCost res;
    int Sa = static_cast<int>(A.size()), Sb = static_cast<int>(B.size());
    if (Sa == 0 && Sb == 0) { res.mismatch = 0; return res; }
    if (Sa == 0 || Sb == 0) { res.mismatch = std::max(Sa,Sb); return res; }

    int z0=INT_MAX,y0=INT_MAX,x0=INT_MAX,z1=-1,y1=-1,x1=-1;
    for (auto& p : A) { z0=std::min(z0,p.z); y0=std::min(y0,p.y); x0=std::min(x0,p.x); z1=std::max(z1,p.z); y1=std::max(y1,p.y); x1=std::max(x1,p.x); }
    std::array<int,3> dims = {z1-z0+1, y1-y0+1, x1-x0+1};
    std::vector<std::array<int,3>> local;
    for (auto& p : A) local.push_back({p.z-z0, p.y-y0, p.x-x0});
    auto [czB,cyB,cxB] = centroid(B);
    double lower = std::abs(Sa - Sb);
    const auto& R = rots();
    for (int ri = 0; ri < static_cast<int>(R.size()); ++ri) {
        std::vector<std::array<int,3>> rr;
        long long sz=0,sy=0,sx=0;
        for (auto& p : local) { auto q = rotate_local(p,dims,R[ri]); rr.push_back(q); sz+=q[0]; sy+=q[1]; sx+=q[2]; }
        double czA = static_cast<double>(sz)/rr.size() + z0, cyA = static_cast<double>(sy)/rr.size() + y0, cxA = static_cast<double>(sx)/rr.size() + x0;
        for (auto s : snaps(czB-czA, cyB-cyA, cxB-cxA)) {
            int overlap = 0;
            for (auto& p : rr) {
                int gz=p[0]+z0+s.z, gy=p[1]+y0+s.y, gx=p[2]+x0+s.x;
                if (0<=gz && gz<Bm.d && 0<=gy && gy<Bm.h && 0<=gx && gx<Bm.w && Bm.at(gz,gy,gx)) ++overlap;
            }
            double mis = static_cast<double>(Sa + Sb - 2 * overlap);
            if (mis < res.mismatch) {
                res.mismatch = mis; res.params.rot = ri; res.params.shift = s;
                if (mis <= lower + 1e-12) return res;
            }
        }
    }
    return res;
}

Match match_labels(const Labels& A, const Labels& B) {
    Match out;
    auto ia = ids(A), ib = ids(B);
    if (ia.empty() || ib.empty()) return out;
    std::map<int,std::vector<Vec>> ca, cb;
    std::map<int,Mask> bm;
    for (int a : ia) ca[a] = coords(A,a);
    for (int b : ib) { cb[b] = coords(B,b); bm[b] = mask_of(B,b); }
    Matrix<double> cost(static_cast<int>(ia.size()), static_cast<int>(ib.size()), std::numeric_limits<double>::infinity());
    for (int i = 0; i < static_cast<int>(ia.size()); ++i) for (int j = 0; j < static_cast<int>(ib.size()); ++j) {
        auto pc = pair_mismatch(ca[ia[i]], cb[ib[j]], bm[ib[j]]);
        cost.at(i,j) = pc.mismatch;
        out.T[{ia[i],ib[j]}] = pc;
    }
    auto asg = bellman_ford_assignment(cost);
    for (int i = 0; i < static_cast<int>(ia.size()); ++i) {
        int j = asg[i];
        if (j >= 0) { out.M[ia[i]] = ib[j]; out.total += cost.at(i,j); }
    }
    return out;
}

int components6(const Mask& M) {
    std::vector<uint8_t> seen(static_cast<size_t>(M.d)*M.h*M.w,0);
    int comps = 0;
    const int dz[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dx[6]={0,0,0,0,1,-1};
    std::deque<Vec> q;
    for (int z=0; z<M.d; ++z) for (int y=0; y<M.h; ++y) for (int x=0; x<M.w; ++x) {
        int id = flat(z,y,x,M.h,M.w);
        if (!M.at(z,y,x) || seen[id]) continue;
        ++comps; seen[id] = 1; q.push_back({z,y,x});
        while (!q.empty()) {
            Vec p=q.front(); q.pop_front();
            for (int k=0;k<6;++k) {
                int nz=p.z+dz[k], ny=p.y+dy[k], nx=p.x+dx[k];
                if (0<=nz && nz<M.d && 0<=ny && ny<M.h && 0<=nx && nx<M.w) {
                    int ni=flat(nz,ny,nx,M.h,M.w);
                    if (M.at(nz,ny,nx) && !seen[ni]) { seen[ni]=1; q.push_back({nz,ny,nx}); }
                }
            }
        }
    }
    return comps;
}

Mask apply_rotation(const Mask& M, const Params& P, const Mask& target) {
    Mask out(target.d,target.h,target.w,0);
    auto [z0,y0,x0,z1,y1,x1] = bbox(M);
    if (z1 < z0) return out;
    std::array<int,3> dims = {z1-z0+1, y1-y0+1, x1-x0+1};
    const Rot& R = rots()[P.rot];
    for (int z=z0; z<=z1; ++z) for (int y=y0; y<=y1; ++y) for (int x=x0; x<=x1; ++x) if (M.at(z,y,x)) {
        auto q = rotate_local({z-z0,y-y0,x-x0}, dims, R);
        int gz=q[0]+z0+P.shift.z, gy=q[1]+y0+P.shift.y, gx=q[2]+x0+P.shift.x;
        if (0<=gz && gz<target.d && 0<=gy && gy<target.h && 0<=gx && gx<target.w && target.at(gz,gy,gx)) out.at(gz,gy,gx)=1;
    }
    return out;
}

Labels forward_copy(const Labels& A, const Labels& B, const Match& m, const Mask& Bfg, std::mt19937& rng) {
    Labels out(B.d,B.h,B.w,0);
    std::vector<std::pair<int,int>> items(m.M.begin(), m.M.end());
    std::shuffle(items.begin(), items.end(), rng);
    for (auto& ab : items) {
        auto it = m.T.find({ab.first,ab.second});
        if (it == m.T.end()) continue;
        Mask placed = apply_rotation(mask_of(A,ab.first), it->second.params, Bfg);
        for (int z=0; z<B.d; ++z) for (int y=0; y<B.h; ++y) for (int x=0; x<B.w; ++x) if (placed.at(z,y,x)) out.at(z,y,x)=ab.second;
    }
    return out;
}

Labels fill_holes_random_flood(const Labels& L, const Mask& fg, std::mt19937& rng) {
    Labels out = L;
    const int dz[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dx[6]={0,0,0,0,1,-1};
    std::vector<Vec> frontier;
    for (int z=0; z<fg.d; ++z) for (int y=0; y<fg.h; ++y) for (int x=0; x<fg.w; ++x) if (fg.at(z,y,x) && out.at(z,y,x)>0) {
        for (int k=0;k<6;++k) { int nz=z+dz[k],ny=y+dy[k],nx=x+dx[k]; if (0<=nz&&nz<fg.d&&0<=ny&&ny<fg.h&&0<=nx&&nx<fg.w&&fg.at(nz,ny,nx)&&out.at(nz,ny,nx)==0) { frontier.push_back({z,y,x}); break; } }
    }
    std::shuffle(frontier.begin(), frontier.end(), rng);
    std::deque<Vec> q(frontier.begin(), frontier.end());
    while (!q.empty()) {
        Vec p=q.front(); q.pop_front();
        int lab=out.at(p.z,p.y,p.x);
        for (int k=0;k<6;++k) { int nz=p.z+dz[k],ny=p.y+dy[k],nx=p.x+dx[k]; if (0<=nz&&nz<fg.d&&0<=ny&&ny<fg.h&&0<=nx&&nx<fg.w&&fg.at(nz,ny,nx)&&out.at(nz,ny,nx)==0) { out.at(nz,ny,nx)=lab; q.push_back({nz,ny,nx}); } }
    }
    int fallback = ids(out).empty() ? 1 : ids(out).front();
    for (int z=0; z<fg.d; ++z) for (int y=0; y<fg.h; ++y) for (int x=0; x<fg.w; ++x) if (fg.at(z,y,x)&&out.at(z,y,x)==0) out.at(z,y,x)=fallback;
    return out;
}

Mask overlap_mask(const Labels& src, const Labels& dst, const Mask& fg) {
    Mask ov(fg.d,fg.h,fg.w,0);
    auto m=match_labels(src,dst);
    for (auto& kv:m.M) {
        auto it=m.T.find({kv.first,kv.second});
        if (it==m.T.end()) continue;
        Mask placed=apply_rotation(mask_of(src,kv.first),it->second.params,fg);
        for (int z=0; z<fg.d; ++z) for (int y=0; y<fg.h; ++y) for (int x=0; x<fg.w; ++x)
            if (fg.at(z,y,x)&&placed.at(z,y,x)&&dst.at(z,y,x)!=0&&dst.at(z,y,x)!=kv.second) ov.at(z,y,x)=1;
    }
    return ov;
}

Mask mismatch_mask(const Labels& src, const Labels& dst, const Mask& fg) {
    Mask mis(fg.d,fg.h,fg.w,0);
    auto m=match_labels(src,dst);
    for (auto& kv:m.M) {
        auto it=m.T.find({kv.first,kv.second});
        if (it==m.T.end()) continue;
        Mask placed=apply_rotation(mask_of(src,kv.first),it->second.params,fg);
        for (int z=0; z<fg.d; ++z) for (int y=0; y<fg.h; ++y) for (int x=0; x<fg.w; ++x)
            if (fg.at(z,y,x)&&placed.at(z,y,x)&&dst.at(z,y,x)!=kv.second) mis.at(z,y,x)=1;
    }
    return mis;
}

Labels random_switch(const Labels& L, const Mask& fg, const Mask& conflict, std::mt19937& rng) {
    Labels out=L;
    const int dz[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dx[6]={0,0,0,0,1,-1};
    std::vector<Vec> pts;
    for (int z=0; z<L.d; ++z) for (int y=0; y<L.h; ++y) for (int x=0; x<L.w; ++x) if (conflict.at(z,y,x)) pts.push_back({z,y,x});
    std::shuffle(pts.begin(), pts.end(), rng);
    for (auto& p:pts) {
        std::vector<int> n;
        for (int k=0;k<6;++k) { int nz=p.z+dz[k],ny=p.y+dy[k],nx=p.x+dx[k]; if (0<=nz&&nz<L.d&&0<=ny&&ny<L.h&&0<=nx&&nx<L.w&&fg.at(nz,ny,nx)&&out.at(nz,ny,nx)>0) n.push_back(out.at(nz,ny,nx)); }
        if (!n.empty()) { std::uniform_int_distribution<int> U(0,static_cast<int>(n.size())-1); out.at(p.z,p.y,p.x)=n[U(rng)]; }
    }
    return out;
}

double level_lambda3(int level,double l,double d){ return (l/2.0)*std::pow(d,static_cast<double>(std::max(0,level-1))); }

double area_penalty3(const Labels& L,const Mask& fg,double lam) {
    if (lam<=0) return 0;
    auto A=areas(L); if (A.empty()) return 0;
    double avg=static_cast<double>(area(fg))/A.size(), s=0;
    for (auto& kv:A) { double a=kv.second; if (a>2*avg) s+=a/avg-1; else if (a<0.5*avg) s+=avg/std::max(1.0,a)-1; }
    return lam*s;
}

double split_penalty3(const Labels& L,double p) {
    if (p<=0) return 0;
    double s=0;
    for (int id:ids(L)) { int c=components6(mask_of(L,id)); if (c>1) s+=(c-1)*p; }
    return s;
}

int extra_components3(const Labels& L) {
    int extra = 0;
    for (int id : ids(L)) {
        int c = components6(mask_of(L, id));
        if (c > 1) extra += c - 1;
    }
    return extra;
}

bool same_ids3(const Labels& A, const Labels& B) {
    return ids(A) == ids(B);
}

bool connected_pair3(const Labels& A, const Labels& B, int target_pieces) {
    auto ia = ids(A), ib = ids(B);
    if (ia.empty() || ia != ib) return false;
    if (target_pieces > 0 && static_cast<int>(ia.size()) != target_pieces) return false;
    for (int id : ia) {
        if (components6(mask_of(A, id)) != 1) return false;
        if (components6(mask_of(B, id)) != 1) return false;
    }
    return true;
}

double label_count_penalty3(const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,int target_pieces) {
    auto ia = ids(A), ib = ids(B);
    int err = std::abs(static_cast<int>(ia.size()) - static_cast<int>(ib.size()));
    if (target_pieces > 0) {
        err += std::abs(static_cast<int>(ia.size()) - target_pieces);
        err += std::abs(static_cast<int>(ib.size()) - target_pieces);
    }
    if (err == 0) return 0.0;
    const double scale = static_cast<double>(std::max<long long>(1, area(Af) + area(Bf)));
    return scale * 1000.0 * static_cast<double>(err);
}

double objective_total(const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,int level,const Args3& args,double* mismatch=nullptr) {
    auto m=match_labels(A,B);
    if (mismatch) *mismatch=m.total;
    double lam=level_lambda3(level,args.lambda0,args.decay);
    return m.total + area_penalty3(A,Af,lam)+area_penalty3(B,Bf,lam)+split_penalty3(A,args.split_penalty)+split_penalty3(B,args.split_penalty);
}

double objective_rank3(const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,int level,const Args3& args,int target_pieces,double* mismatch=nullptr) {
    // As in the paper, disconnected/split pieces are penalized by
    // split_penalty3, not forbidden during the beam search. Zhou & Wang note
    // that split pieces can be temporarily unavoidable in early levels.
    // args.require_connected is therefore a final/diagnostic constraint only.
    return objective_total(A, B, Af, Bf, level, args, mismatch) + label_count_penalty3(A, B, Af, Bf, target_pieces);
}

double candidate_score3(const Refine& r,const Mask& Af,const Mask& Bf,int level,const Args3& args,int target_pieces) {
    // Do not hard-filter disconnected intermediate pieces. The objective's
    // split penalty ranks them down, but the beam may need to pass through
    // them before global refinement reconnects the labels.
    const double count_penalty = label_count_penalty3(r.A, r.B, Af, Bf, target_pieces);
    const bool use_full = (args.lambda0 > 0.0) || (args.split_penalty > 0.0);
    if (!use_full) return r.em + count_penalty;
    return objective_total(r.A, r.B, Af, Bf, level, args) + count_penalty;
}

Refine refine(Labels A,Labels B,const Mask& Af,const Mask& Bf,int level,const Args3& args,uint64_t seed,int target_pieces) {
    (void)level; (void)target_pieces;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    Labels ca=A, cb=B, ba=A, bb=B;
    // Match the 2D reference: refinement keeps the best EM/mismatch state.
    // Area and split penalties are used to rank finished level candidates.
    double best=match_labels(ca,cb).total;
    for (int it=0; it<args.R; ++it) {
        auto mab=match_labels(ca,cb); if (mab.M.empty()) break;
        Labels nb=forward_copy(ca,cb,mab,Bf,rng);
        nb=fill_holes_random_flood(nb,Bf,rng);
        nb=random_switch(nb,Bf,overlap_mask(ca,nb,Bf),rng);
        auto mba=match_labels(nb,ca); if (mba.M.empty()) break;
        Labels na=forward_copy(nb,ca,mba,Af,rng);
        na=fill_holes_random_flood(na,Af,rng);
        na=random_switch(na,Af,overlap_mask(nb,na,Af),rng);
        ca=std::move(na); cb=std::move(nb);
        double e=match_labels(ca,cb).total;
        if (e+1e-12<best) { best=e; ba=ca; bb=cb; }
    }
    return {ba,bb,best};
}

Labels two_seed_grow(const Mask& fg,Vec s1,Vec s2) {
    Labels L(fg.d,fg.h,fg.w,0);
    auto in=[&](Vec p){ return 0<=p.z&&p.z<fg.d&&0<=p.y&&p.y<fg.h&&0<=p.x&&p.x<fg.w; };
    if (!in(s1)||!in(s2)||!fg.at(s1.z,s1.y,s1.x)||!fg.at(s2.z,s2.y,s2.x)) return L;
    const int dz[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dx[6]={0,0,0,0,1,-1};
    std::deque<Vec> q1,q2; std::vector<uint8_t> seen(static_cast<size_t>(fg.d)*fg.h*fg.w,0);
    L.at(s1.z,s1.y,s1.x)=1; L.at(s2.z,s2.y,s2.x)=2; seen[flat(s1.z,s1.y,s1.x,fg.h,fg.w)]=1; seen[flat(s2.z,s2.y,s2.x,fg.h,fg.w)]=1; q1.push_back(s1); q2.push_back(s2);
    while(!q1.empty()||!q2.empty()) {
        auto step=[&](std::deque<Vec>& q,int lab){ if(q.empty())return; Vec p=q.front(); q.pop_front(); for(int k=0;k<6;++k){ Vec n{p.z+dz[k],p.y+dy[k],p.x+dx[k]}; if(in(n)&&fg.at(n.z,n.y,n.x)&&!seen[flat(n.z,n.y,n.x,fg.h,fg.w)]){seen[flat(n.z,n.y,n.x,fg.h,fg.w)]=1; L.at(n.z,n.y,n.x)=lab; q.push_back(n);}}};
        step(q1,1); step(q2,2);
    }
    int a=0,b=0; for(int z=0;z<fg.d;++z)for(int y=0;y<fg.h;++y)for(int x=0;x<fg.w;++x)if(fg.at(z,y,x)){if(L.at(z,y,x)==1)++a;else if(L.at(z,y,x)==2)++b;}
    int f=a>=b?1:2; for(int z=0;z<fg.d;++z)for(int y=0;y<fg.h;++y)for(int x=0;x<fg.w;++x)if(fg.at(z,y,x)&&L.at(z,y,x)==0)L.at(z,y,x)=f;
    return L;
}

Refine level1_trial(const Mask& Af,const Mask& Bf,const Args3& args,uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    auto av=voxels(Af), bv=voxels(Bf);
    if(av.size()<2||bv.size()<2) return {};
    std::uniform_int_distribution<int> da(0,static_cast<int>(av.size())-1), db(0,static_cast<int>(bv.size())-1);
    Vec a1=av[da(rng)],a2=av[da(rng)],b1=bv[db(rng)],b2=bv[db(rng)];
    while(a2==a1)a2=av[da(rng)]; while(b2==b1)b2=bv[db(rng)];
    return refine(two_seed_grow(Af,a1,a2),two_seed_grow(Bf,b1,b2),Af,Bf,1,args,seed+13,2);
}

std::vector<Refine> level1(const Mask& Af,const Mask& Bf,const Args3& args,ThreadPool* pool=nullptr) {
    std::vector<Refine> out;
    if (pool) {
        std::vector<std::future<Refine>> futs;
        for(int i=0;i<args.N;++i){uint64_t s=args.seed+1009ull*i+17ull; futs.push_back(pool->enqueue([&Af,&Bf,&args,s]{return level1_trial(Af,Bf,args,s);}));}
        for(auto& f:futs){auto r=f.get(); if(!r.A.empty()) out.push_back(std::move(r));}
    } else {
        for(int i=0;i<args.N;++i){auto r=level1_trial(Af,Bf,args,args.seed+1009ull*i+17ull); if(!r.A.empty()) out.push_back(std::move(r));}
    }
    std::sort(out.begin(),out.end(),[&](const Refine& a,const Refine& b){return candidate_score3(a,Af,Bf,1,args,2)<candidate_score3(b,Af,Bf,1,args,2);});
    if((int)out.size()>args.Nb) out.resize(args.Nb);
    return out;
}

std::pair<int,int> worst_pair(const Labels& A,const Labels& B) {
    auto m=match_labels(A,B); double w=-1; std::pair<int,int> out{0,0};
    for(auto& kv:m.M){auto it=m.T.find({kv.first,kv.second}); if(it!=m.T.end()&&it->second.mismatch>w){w=it->second.mismatch; out=kv;}}
    return out;
}

std::pair<Labels,Labels> best_split(const Mask& A,const Mask& B,const Args3& args,uint64_t seed) {
    Args3 sub=args; sub.N=std::max(1,args.parent_N); sub.R=std::max(1,args.parent_R); sub.Nb=1; sub.seed=seed+811;
    auto r=level1(A,B,sub,nullptr); if(r.empty())return {}; return {r.front().A,r.front().B};
}

std::pair<Labels,Labels> local_split(const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,const Args3& args,uint64_t seed) {
    auto [a,b]=worst_pair(A,B); if(a<=0||b<=0)return{A,B};
    Mask ma=mask_of(A,a), mb=mask_of(B,b); if(area(ma)<2||area(mb)<2)return{A,B};
    auto [sa,sb]=best_split(ma,mb,args,seed);
    if(sa.empty()||sb.empty()){
        std::mt19937 rng(static_cast<uint32_t>(seed));
        auto va=voxels(ma), vb=voxels(mb);
        if(va.size()<2||vb.size()<2)return{A,B};
        std::uniform_int_distribution<int> da(0,static_cast<int>(va.size())-1), db(0,static_cast<int>(vb.size())-1);
        Vec a1=va[da(rng)], a2=va[da(rng)], b1=vb[db(rng)], b2=vb[db(rng)];
        while(a2==a1)a2=va[da(rng)];
        while(b2==b1)b2=vb[db(rng)];
        sa=two_seed_grow(ma,a1,a2);
        sb=two_seed_grow(mb,b1,b2);
    }
    int mx=0; for(int id:ids(A))mx=std::max(mx,id); for(int id:ids(B))mx=std::max(mx,id); int nl=mx+1;
    Labels A2=A,B2=B;
    for(int z=0;z<A.d;++z)for(int y=0;y<A.h;++y)for(int x=0;x<A.w;++x)if(ma.at(z,y,x)&&sa.at(z,y,x)==2)A2.at(z,y,x)=nl;
    for(int z=0;z<B.d;++z)for(int y=0;y<B.h;++y)for(int x=0;x<B.w;++x)if(mb.at(z,y,x)&&sb.at(z,y,x)==2)B2.at(z,y,x)=nl;
    return{A2,B2};
}

std::pair<Labels,Labels> mismatch_cluster(const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,std::mt19937& rng) {
    Mask mb=mismatch_mask(A,B,Bf), ma=mismatch_mask(B,A,Af); long long ca=area(ma), cb=area(mb); if(ca==0&&cb==0)return{A,B};
    bool chooseA = ca>0 && (cb==0 || std::uniform_real_distribution<double>(0,1)(rng)<(double)ca/(ca+cb));
    int mx=0; for(int id:ids(A))mx=std::max(mx,id); for(int id:ids(B))mx=std::max(mx,id); int nl=mx+1;
    Labels A2=A,B2=B; Mask& M=chooseA?ma:mb; Labels& L=chooseA?A2:B2;
    auto pts=voxels(M); if(pts.empty())return{A,B}; Vec s=pts[std::uniform_int_distribution<int>(0,(int)pts.size()-1)(rng)];
    const int dz[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dx[6]={0,0,0,0,1,-1}; std::deque<Vec> q; L.at(s.z,s.y,s.x)=nl; q.push_back(s);
    while(!q.empty()){Vec p=q.front();q.pop_front();for(int k=0;k<6;++k){Vec n{p.z+dz[k],p.y+dy[k],p.x+dx[k]};if(0<=n.z&&n.z<M.d&&0<=n.y&&n.y<M.h&&0<=n.x&&n.x<M.w&&M.at(n.z,n.y,n.x)&&L.at(n.z,n.y,n.x)!=nl){L.at(n.z,n.y,n.x)=nl;q.push_back(n);}}}
    auto ia=ids(A2), ib=ids(B2);
    auto peel=[&](Labels& X){auto ar=areas(X);int big=ar.begin()->first,sz=-1;for(auto&kv:ar)if(kv.second>sz){big=kv.first;sz=kv.second;}for(int z=0;z<X.d;++z)for(int y=0;y<X.h;++y)for(int x=0;x<X.w;++x)if(X.at(z,y,x)==big){X.at(z,y,x)=nl;return;}};
    if(ia.size()>ib.size())peel(B2); else if(ib.size()>ia.size())peel(A2);
    return{A2,B2};
}

Refine trialK(const Refine& c,const Mask& Af,const Mask& Bf,int level,const Args3& args,uint64_t seed,int target_pieces) {
    Labels A=c.A,B=c.B; std::mt19937 rng((uint32_t)seed);
    if(std::uniform_real_distribution<double>(0,1)(rng)<0.75){auto p=local_split(A,B,Af,Bf,args,seed+37);A=std::move(p.first);B=std::move(p.second);}
    else {auto p=mismatch_cluster(A,B,Af,Bf,rng);A=std::move(p.first);B=std::move(p.second);}
    return refine(A,B,Af,Bf,level,args,seed+73,target_pieces);
}

void save_beam(const Args3& args,const std::string& name,const std::vector<Refine>& beam);

std::vector<Refine> explore(const std::vector<Refine>& prev,const Mask& Af,const Mask& Bf,int level,const Args3& args,ThreadPool& pool,uint64_t seed,int target_pieces) {
    std::mt19937 prng((uint32_t)seed); std::uniform_int_distribution<int> pick(0,(int)prev.size()-1); std::vector<std::future<Refine>> futs;
    for(int i=0;i<args.N;++i){const Refine* c=&prev[pick(prng)];uint64_t s=seed+2003ull*i+23ull;futs.push_back(pool.enqueue([=,&Af,&Bf,&args]{return trialK(*c,Af,Bf,level,args,s,target_pieces);}));}
    std::vector<Refine> out;
    std::filesystem::path progress_path = std::filesystem::path(args.outdir) / ("progress_level_" + std::to_string(target_pieces) + "pcs.tsv");
    if (args.save_beam) {
        std::ofstream progress(progress_path.string());
        progress << "completed\tbest_score\tbest_mismatch\tbest_pieces\tconnected\n";
    }
    for(size_t i=0;i<futs.size();++i){
        out.push_back(futs[i].get());
        std::sort(out.begin(),out.end(),[&](const Refine&a,const Refine&b){return candidate_score3(a,Af,Bf,level,args,target_pieces)<candidate_score3(b,Af,Bf,level,args,target_pieces);});
        if((int)out.size()>args.Nb)out.resize(args.Nb);
        int completed = (int)i + 1;
        if(args.save_beam && args.checkpoint_every > 0 && (completed % args.checkpoint_every == 0 || completed == args.N)){
            std::string partial = "partial_level_" + std::to_string(target_pieces) + "pcs";
            save_beam(args, partial, out);
            double best_mismatch = out.empty() ? 0.0 : out.front().em;
            double best_score = out.empty() ? 0.0 : candidate_score3(out.front(),Af,Bf,level,args,target_pieces);
            int best_pieces = out.empty() ? 0 : (int)ids(out.front().A).size();
            bool conn = !out.empty() && connected_pair3(out.front().A,out.front().B,target_pieces);
            std::ofstream progress(progress_path.string(), std::ios::app);
            progress << completed << "\t" << best_score << "\t" << best_mismatch << "\t" << best_pieces << "\t" << (conn?1:0) << "\n";
            std::cout << "Level " << level << " progress " << completed << "/" << args.N
                      << ": best Score=" << best_score
                      << " M=" << best_mismatch
                      << " pieces=" << best_pieces
                      << " connected=" << (conn?1:0)
                      << " checkpoint=beam_" << partial << "\n";
        }
    }
    std::sort(out.begin(),out.end(),[&](const Refine&a,const Refine&b){return candidate_score3(a,Af,Bf,level,args,target_pieces)<candidate_score3(b,Af,Bf,level,args,target_pieces);});
    if((int)out.size()>args.Nb)out.resize(args.Nb); return out;
}

void save_pair(const Args3& args,const std::string& name,const Labels& A,const Labels& B){save_stack(args.outdir+"/"+name+"_A",A);save_stack(args.outdir+"/"+name+"_B",B);}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void save_beam(const Args3& args,const std::string& name,const std::vector<Refine>& beam) {
    std::filesystem::path root = std::filesystem::path(args.outdir) / ("beam_" + name);
    std::filesystem::create_directories(root);
    std::ofstream meta((root / "beam.tsv").string());
    meta << "rank\tmismatch\tpieces\n";
    for (size_t i = 0; i < beam.size(); ++i) {
        char stem[64];
        std::snprintf(stem, sizeof(stem), "cand_%03zu", i);
        save_stack((root / (std::string(stem) + "_A")).string(), beam[i].A);
        save_stack((root / (std::string(stem) + "_B")).string(), beam[i].B);
        meta << i << "\t" << beam[i].em << "\t" << ids(beam[i].A).size() << "\n";
    }
}

std::vector<Refine> load_beam(const std::string& dir) {
    std::filesystem::path root(dir);
    if (!std::filesystem::exists(root)) throw std::runtime_error("resume beam directory does not exist: " + dir);
    std::vector<std::string> stems;
    for (auto& e : std::filesystem::directory_iterator(root)) {
        if (!e.is_directory()) continue;
        std::string name = e.path().filename().string();
        if (ends_with(name, "_A")) stems.push_back(name.substr(0, name.size() - 2));
    }
    std::sort(stems.begin(), stems.end());
    std::vector<Refine> out;
    for (const auto& stem : stems) {
        std::filesystem::path ad = root / (stem + "_A");
        std::filesystem::path bd = root / (stem + "_B");
        if (!std::filesystem::exists(bd)) {
            throw std::runtime_error("resume beam candidate missing B stack for " + stem);
        }
        Labels A = load_label_stack(ad.string());
        Labels B = load_label_stack(bd.string());
        double em = match_labels(A, B).total;
        out.push_back({std::move(A), std::move(B), em});
    }
    if (out.empty()) throw std::runtime_error("resume beam contains no cand_*_A/B stacks: " + dir);
    return out;
}

bool labels_match_support(const Labels& L, const Mask& M) {
    if (L.d != M.d || L.h != M.h || L.w != M.w) return false;
    for (int z=0; z<M.d; ++z) for (int y=0; y<M.h; ++y) for (int x=0; x<M.w; ++x)
        if ((M.at(z,y,x)!=0) != (L.at(z,y,x)>0)) return false;
    return true;
}

void print_summary3(const std::string& tag,const Labels& A,const Labels& B,const Mask& Af,const Mask& Bf,int level,const Args3& args,int target_pieces) {
    double mismatch = 0.0;
    double obj = objective_total(A, B, Af, Bf, level, args, &mismatch);
    double score = candidate_score3({A, B, mismatch}, Af, Bf, level, args, target_pieces);
    auto ia = ids(A), ib = ids(B);
    std::cout << tag
              << ": Obj=" << obj
              << " Score=" << score
              << " M=" << mismatch
              << " labelsA=" << ia.size()
              << " labelsB=" << ib.size()
              << " target=" << target_pieces
              << " extraA=" << extra_components3(A)
              << " extraB=" << extra_components3(B)
              << " connected=" << (connected_pair3(A, B, target_pieces) ? 1 : 0)
              << "\n";
}

bool self_test3() {
    auto fail = [](const std::string& msg)->bool {
        std::cerr << "self-test failed: " << msg << "\n";
        return false;
    };
    if (rots().size() != 24) return fail("expected 24 proper 3D rotations");

    Mask src(9, 10, 11, 0);
    std::vector<Vec> shape = {{1,1,1},{1,1,2},{1,2,1},{2,1,1},{3,1,2},{3,2,2}};
    for (auto p : shape) src.at(p.z,p.y,p.x) = 1;
    auto src_coords = voxels(src);
    auto [z0,y0,x0,z1,y1,x1] = bbox(src);
    std::array<int,3> dims = {z1-z0+1, y1-y0+1, x1-x0+1};

    for (int ri=0; ri<static_cast<int>(rots().size()); ++ri) {
        Mask dst(9, 10, 11, 0);
        for (auto p : shape) {
            auto q = rotate_local({p.z-z0,p.y-y0,p.x-x0}, dims, rots()[ri]);
            dst.at(q[0]+3,q[1]+3,q[2]+3) = 1;
        }
        auto dst_coords = voxels(dst);
        PairCost pc = pair_mismatch(src_coords, dst_coords, dst);
        if (pc.mismatch != 0.0) return fail("pair_mismatch did not recover exact rotated piece");
        Mask placed = apply_rotation(src, pc.params, dst);
        if (area(placed) != area(dst)) return fail("apply_rotation did not reproduce exact matched piece");
    }

    Labels A(5,5,5,0), B(5,5,5,0);
    A.at(1,1,1)=1; A.at(1,1,2)=1; A.at(3,3,3)=2; A.at(3,3,2)=2;
    B.at(1,1,1)=10; B.at(1,1,2)=10; B.at(3,3,3)=20; B.at(3,3,2)=20;
    Match m = match_labels(A, B);
    if (m.M.size() != 2 || m.total != 0.0) return fail("match_labels did not recover two exact pieces");

    Mask fg(3,3,3,1);
    Labels grown = two_seed_grow(fg, {0,0,0}, {2,2,2});
    if (ids(grown).size() != 2) return fail("two_seed_grow did not create two labels");
    for (int z=0; z<fg.d; ++z) for (int y=0; y<fg.h; ++y) for (int x=0; x<fg.w; ++x)
        if (fg.at(z,y,x) && grown.at(z,y,x)==0) return fail("two_seed_grow left foreground unlabeled");

    Labels split(3,3,3,0);
    split.at(0,0,0)=1; split.at(2,2,2)=1;
    if (extra_components3(split) != 1) return fail("extra_components3 did not detect split label");

    std::cout << "zhou3d self-test: passed\n";
    return true;
}

int main3d(int argc,char**argv){
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    if(argc==2 && std::string(argv[1])=="--self-test") return self_test3()?0:1;
    Args3 args; if(argc<3){std::cerr<<"Usage: "<<argv[0]<<" <A_slice_dir> <B_slice_dir> [options]\n";return 1;} args.a_dir=argv[1]; args.b_dir=argv[2];
    for(int i=3;i<argc;++i){std::string a=argv[i]; if(a=="-o"&&i+1<argc)args.outdir=argv[++i]; else if(a=="-R"&&i+1<argc)args.R=std::stoi(argv[++i]); else if(a=="-N"&&i+1<argc)args.N=std::stoi(argv[++i]); else if(a=="-Nb"&&i+1<argc)args.Nb=std::stoi(argv[++i]); else if(a=="--max-level"&&i+1<argc)args.max_level=std::stoi(argv[++i]); else if(a=="--parent-split-N"&&i+1<argc)args.parent_N=std::stoi(argv[++i]); else if(a=="--parent-split-R"&&i+1<argc)args.parent_R=std::stoi(argv[++i]); else if(a=="--lambda0"&&i+1<argc)args.lambda0=std::stod(argv[++i]); else if(a=="--decay"&&i+1<argc)args.decay=std::stod(argv[++i]); else if(a=="--split-penalty"&&i+1<argc)args.split_penalty=std::stod(argv[++i]); else if(a=="--seed"&&i+1<argc)args.seed=std::stoull(argv[++i]); else if(a=="-j"&&i+1<argc)args.jobs=std::stoi(argv[++i]); else if(a=="--checkpoint-every"&&i+1<argc)args.checkpoint_every=std::stoi(argv[++i]); else if(a=="--resume-labels-a"&&i+1<argc)args.resume_a_dir=argv[++i]; else if(a=="--resume-labels-b"&&i+1<argc)args.resume_b_dir=argv[++i]; else if(a=="--resume-beam"&&i+1<argc)args.resume_beam_dir=argv[++i]; else if(a=="--save-beam")args.save_beam=true; else if(a=="--best-any-level")args.fixed_final=false; else if(a=="--fixed-final-level")args.fixed_final=true; else if(a=="--allow-disconnected")args.require_connected=false; else if(a=="--require-connected")args.require_connected=true; }
    std::cout<<"Starting 3D Zhou-style dissection\n  A: "<<args.a_dir<<"\n  B: "<<args.b_dir<<"\n  N "<<args.N<<" R "<<args.R<<" Nb "<<args.Nb<<" max-level "<<args.max_level<<" seed "<<args.seed<<" require-connected "<<(args.require_connected?1:0)<<"\n";
    Mask A=load_stack(args.a_dir), B=load_stack(args.b_dir); if(area(A)!=area(B)){std::cerr<<"Equal foreground voxel count required. A="<<area(A)<<" B="<<area(B)<<"\n";return 3;}
    int nt=args.jobs<=0?(int)std::max(1u,std::thread::hardware_concurrency()):args.jobs; ThreadPool pool(nt); std::filesystem::create_directories(args.outdir); std::cout<<"Using "<<nt<<" threads; rotations="<<rots().size()<<"; A volume="<<A.d<<"x"<<A.h<<"x"<<A.w<<" B volume="<<B.d<<"x"<<B.h<<"x"<<B.w<<" fg="<<area(A)<<"\n";
    std::vector<Refine> cands;
    Refine best_over;
    int start_level = 2;
    const bool resume_labels = !args.resume_a_dir.empty() || !args.resume_b_dir.empty();
    const bool resume_beam = !args.resume_beam_dir.empty();
    if (resume_labels && resume_beam) {
        std::cerr << "--resume-beam cannot be combined with --resume-labels-a/b.\n";
        return 8;
    }
    if (resume_beam) {
        cands = load_beam(args.resume_beam_dir);
        for (const auto& c : cands) {
            if (!labels_match_support(c.A, A) || !labels_match_support(c.B, B)) {
                std::cerr << "Resume beam candidate support does not match input foreground stacks.\n";
                return 9;
            }
        }
        int pcs0 = static_cast<int>(ids(cands.front().A).size());
        for (const auto& c : cands) {
            if ((int)ids(c.A).size() != pcs0 || (int)ids(c.B).size() != pcs0) {
                std::cerr << "Resume beam candidates do not all have the same piece count.\n";
                return 10;
            }
        }
        int resume_level = std::max(1, pcs0 - 1);
        std::sort(cands.begin(), cands.end(), [&](const Refine& a,const Refine& b){
            return candidate_score3(a,A,B,resume_level,args,pcs0)<candidate_score3(b,A,B,resume_level,args,pcs0);
        });
        best_over = cands.front();
        start_level = std::max(2, pcs0);
        std::cout << "Resuming from beam:\n  beam: " << args.resume_beam_dir
                  << "\n  candidates " << cands.size()
                  << "\n  pieces " << pcs0 << "\n";
        print_summary3("ResumeBeam L"+std::to_string(resume_level), best_over.A, best_over.B, A, B, resume_level, args, pcs0);
        save_pair(args,"resume_beam_best_"+std::to_string(pcs0)+"pcs",best_over.A,best_over.B);
        if (args.save_beam) save_beam(args,"resume_"+std::to_string(pcs0)+"pcs",cands);
    } else if (resume_labels) {
        if (args.resume_a_dir.empty() || args.resume_b_dir.empty()) {
            std::cerr << "Both --resume-labels-a and --resume-labels-b are required.\n";
            return 5;
        }
        Labels RA = load_label_stack(args.resume_a_dir);
        Labels RB = load_label_stack(args.resume_b_dir);
        if (!labels_match_support(RA, A) || !labels_match_support(RB, B)) {
            std::cerr << "Resume label support/dimensions do not match input foreground stacks.\n";
            return 7;
        }
        double em = match_labels(RA, RB).total;
        best_over = {RA, RB, em};
        cands.push_back(best_over);
        int pcs0 = static_cast<int>(ids(RA).size());
        start_level = std::max(2, pcs0);
        std::cout << "Resuming from labels:\n  A labels: " << args.resume_a_dir
                  << "\n  B labels: " << args.resume_b_dir
                  << "\n  pieces " << pcs0 << "\n";
        print_summary3("Resume L"+std::to_string(std::max(0, pcs0-1)), best_over.A, best_over.B, A, B, std::max(1, pcs0-1), args, pcs0);
        save_pair(args,"resume_"+std::to_string(pcs0)+"pcs",best_over.A,best_over.B);
        if (args.save_beam) save_beam(args,"resume_"+std::to_string(pcs0)+"pcs",cands);
    } else {
        cands=level1(A,B,args,&pool);
        if(cands.empty()){std::cerr<<"Level 1 failed: no 2-piece candidates\n";return 4;}
        best_over=cands.front();
        print_summary3("Best L1",best_over.A,best_over.B,A,B,1,args,2);
        save_pair(args,"level_2pcs",best_over.A,best_over.B);
        if (args.save_beam) save_beam(args,"level_2pcs",cands);
    }
    for(int level=start_level;level<=args.max_level;++level){
        int target_pieces=level+1;
        std::cout<<"\n=== Level "<<level<<" ("<<target_pieces<<" clusters) ===\n";
        cands=explore(cands,A,B,level,args,pool,args.seed+(uint64_t)level*10007ull,target_pieces);
        if(cands.empty()){std::cout<<"No candidates for target pieces "<<target_pieces<<"; stopping at previous valid level.\n"; break;}
        Refine cur=cands.front();
        int pcs=(int)ids(cur.A).size();
        print_summary3("Best L"+std::to_string(level),cur.A,cur.B,A,B,level,args,target_pieces);
        save_pair(args,"level_"+std::to_string(pcs)+"pcs",cur.A,cur.B);
        if (args.save_beam) save_beam(args,"level_"+std::to_string(target_pieces)+"pcs",cands);
        if(args.fixed_final)best_over=cur;
        else{
            int best_level=std::max(0,(int)ids(best_over.A).size()-1);
            double bo=candidate_score3(best_over,A,B,best_level,args,(int)ids(best_over.A).size());
            double co=candidate_score3(cur,A,B,level,args,target_pieces);
            if(co<bo)best_over=cur;
        }
    }
    int pcs=(int)ids(best_over.A).size();
    double fm=0,fo=objective_total(best_over.A,best_over.B,A,B,std::max(0,pcs-1),args,&fm);
    double fs=split_penalty3(best_over.A,args.split_penalty)+split_penalty3(best_over.B,args.split_penalty);
    bool final_connected=connected_pair3(best_over.A,best_over.B,pcs);
    bool final_ok=(fm<CONVERGENCE_EPS && fs<1.0 && same_ids3(best_over.A,best_over.B) && (!args.require_connected || final_connected));
    std::cout<<"\n=== Final ===\nPieces: "<<pcs<<"\nObjective: "<<fo<<" Mismatch: "<<fm
             <<" labelsB="<<ids(best_over.B).size()
             <<" split="<<fs
             <<" extraA="<<extra_components3(best_over.A)
             <<" extraB="<<extra_components3(best_over.B)
             <<" connected="<<(final_connected?1:0)<<"\n";
    save_pair(args,"result_"+std::to_string(pcs)+"pcs",best_over.A,best_over.B);
    return final_ok?0:1;
}

} // namespace zhou3d

int main(int argc, char** argv) {
    return zhou3d::main3d(argc, argv);
}

