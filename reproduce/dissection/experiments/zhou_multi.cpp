/*
 * Zhou & Wang (2012): multi-figure chain generalization
 *
 * Extends our two-figure implementation of the paper to N >= 2 figures, following the paper's note that the
 * bidirectional refinement generalizes to multiple figures:
 *
 * - State is a vector of label grids L[0..T-1], one per figure, all with
 *   equal foreground area.
 * - Refinement sweeps copy-paste forward along the chain (L0->L1->...->LT-1)
 *   and then backward (LT-1->...->L0); each hop is the original paper step
 *   (match, copy-paste, random flood-fill of holes, random switching over
 *   overwritten pixels). For T=2 this reduces exactly to the original
 *   A->B, B->A iteration.
 * - EM is the sum of matched mismatch over consecutive pairs (for T=2 it is
 *   the original EM).
 * - Level >= 2 heuristics: H1 splits the worst-matched cluster pair across
 *   any consecutive pair of figures (Level-1 on the parent masks) and peels
 *   one pixel in every other figure to keep cluster counts equal; H2 creates
 *   a new cluster from mismatch pixels of a randomly chosen chain direction
 *   and peels the rest.
 * - Objective: E = EM + lambda * sum_t Ea(L[t]) + sum_t split-penalty.
 *
 * Build (MSVC):  cl /std:c++17 /O2 /EHsc zhou_multi.cpp
 * Usage:         zhou_multi A.png B.png C.png [-o outdir -N 400 -R 100 ...]
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <deque>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <tuple>
#include <stdexcept>
#include <functional>
#include <climits>
#include <cstdint>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

using Mask = Matrix<uint8_t>;
using Labels = Matrix<int>;
using Point = std::pair<int, int>;
using Coords = std::vector<Point>;
using BBox = std::tuple<int, int, int, int>;

struct MismatchParams {
    int rot = 0;
    bool flip = false;
    Point shift = {0, 0};
};
struct MismatchResult {
    double mismatch = std::numeric_limits<double>::infinity();
    MismatchParams params;
};

using MatchMap = std::map<int, int>;
using TransformMap = std::map<std::pair<int, int>, MismatchResult>;
struct MatchResult {
    MatchMap M;
    double totalMismatch = 0.0;
    TransformMap T;
};

// Multi-figure state: one label grid per figure.
struct MultiResult {
    std::vector<Labels> L;
    double em = std::numeric_limits<double>::infinity();
    bool valid() const { return !L.empty() && L.front().rows() > 0; }
};

struct ObjectiveResult {
    double total, mismatch, area, split;
};

struct Edge {
    int to;
    size_t rev;
    int cap;
    double cost;
};

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

const int R_DEFAULT = 100;
const int N_DEFAULT = 400;
const int NB_DEFAULT = 30;
const int MAX_LEVEL_DEFAULT = 15;
const double LAMBDA0_DEFAULT = 1.0;
const double DECAY_DEFAULT = 0.8;
const double SPLIT_PENALTY_DEFAULT = 1e6;
const uint64_t SEED_DEFAULT = 2012;
const double CONVERGENCE_EPS = 1e-9;

struct Args {
    std::vector<std::string> shape_paths;
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
    // EM mode: false = chain sum over consecutive pairs (original),
    // true = all-pairs sum. Chain EM never compares L0 to L2 directly, so
    // for T>=3 endpoints can drift while EM looks small; all-pairs closes
    // that hole at the cost of T(T-1)/2 instead of T-1 matchings.
    bool em_allpairs = false;
    // Sweep topology: false = open chain (original), true = cycle. The open
    // chain never copy-pastes between the endpoint figures, so that pair
    // accumulates the most mismatch (observed: bear<->fish 652 vs ~480 for
    // the adjacent pairs). A cyclic sweep visits (t, (t+1)%T), making every
    // pair adjacent for T=3. Ignored for T=2 (reduces to the original).
    bool sweep_cycle = false;
};


// ----------------- I/O -----------------
Mask load_binary(const std::string& path, int thr = 128) {
    int w = 0, h = 0, comp = 0;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &comp, 1);
    if (!img) {
        throw std::runtime_error("stbi_load failed for: " + path);
    }
    Mask out(h, w, uint8_t(0));
    for (int y = 0; y < h; ++y) {
        const unsigned char* row = img + y * w;
        for (int x = 0; x < w; ++x) {
            out.at(y, x) = (row[x] < thr) ? uint8_t(1) : uint8_t(0);
        }
    }
    stbi_image_free(img);
    return out;
}

static inline std::tuple<uint8_t,uint8_t,uint8_t> _label_color(int id) {
    if (id <= 0) return {0,0,0};
    uint32_t v = static_cast<uint32_t>(id) * 0x9E3779B1u;
    v ^= v >> 16; v *= 0x85EBCA6Bu; v ^= v >> 13; v *= 0xC2B2AE35u; v ^= v >> 16;
    uint8_t r = static_cast<uint8_t>( (v      ) & 0xFF );
    uint8_t g = static_cast<uint8_t>( (v >>  8) & 0xFF );
    uint8_t b = static_cast<uint8_t>( (v >> 16) & 0xFF );
    if (r < 24 && g < 24 && b < 24) { r ^= 0x55; g ^= 0xAA; b ^= 0xCC; }
    return {r,g,b};
}

void save_labels_png(const std::string& path, const Labels& labels) {
    const int H = labels.rows();
    const int W = labels.cols();
    if (H <= 0 || W <= 0) {
        throw std::runtime_error("save_labels_png: empty labels.");
    }
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
    if (ymax < 0) {
        return {0, 0, -1, -1};
    }
    return {ymin, xmin, ymax, xmax};
}

int components4(const Mask& mask) {
    const int H = mask.rows(), W = mask.cols();
    if (H == 0 || W == 0) return 0;

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
struct RotflipResult {
    int h = 0, w = 0;
    Coords coords;
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
    return { static_cast<int>(std::lround(cy)), static_cast<int>(std::lround(cx)) };
}

RotflipResult rotflip(const Coords& coords, int h, int w, int k, bool flip) {
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
            case 0: y2 = y;          x2 = x;          break;
            case 1: y2 = x;          x2 = (h - 1) - y;break;
            case 2: y2 = (h - 1)- y; x2 = (w - 1) - x;break;
            default:y2 = (w - 1)- x; x2 = y;          break;
        }
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
    out.reserve(static_cast<size_t>(H) * 4);
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

    int y0 = INT_MAX, x0 = INT_MAX, y1 = -1, x1 = -1;
    for (auto &p : Ayx) {
        int y = p.first, x = p.second;
        if (y < y0) y0 = y; if (x < x0) x0 = x;
        if (y > y1) y1 = y; if (x > x1) x1 = x;
    }
    const int h = y1 - y0 + 1, w = x1 - x0 + 1;

    Coords A_local; A_local.reserve(Ayx.size());
    for (auto &p : Ayx) A_local.emplace_back(p.first - y0, p.second - x0);

    auto [cyB, cxB] = _centroid_f(Byx);
    double lb = std::abs(Sa - Sb);

    double best_mis = std::numeric_limits<double>::infinity();
    MismatchParams best_par{0,false,{0,0}};

    for (int flip = 0; flip < 2; ++flip) {
        for (int rotk = 0; rotk < 4; ++rotk) {
            auto R = rotflip(A_local, h, w, rotk, flip != 0);
            if (R.coords.empty()) continue;

            auto [cyA_l, cxA_l] = _centroid_f(R.coords);
            if (cyA_l < 0) continue;
            double cyA = cyA_l + y0;
            double cxA = cxA_l + x0;

            auto snaps = four_snaps(cyB - cyA, cxB - cxA);
            for (auto s : snaps) {
                int dy = s.first, dx = s.second;
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
    bool any = false;
    for (auto v : mask.m_data) { if (v) { any = true; break; } }
    if (!any) return out;

    auto [y0, x0, y1, x1] = bbox(mask);
    if (y1 < y0 || x1 < x0) return out;

    Coords local;
    local.reserve(static_cast<size_t>((y1 - y0 + 1) * (x1 - x0 + 1)));
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (mask.at(y, x)) local.emplace_back(y - y0, x - x0);

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

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (!shapeB.at(y, x)) out.at(y, x) = 0;

    return out;
}

// ----------------- Random flood fill for holes (as in the paper) -----------------
Labels fill_holes_random_flood(const Labels& L, const Mask& fg, std::mt19937& rng) {
    const int H = L.rows(), W = L.cols();
    Labels out = L;

    auto hole_at = [&](int y, int x){ return fg.at(y,x) && (out.at(y,x) == 0); };

    bool any_hole = false;
    for (int y = 0; y < H && !any_hole; ++y)
        for (int x = 0; x < W && !any_hole; ++x)
            if (hole_at(y,x)) any_hole = true;
    if (!any_hole) return out;

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
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (fg.at(y,x) && out.at(y,x) == 0) out.at(y,x) = 1;
        return out;
    }

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

    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(H*W/16));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (conflict.at(y,x)) pts.emplace_back(y, x);

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

// ----------------- Objective (Eq. 3/4, summed over figures) -----------------
double level_lambda(int level, double lambda0, double decay) {
    int k = std::max(0, level - 1);
    return (lambda0 / 2.0) * std::pow(decay, static_cast<double>(k));
}

double area_penalty(const Labels& L, const Mask& fg, double lam) {
    if (lam <= 0.0) return 0.0;
    auto A = areas(L);
    if (A.empty()) return 0.0;

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
        int c = components4(m);
        if (c > 1) s += (c - 1) * penalty;
    }
    return s;
}

// EM: matched mismatch summed over consecutive pairs (chain) or all pairs.
// The refinement sweeps and the H1/H2 heuristics stay chain-based either
// way; only the objective/scoring changes.
double em_total(const std::vector<Labels>& L, const std::vector<Mask>& fg, const Args& args) {
    double s = 0.0;
    if (args.em_allpairs) {
        for (size_t t = 0; t < L.size(); ++t)
            for (size_t u = t + 1; u < L.size(); ++u)
                s += match_labels(L[t], L[u], fg[u]).totalMismatch;
        return s;
    }
    for (size_t t = 0; t + 1 < L.size(); ++t)
        s += match_labels(L[t], L[t + 1], fg[t + 1]).totalMismatch;
    return s;
}

ObjectiveResult objective_multi(const std::vector<Labels>& L, const std::vector<Mask>& fg,
                                int level, const Args& args) {
    ObjectiveResult out{};
    const double lam = level_lambda(level, args.lambda0, args.decay);
    double area = 0.0, spl = 0.0;
    for (size_t t = 0; t < L.size(); ++t) {
        area += area_penalty(L[t], fg[t], lam);
        spl  += split_penalty(L[t], args.split_penalty);
    }
    out.mismatch = em_total(L, fg, args);
    out.area     = area;
    out.split    = spl;
    out.total    = out.mismatch + area + spl;
    return out;
}

std::set<int> cluster_id_set(const Labels& L) {
    std::set<int> out;
    for (int id : cluster_ids(L)) out.insert(id);
    return out;
}

int label_count_error(const std::vector<Labels>& L, int target_pieces) {
    if (target_pieces <= 0 || L.empty()) return 0;
    int err = 0;
    const std::set<int> ref = cluster_id_set(L.front());
    err += std::abs(static_cast<int>(ref.size()) - target_pieces);

    for (size_t t = 1; t < L.size(); ++t) {
        const std::set<int> cur = cluster_id_set(L[t]);
        err += std::abs(static_cast<int>(cur.size()) - target_pieces);
        for (int id : ref) if (!cur.count(id)) ++err;
        for (int id : cur) if (!ref.count(id)) ++err;
    }
    return err;
}

bool target_piece_count_all(const std::vector<Labels>& L, int target_pieces) {
    return label_count_error(L, target_pieces) == 0;
}

double total_foreground_area(const std::vector<Mask>& fg) {
    double s = 0.0;
    for (const auto& m : fg)
        for (auto v : m.m_data) s += (v ? 1.0 : 0.0);
    return s;
}

double candidate_score_multi(const std::vector<Labels>& L, const std::vector<Mask>& fg,
                             int level, int target_pieces, const Args& args) {
    auto obj = objective_multi(L, fg, level, args);
    const int count_err = label_count_error(L, target_pieces);
    if (count_err == 0) return obj.total;
    return obj.total + (1000.0 * total_foreground_area(fg) * static_cast<double>(count_err));
}

// ----------------- Refinement loop (chain sweep) -----------------
// One copy-paste hop src -> dst: the original paper step.
bool copy_step(std::vector<Labels>& L, int src, int dst,
               const std::vector<Mask>& fg, std::mt19937& rng) {
    MatchResult mr = match_labels(L[src], L[dst], fg[dst]);
    if (mr.M.empty()) return false;
    Labels D = forward_copy(L[src], L[dst], mr, fg[dst], rng);
    D = fill_holes_random_flood(D, fg[dst], rng);
    Mask conf = overlap_mask_src_to_dst(L[src], D, fg[dst]);
    D = random_switch(D, fg[dst], conf, rng);
    L[dst] = std::move(D);
    return true;
}

MultiResult refine_multi(std::vector<Labels> L, const std::vector<Mask>& fg,
                         int level, int target_pieces, const Args& args, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    const int T = static_cast<int>(L.size());

    std::vector<Labels> best = L;
    double best_score = candidate_score_multi(L, fg, level, target_pieces, args);
    double best_em = em_total(best, fg, args);

    for (int it = 0; it < args.R; ++it) {
        bool ok = true;
        if (args.sweep_cycle && T > 2) {
            // Cyclic sweep: forward L0 -> L1 -> ... -> L(T-1) -> L0, then
            // backward L0 -> L(T-1) -> ... -> L1 -> L0; every ordered pair
            // of adjacent-on-the-ring figures gets a copy-paste hop.
            for (int t = 0; t < T && ok; ++t) ok = copy_step(L, t, (t + 1) % T, fg, rng);
            for (int t = T - 1; t >= 0 && ok; --t) ok = copy_step(L, (t + 1) % T, t, fg, rng);
        } else {
            // Forward sweep: L0 -> L1 -> ... -> L(T-1)
            for (int t = 0; t + 1 < T && ok; ++t) ok = copy_step(L, t, t + 1, fg, rng);
            // Backward sweep: L(T-1) -> ... -> L0
            for (int t = T - 1; t - 1 >= 0 && ok; --t) ok = copy_step(L, t, t - 1, fg, rng);
        }
        if (!ok) break;

        double score = candidate_score_multi(L, fg, level, target_pieces, args);
        if (score + 1e-12 < best_score) {
            best_score = score;
            best = L;
            best_em = em_total(best, fg, args);
        }
    }

    return {best, best_em};
}

// ----------------- Level 1 -----------------
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

MultiResult _level1_trial(const std::vector<Mask>& fg, const Args& args, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    const int T = static_cast<int>(fg.size());

    std::vector<Labels> L(T);
    for (int t = 0; t < T; ++t) {
        auto pts = _fg_pixels(fg[t]);
        if (pts.size() < 2) return MultiResult{};
        std::uniform_int_distribution<int> d(0, static_cast<int>(pts.size()) - 1);
        Point s1 = pts[d(rng)], s2 = pts[d(rng)];
        while (s2 == s1) s2 = pts[d(rng)];
        L[t] = two_seed_grow(fg[t], s1, s2);
    }

    return refine_multi(std::move(L), fg, 1, 2, args, seed + 13);
}

std::vector<MultiResult> level1(const std::vector<Mask>& fg, const Args& args, ThreadPool& pool) {
    const int N = args.N;
    std::vector<uint64_t> seeds; seeds.reserve(N);
    for (int i = 0; i < N; ++i) seeds.push_back(args.seed + 1009ull*i + 17ull);

    std::vector<std::future<MultiResult>> futs;
    futs.reserve(N);

    for (auto s : seeds) {
        futs.emplace_back(pool.enqueue([&fg, &args, s](){
            return _level1_trial(fg, args, s);
        }));
    }

    std::vector<MultiResult> outs;
    outs.reserve(N);
    for (auto &f : futs) {
        MultiResult r = f.get();
        if (r.valid()) outs.push_back(std::move(r));
    }
    if (outs.empty()) return outs;

    auto score = [&](const MultiResult& r)->double {
        return candidate_score_multi(r.L, fg, 1, 2, args);
    };

    std::sort(outs.begin(), outs.end(),
        [&](const MultiResult& a, const MultiResult& b){ return score(a) < score(b); });

    if (static_cast<int>(outs.size()) > args.Nb) outs.resize(args.Nb);
    return outs;
}

// ----------------- Level >= 2 helpers -----------------
struct WorstPairResult {
    int t = -1, u = -1; // figure pair (t, u); u = t+1 (chain) or (t+1)%T (cycle)
    int a = 0, b = 0;   // cluster ids in L[t], L[u]
    double cost = -1.0;
};

WorstPairResult worst_pair_chain(const std::vector<Labels>& L, const std::vector<Mask>& fg,
                                 const Args& args) {
    WorstPairResult out;
    const int T = static_cast<int>(L.size());
    const int npairs = (args.sweep_cycle && T > 2) ? T : T - 1;
    for (int t = 0; t < npairs; ++t) {
        const int u = (t + 1) % T;
        MatchResult mr = match_labels(L[t], L[u], fg[u]);
        for (auto &kv : mr.M) {
            int a = kv.first, b = kv.second;
            auto it = mr.T.find({a, b});
            if (it == mr.T.end()) continue;
            double c = it->second.mismatch;
            if (c > out.cost) { out.cost = c; out.t = t; out.u = u; out.a = a; out.b = b; }
        }
    }
    return out;
}

// Best 2-piece split of a parent mask pair: serial pairwise Level-1.
std::pair<Labels, Labels> best_2piece_split_via_level1(const Mask& mA, const Mask& mB,
                                                       const Args& args, uint64_t seed) {
    const int N_sub = 60, R_sub = 60;
    Args sub = args;
    sub.N = N_sub; sub.R = R_sub; sub.Nb = 1;
    sub.seed = seed + 811;

    std::vector<Mask> fg = {mA, mB};
    std::vector<MultiResult> outs;
    outs.reserve(N_sub);
    for (int i = 0; i < N_sub; ++i) {
        uint64_t s = sub.seed + 1009ull*i + 17ull;
        MultiResult r = _level1_trial(fg, sub, s);
        if (r.valid()) outs.push_back(std::move(r));
    }
    if (outs.empty()) return { Labels(), Labels() };

    auto score = [&](const MultiResult& r)->double {
        return candidate_score_multi(r.L, fg, 1, 2, sub);
    };
    std::sort(outs.begin(), outs.end(),
              [&](const MultiResult& a, const MultiResult& b){ return score(a) < score(b); });

    return { outs.front().L[0], outs.front().L[1] };
}

int max_label_all(const std::vector<Labels>& L) {
    int m = 0;
    for (const auto& g : L)
        for (int y = 0; y < g.rows(); ++y)
            for (int x = 0; x < g.cols(); ++x)
                if (g.at(y,x) > m) m = g.at(y,x);
    return m;
}

// Move one pixel of the largest cluster to new_id (cluster-count harmonizer).
Labels peel(const Labels& L, int nid) {
    Labels out = L;
    auto A = areas(out);
    if (A.empty()) return out;
    int big = 0, bigsz = -1;
    for (auto &kv : A) if (kv.second > bigsz) { big = kv.first; bigsz = kv.second; }
    for (int y = 0; y < out.rows(); ++y) {
        for (int x = 0; x < out.cols(); ++x) {
            if (out.at(y,x) == big) { out.at(y,x) = nid; return out; }
        }
    }
    return out;
}

// Paper heuristic #1, chain/cycle version: split the worst-matched cluster
// pair across any adjacent figures; peel every other figure to keep counts
// equal.
std::vector<Labels> local_split_multi(const std::vector<Labels>& L, const std::vector<Mask>& fg,
                                      const Args& args, uint64_t seed) {
    auto wp = worst_pair_chain(L, fg, args);
    if (wp.cost < 0.0) return L;

    const int t = wp.t;
    const int u = wp.u;
    Mask mA = mask_of(L[t], wp.a);
    Mask mB = mask_of(L[u], wp.b);

    long long areaA = 0, areaB = 0;
    for (auto v : mA.m_data) areaA += (v ? 1 : 0);
    for (auto v : mB.m_data) areaB += (v ? 1 : 0);
    if (areaA < 2 || areaB < 2) return L;

    auto [bestA, bestB] = best_2piece_split_via_level1(mA, mB, args, seed);

    if (bestA.rows() == 0 || bestB.rows() == 0) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        auto ptsA = _fg_pixels(mA);
        auto ptsB = _fg_pixels(mB);
        if (ptsA.size() < 2 || ptsB.size() < 2) return L;
        std::uniform_int_distribution<int> dA(0, static_cast<int>(ptsA.size())-1);
        std::uniform_int_distribution<int> dB(0, static_cast<int>(ptsB.size())-1);
        Point a1 = ptsA[dA(rng)], a2 = ptsA[dA(rng)];
        while (a2 == a1) a2 = ptsA[dA(rng)];
        Point b1 = ptsB[dB(rng)], b2 = ptsB[dB(rng)];
        while (b2 == b1) b2 = ptsB[dB(rng)];
        bestA = two_seed_grow(mA, a1, a2);
        bestB = two_seed_grow(mB, b1, b2);
    }

    int new_id = max_label_all(L) + 1;

    std::vector<Labels> out = L;
    for (int y = 0; y < out[t].rows(); ++y)
        for (int x = 0; x < out[t].cols(); ++x)
            if (mA.at(y,x) && bestA.at(y,x) == 2) out[t].at(y,x) = new_id;
    for (int y = 0; y < out[u].rows(); ++y)
        for (int x = 0; x < out[u].cols(); ++x)
            if (mB.at(y,x) && bestB.at(y,x) == 2) out[u].at(y,x) = new_id;

    for (int g = 0; g < static_cast<int>(out.size()); ++g) {
        if (g == t || g == u) continue;
        out[g] = peel(out[g], new_id);
    }
    return out;
}

// Paper heuristic #2, chain/cycle version: new cluster from mismatch pixels
// of a randomly chosen adjacent direction; peel the rest.
std::vector<Labels> new_cluster_from_mismatches_multi(const std::vector<Labels>& L,
                                                      const std::vector<Mask>& fg,
                                                      const Args& args,
                                                      std::mt19937& rng) {
    const int T = static_cast<int>(L.size());
    struct Dir { int src, dst; Mask mis; long long cnt; };
    std::vector<Dir> dirs;
    long long total = 0;
    const int npairs = (args.sweep_cycle && T > 2) ? T : T - 1;
    for (int t = 0; t < npairs; ++t) {
        const int up = (t + 1) % T;
        for (auto [s, d] : { std::pair<int,int>{t, up}, std::pair<int,int>{up, t} }) {
            Mask m = mismatches_src_outside_dst(L[s], L[d], fg[d]);
            long long c = 0; for (auto v : m.m_data) c += (v ? 1 : 0);
            if (c > 0) { total += c; dirs.push_back({s, d, std::move(m), c}); }
        }
    }
    if (dirs.empty()) return L;

    // Weighted pick by mismatch count (mirrors the 2-figure weighted choice)
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng) * static_cast<double>(total);
    int pick = 0;
    for (int i = 0; i < static_cast<int>(dirs.size()); ++i) {
        r -= static_cast<double>(dirs[i].cnt);
        if (r <= 0.0) { pick = i; break; }
    }
    const Dir& D = dirs[pick];

    int new_id = max_label_all(L) + 1;

    // Flood-fill one connected component of the mismatch mask into new_id
    std::vector<Labels> out = L;
    {
        Labels& g = out[D.dst];
        const int H = g.rows(), W = g.cols();
        std::vector<Point> seeds;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (D.mis.at(y,x)) seeds.emplace_back(y,x);
        if (seeds.empty()) return L;
        std::uniform_int_distribution<int> d(0, static_cast<int>(seeds.size())-1);
        Point s = seeds[d(rng)];

        std::deque<Point> q; q.emplace_back(s);
        g.at(s.first, s.second) = new_id;
        const int dy4[4] = {1,-1,0,0};
        const int dx4[4] = {0,0,1,-1};
        auto inb = [&](int y, int x){ return 0 <= y && y < H && 0 <= x && x < W; };

        while (!q.empty()) {
            auto [y, x] = q.front(); q.pop_front();
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy4[k], nx = x + dx4[k];
                if (!inb(ny, nx) || !D.mis.at(ny,nx)) continue;
                if (g.at(ny,nx) == new_id) continue;
                g.at(ny,nx) = new_id;
                q.emplace_back(ny, nx);
            }
        }
    }

    // Harmonize cluster counts: peel any figure that fell behind
    size_t target = cluster_ids(out[D.dst]).size();
    for (int g = 0; g < T; ++g) {
        if (g == D.dst) continue;
        if (cluster_ids(out[g]).size() < target) out[g] = peel(out[g], new_id);
    }
    return out;
}

// ----------------- Level >= 2 random exploration -----------------
MultiResult _levelK_trial(const MultiResult& cand, const std::vector<Mask>& fg,
                          int level, const Args& args, uint64_t seed) {
    std::vector<Labels> L = cand.L;

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> U(0.0, 1.0);

    if (U(rng) < 0.75) {
        L = local_split_multi(L, fg, args, seed + 37);
    } else {
        L = new_cluster_from_mismatches_multi(L, fg, args, rng);
    }

    return refine_multi(std::move(L), fg, level, level + 1, args, seed + 73);
}

std::vector<MultiResult> explore_next(const std::vector<MultiResult>& prev,
                                      const std::vector<Mask>& fg,
                                      int level, const Args& args, ThreadPool& pool,
                                      uint64_t level_seed) {
    std::vector<MultiResult> outs;
    if (prev.empty()) return outs;

    const int N = args.N;
    std::vector<uint64_t> seeds; seeds.reserve(N);
    for (int i = 0; i < N; ++i) seeds.push_back(level_seed + 2003ull*i + 23ull);

    std::vector<const MultiResult*> picks; picks.reserve(N);
    std::mt19937 picks_rng(static_cast<uint32_t>(level_seed));
    std::uniform_int_distribution<int> d(0, static_cast<int>(prev.size()) - 1);
    for (int i = 0; i < N; ++i) {
        picks.push_back(&prev[d(picks_rng)]);
    }

    std::vector<std::future<MultiResult>> futs; futs.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto s = seeds[i];
        const MultiResult* c = picks[i];
        futs.emplace_back(pool.enqueue([=, &fg, &args](){
            return _levelK_trial(*c, fg, level, args, s);
        }));
    }

    outs.reserve(N);
    for (auto &f : futs) outs.push_back(f.get());
    if (outs.empty()) return outs;

    auto score = [&](const MultiResult& r)->double {
        return candidate_score_multi(r.L, fg, level, level + 1, args);
    };

    std::sort(outs.begin(), outs.end(),
              [&](const MultiResult& a, const MultiResult& b){ return score(a) < score(b); });
    if (static_cast<int>(outs.size()) > args.Nb) outs.resize(args.Nb);
    return outs;
}

// ----------------- Runner -----------------
void save_state(const std::string& outdir, const std::string& prefix,
                const std::vector<Labels>& L, int pcs) {
    std::filesystem::create_directories(outdir);
    for (size_t t = 0; t < L.size(); ++t) {
        save_labels_png(outdir + "/" + prefix + "_" + std::to_string(pcs) + "pcs_S"
                        + std::to_string(t) + ".png", L[t]);
    }
}

int run(Args& args) {
    std::vector<Mask> fg;
    try {
        for (const auto& p : args.shape_paths) fg.push_back(load_binary(p));
    } catch (const std::exception& e) {
        std::cerr << "Error loading images: " << e.what() << std::endl;
        return 1;
    }
    const int T = static_cast<int>(fg.size());

    auto sum_mask = [](const Mask& m)->long long {
        long long s = 0; for (auto v : m.m_data) s += (v ? 1 : 0); return s;
    };

    long long area0 = sum_mask(fg[0]);
    if (area0 <= 0) {
        std::cerr << "Error: empty shape 0." << std::endl;
        return 2;
    }
    for (int t = 1; t < T; ++t) {
        long long at = sum_mask(fg[t]);
        if (at != area0) {
            std::cerr << "Error: shapes must have equal area. S0=" << area0
                      << " S" << t << "=" << at << std::endl;
            return 3;
        }
    }

    int num_threads = args.jobs;
    if (num_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        num_threads = (hc == 0 ? 1 : static_cast<int>(hc));
    }
    ThreadPool pool(static_cast<size_t>(num_threads));
    std::cout << "Using " << num_threads << " threads, " << T << " figures." << std::endl;

    std::cout << "\n=== Level 1 ===" << std::endl;
    auto L1_results = level1(fg, args, pool);
    if (L1_results.empty()) {
        std::cerr << "Level 1 failed." << std::endl;
        return 4;
    }

    MultiResult best = L1_results.front();
    int pcs = static_cast<int>(cluster_ids(best.L.front()).size());
    int lvl = 1;
    auto obj1 = objective_multi(best.L, fg, lvl, args);
    std::cout << "Best L1: Obj=" << std::fixed << std::setprecision(3) << obj1.total
              << "  M=" << std::setprecision(1) << obj1.mismatch
              << "  Area=" << std::setprecision(1) << obj1.area
              << "  Split=" << std::setprecision(0) << obj1.split
              << "  Pieces=" << pcs << std::endl;
    if (!target_piece_count_all(best.L, 2)) {
        std::cout << "Warning: L1 candidate does not have a shared 2-piece label set." << std::endl;
    }

    try {
        save_state(args.outdir, "level", best.L, pcs);
    } catch (const std::exception& e) {
        std::cerr << "Save level error: " << e.what() << std::endl;
    }

    if (obj1.mismatch < CONVERGENCE_EPS && obj1.split < 1.0) {
        std::cout << "*** Perfect dissection at L1 ***" << std::endl;
        try {
            save_state(args.outdir, "result", best.L, pcs);
        } catch (const std::exception& e) {
            std::cerr << "Save error: " << e.what() << std::endl;
        }
        return 0;
    }

    std::vector<MultiResult> cands = std::move(L1_results);
    MultiResult best_overall = best;

    for (int level = 2; level <= args.max_level; ++level) {
        const int target_pieces = level + 1;
        std::cout << "\n=== Level " << level << " (" << target_pieces << " clusters) ===" << std::endl;
        uint64_t current_level_seed = args.seed + static_cast<uint64_t>(level) * 10007ull;
        auto next = explore_next(cands, fg, level, args, pool, current_level_seed);
        if (next.empty()) {
            std::cout << "No candidates; stopping exploration." << std::endl;
            break;
        }
        cands = std::move(next);

        MultiResult cur = cands.front();
        int pcs_cur = static_cast<int>(cluster_ids(cur.L.front()).size());
        auto obj = objective_multi(cur.L, fg, level, args);

        std::cout << "Best L" << level << ": Obj=" << std::fixed << std::setprecision(3) << obj.total
                  << "  M=" << std::setprecision(1) << obj.mismatch
                  << "  Area=" << std::setprecision(1) << obj.area
                  << "  Split=" << std::setprecision(0) << obj.split
                  << "  Pieces=" << pcs_cur << std::endl;
        if (!target_piece_count_all(cur.L, target_pieces)) {
            std::cout << "Warning: L" << level << " candidate does not have a shared "
                      << target_pieces << "-piece label set." << std::endl;
        }

        try {
            save_state(args.outdir, "level", cur.L, pcs_cur);
        } catch (const std::exception& e) {
            std::cerr << "Save level error: " << e.what() << std::endl;
        }

        if (obj.mismatch < CONVERGENCE_EPS && obj.split < 1.0) {
            std::cout << "*** Perfect dissection at L" << level << " ***" << std::endl;
            best_overall = cur;
            break;
        }

        auto best_over_obj_lvl = std::max(1, static_cast<int>(cluster_ids(best_overall.L.front()).size()) - 1);
        auto best_over_obj = objective_multi(best_overall.L, fg, best_over_obj_lvl, args);
        const bool cur_has_target = target_piece_count_all(cur.L, target_pieces);
        const bool best_has_any_target =
            target_piece_count_all(best_overall.L,
                                   std::max(2, static_cast<int>(cluster_ids(best_overall.L.front()).size())));
        if (cur_has_target && (!best_has_any_target || obj.total + 1e-12 < best_over_obj.total)) {
            best_overall = cur;
        }
    }

    pcs = static_cast<int>(cluster_ids(best_overall.L.front()).size());
    lvl = std::max(1, pcs - 1);
    auto objF = objective_multi(best_overall.L, fg, lvl, args);

    std::cout << "\n=== Final ===" << std::endl;
    std::cout << "Pieces: " << pcs
              << "\nObjective: " << std::fixed << std::setprecision(3) << objF.total
              << "  Mismatch: " << std::setprecision(1) << objF.mismatch
              << "  Area: " << std::setprecision(1) << objF.area
              << "  Split: " << std::setprecision(0) << objF.split << std::endl;

    try {
        save_state(args.outdir, "result", best_overall.L, pcs);
    } catch (const std::exception& e) {
        std::cerr << "Save error: " << e.what() << std::endl;
    }

    return (objF.mismatch < CONVERGENCE_EPS && objF.split < 1.0) ? 0 : 1;
}

// ----------------- Main -----------------
int main(int argc, char* argv[]) {
    Args args;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <shape0.png> <shape1.png> [<shape2.png> ...] [options...]" << std::endl;
        std::cerr << "Options: -o outdir -N trials -R refinements -Nb keep --max-level L --seed S -j jobs" << std::endl;
        return 1;
    }

    int i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (!a.empty() && a[0] == '-') break;
        args.shape_paths.push_back(a);
    }
    if (args.shape_paths.size() < 2) {
        std::cerr << "Error: need at least two shape paths." << std::endl;
        return 1;
    }

    for (; i < argc; ++i) {
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
        else if (a == "--em" && i+1 < argc) {
            std::string v = argv[++i];
            args.em_allpairs = (v == "all-pairs" || v == "allpairs" || v == "all");
        }
        else if (a == "--sweep" && i+1 < argc) {
            std::string v = argv[++i];
            args.sweep_cycle = (v == "cycle" || v == "ring");
        }
    }

    if (args.outdir.empty()) {
        args.outdir = "out_dir_multi";
    }

    std::cout << "Starting C++ multi-figure dissection..." << std::endl;
    for (size_t t = 0; t < args.shape_paths.size(); ++t)
        std::cout << "  Shape " << t << ": " << args.shape_paths[t] << std::endl;
    std::cout << "  Params: -N " << args.N << " -Nb " << args.Nb << " -R " << args.R
              << " -j " << args.jobs << " --seed " << args.seed
              << " --max-level " << args.max_level
              << " --em " << (args.em_allpairs ? "all-pairs" : "chain")
              << " --sweep " << (args.sweep_cycle ? "cycle" : "chain")
              << " --lambda0 " << args.lambda0
              << " --decay " << args.decay << std::endl;

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
