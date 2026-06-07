
#include <nlopt.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <tuple>
#include <iomanip>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>   

static constexpr double PI = 3.141592653589793238462643383279502884;
static std::vector<double> linspace(double a, double b, int n, bool endpoint = true) {
    std::vector<double> x;
    x.reserve(n);
    if (n <= 0) return x;
    if (n == 1) { x.push_back(a); return x; }
    if (endpoint) {
        double step = (b - a) / (n - 1);
        for (int i = 0; i < n; ++i) x.push_back(a + step * i);
    } else {
        double step = (b - a) / n;
        for (int i = 0; i < n; ++i) x.push_back(a + step * i);
    }
    return x;
}
static double interp1(const std::vector<double>& x, const std::vector<double>& y, double xq) {
    if (xq <= x.front()) return y.front();
    if (xq >= x.back())  return y.back();
    int lo = 0, hi = (int)x.size() - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[mid] <= xq) lo = mid; else hi = mid;
    }
    double t = (xq - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + t * (y[hi] - y[lo]);
}
static std::vector<Eigen::Vector2d> subsample_fixed(const std::vector<Eigen::Vector2d>& full,
                                                    int M_desired) {
    const int Mmax = (int)full.size();
    if (M_desired <= 0) throw std::runtime_error("M_desired must be positive");
    if (M_desired > Mmax) throw std::runtime_error("M_desired cannot exceed full.size()");
    if (Mmax % M_desired != 0)
        throw std::runtime_error("For nested subsampling, Mmax must be divisible by M_desired.");
    const int step = Mmax / M_desired;
    std::vector<Eigen::Vector2d> out;
    out.reserve(M_desired);
    for (int i = 0; i < Mmax; i += step) out.push_back(full[i]);
    if ((int)out.size() != M_desired)
        throw std::runtime_error("subsample_fixed produced wrong size");
    return out;
}
static std::vector<int> hungarian_min_cost(const std::vector<std::vector<double>>& cost) {
    const int n = (int)cost.size();
    if (n == 0) return {};
    for (int i = 0; i < n; ++i)
        if ((int)cost[i].size() != n) throw std::runtime_error("hungarian: cost must be NxN");
    const double INF = 1e300;
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i; int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true; int i0 = p[j0]; int j1 = 0; double delta = INF;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
    }
    std::vector<int> col_for_row(n, -1);
    for (int j = 1; j <= n; ++j) {
        int row = p[j];
        if (row >= 1 && row <= n) col_for_row[row - 1] = j - 1;
    }
    for (int i = 0; i < n; ++i)
        if (col_for_row[i] < 0) throw std::runtime_error("hungarian: incomplete assignment");
    return col_for_row;
}
static double rms_hungarian(const std::vector<Eigen::Vector2d>& eq,
                            const std::vector<Eigen::Vector2d>& target) {
    const int n = (int)eq.size();
    if ((int)target.size() != n) return std::numeric_limits<double>::infinity();
    if (n == 0) return 0.0;
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cost[i][j] = (eq[i] - target[j]).squaredNorm();
    std::vector<int> assign = hungarian_min_cost(cost);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += cost[i][assign[i]];
    return std::sqrt(sum / (double)n);
}
static void write_best_summary_csv(const std::string& path,
                                   const std::vector<double>& c_cos,  
                                   const std::vector<double>& c_sin,  
                                   double best_rms, double energy, bool success,
                                   const std::string& message) {
    const int K = (int)c_sin.size();  
    std::ofstream f(path);
    f << "c0";
    for (int k = 1; k <= K; ++k) f << ",c" << k;
    for (int k = 1; k <= K; ++k) f << ",s" << k;
    f << ",best_rms,energy,success,message\n";
    f << std::setprecision(17) << c_cos[0];
    for (int k = 1; k <= K; ++k) f << "," << c_cos[k];
    for (int k = 0; k <  K; ++k) f << "," << c_sin[k];
    f << "," << best_rms << "," << energy << ","
      << (success ? 1 : 0) << ",\"";
    for (char ch : message) { if (ch == '"') f << "\"\""; else f << ch; }
    f << "\"\n";
}
static void write_equilibrium_csv(const std::string& path,
                                  const std::vector<Eigen::Vector2d>& eq,
                                  const std::vector<Eigen::Vector2d>& target) {
    std::ofstream f(path);
    f << "i,eq_x,eq_y,target_x,target_y\n";
    f << std::setprecision(17);
    const size_t n = std::min(eq.size(), target.size());
    for (size_t i = 0; i < n; ++i)
        f << i << "," << eq[i].x() << "," << eq[i].y() << ","
          << target[i].x() << "," << target[i].y() << "\n";
}
static void write_fixed_csv(const std::string& path,
                            const std::vector<Eigen::Vector2d>& fixed) {
    std::ofstream f(path);
    f << "k,x,y\n";
    f << std::setprecision(17);
    for (size_t k = 0; k < fixed.size(); ++k)
        f << k << "," << fixed[k].x() << "," << fixed[k].y() << "\n";
}

static void write_density_csv(const std::string& path,
                              const std::vector<double>& c_cos,  
                              const std::vector<double>& c_sin,  
                              int n = 600) {
    const int K = (int)c_sin.size();
    std::ofstream f(path);
    f << "theta,n\n";
    f << std::setprecision(17);
    auto theta = linspace(0.0, 2.0 * PI, n, false);
    for (double th : theta) {
        double n_th = c_cos[0];
        for (int k = 1; k <= K; ++k)
            n_th += c_cos[k] * std::cos(k * th) + c_sin[k - 1] * std::sin(k * th);
        f << th << "," << n_th << "\n";
    }
}
struct MagnetSystem {
    int M;
    int N;
    double R;
    double m;
    double m_ext;
    double mu0;
    std::vector<double> c_cos;
    std::vector<double> c_sin;
    std::vector<Eigen::Vector2d> fixed_positions;
    MagnetSystem(int M_, int N_, double R_, double m_, double m_ext_,
                 const std::vector<double>& c_cos_,
                 const std::vector<double>& c_sin_,
                 const std::vector<Eigen::Vector2d>& fixed_positions_,
                 double mu0_ = 4.0 * PI * 1e-7)
        : M(M_), N(N_), R(R_), m(m_), m_ext(m_ext_), mu0(mu0_),
          c_cos(c_cos_), c_sin(c_sin_), fixed_positions(fixed_positions_) {
        if ((int)c_sin.size() != (int)c_cos.size() - 1)
            throw std::runtime_error("c_sin must have length len(c_cos)-1");
        if ((int)fixed_positions.size() != M)
            throw std::runtime_error("fixed_positions size must equal M");
    }
    static std::vector<Eigen::Vector2d> create_fixed_magnets_from_density(
        int M_create, double R,
        const std::vector<double>& c_cos,
        const std::vector<double>& c_sin,
        int m_grid = 8000) {
        if ((int)c_sin.size() != (int)c_cos.size() - 1)
            throw std::runtime_error("c_sin must have length len(c_cos)-1");
        if (M_create <= 0) throw std::runtime_error("M_create must be positive");
        if (m_grid < 32)   throw std::runtime_error("m_grid too small");
        auto density_fn = [&](double theta) {
            double val = c_cos[0];
            int K = (int)c_cos.size() - 1;
            for (int k = 1; k <= K; ++k)
                val += c_cos[k] * std::cos(k * theta) + c_sin[k - 1] * std::sin(k * theta);
            if (val < 1e-12) val = 1e-12;
            return val;
        };
        std::vector<double> grid = linspace(0.0, 2.0 * PI, m_grid, false);
        std::vector<double> vals; vals.reserve(m_grid);
        for (double th : grid) vals.push_back(density_fn(th));
        std::vector<double> cdf(m_grid, 0.0);
        double cum = 0.0;
        for (int i = 0; i < m_grid; ++i) { cum += vals[i]; cdf[i] = cum; }
        for (int i = 0; i < m_grid; ++i) cdf[i] /= cdf.back();
        std::vector<Eigen::Vector2d> pos; pos.reserve(M_create);
        for (int i = 0; i < M_create; ++i) {
            double u = (i + 0.5) / (double)M_create;
            double ang = interp1(cdf, grid, u);
            pos.emplace_back(R * std::cos(ang), R * std::sin(ang));
        }
        return pos;
    }
    static inline double dist(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return (a - b).norm();
    }
    double potential_energy(const std::vector<double>& x) const {
        std::vector<Eigen::Vector2d> pos(N);
        for (int i = 0; i < N; ++i) pos[i] = Eigen::Vector2d(x[2*i], x[2*i+1]);
        double total = 0.0;
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < M; ++k) {
                double r = dist(pos[i], fixed_positions[k]);
                if (r > 1e-10) total += (m * m_ext) / (r * r * r);
            }
        total *= mu0 / (4.0 * PI);
        double mm = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j) {
                double r = dist(pos[i], pos[j]);
                if (r > 1e-10) mm += (m * m) / (r * r * r);
            }
        total += (mu0 / (8.0 * PI)) * mm;
        return total;
    }
    void gradient(const std::vector<double>& x, std::vector<double>& g) const {
        g.assign(2 * N, 0.0);
        std::vector<Eigen::Vector2d> pos(N);
        for (int i = 0; i < N; ++i) pos[i] = Eigen::Vector2d(x[2*i], x[2*i+1]);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector2d gi(0.0, 0.0);
            for (int k = 0; k < M; ++k) {
                Eigen::Vector2d rvec = pos[i] - fixed_positions[k];
                double r = rvec.norm();
                if (r > 1e-10) {
                    double force_mag = (mu0 / (4.0 * PI)) * (m * m_ext) * (-3.0 / std::pow(r, 5));
                    gi += force_mag * rvec;
                }
            }
            g[2*i]   += gi.x();
            g[2*i+1] += gi.y();
        }
        for (int i = 0; i < N; ++i) {
            Eigen::Vector2d gi(0.0, 0.0);
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                Eigen::Vector2d rvec = pos[i] - pos[j];
                double r = rvec.norm();
                if (r > 1e-10) {
                    double force_mag = (mu0 / (8.0 * PI)) * (m * m) * (-3.0 / std::pow(r, 5));
                    gi += force_mag * rvec;
                }
            }
            g[2*i]   += gi.x();
            g[2*i+1] += gi.y();
        }
    }
    struct OptResult {
        std::vector<Eigen::Vector2d> eq_pos;
        double energy = std::numeric_limits<double>::quiet_NaN();
        bool success  = false;
        std::string message;
    };
    static double nlopt_obj_cb(unsigned n, const double* x, double* grad, void* data) {
        MagnetSystem* sys = reinterpret_cast<MagnetSystem*>(data);
        std::vector<double> xv(x, x + n);
        if (grad) {
            std::vector<double> g;
            sys->gradient(xv, g);
            for (unsigned i = 0; i < n; ++i) grad[i] = g[i];
        }
        return sys->potential_energy(xv);
    }
    OptResult find_equilibrium(const std::vector<Eigen::Vector2d>& initial_positions,
                               double tol = 1e-10) {
        if ((int)initial_positions.size() != N)
            throw std::runtime_error("initial_positions wrong size");
        std::vector<double> x0(2 * N);
        for (int i = 0; i < N; ++i) {
            x0[2*i]     = initial_positions[i].x();
            x0[2*i + 1] = initial_positions[i].y();
        }
        nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, (unsigned)(2 * N));
        if (!opt) throw std::runtime_error("nlopt_create failed");
        std::vector<double> lb(2 * N, -R), ub(2 * N, R);
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_upper_bounds(opt, ub.data());
        nlopt_set_min_objective(opt, &MagnetSystem::nlopt_obj_cb, this);
        nlopt_set_xtol_abs1(opt, tol);
        nlopt_set_ftol_abs(opt, 1e-14);
        nlopt_set_ftol_rel(opt, 1e-12);
        nlopt_set_maxeval(opt, 20000);  
        OptResult out;
        double minf = 0.0;
        nlopt_result r = nlopt_optimize(opt, x0.data(), &minf);
        out.energy = minf;
        out.eq_pos.resize(N);
        for (int i = 0; i < N; ++i)
            out.eq_pos[i] = Eigen::Vector2d(x0[2*i], x0[2*i+1]);
        out.success = (r > 0);
        out.message = "nlopt_result code: " + std::to_string((int)r);
        nlopt_destroy(opt);
        return out;
    }
};
static MagnetSystem::OptResult find_best_equilibrium(
    MagnetSystem& system,
    const std::vector<Eigen::Vector2d>& initial_positions,  
    const std::vector<Eigen::Vector2d>& target_positions,
    int N_RESTARTS,
    std::mt19937& rng)
{
    const int N   = system.N;
    const double R = system.R * 0.75;  
    MagnetSystem::OptResult best_res;
    double best_rms = std::numeric_limits<double>::infinity();
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * PI);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    for (int trial = 0; trial < N_RESTARTS; ++trial) {
        std::vector<Eigen::Vector2d> init(N);
        if (trial == 0) {
            init = initial_positions;  
        } else {
            for (int i = 0; i < N; ++i) {
                double r   = R * std::sqrt(unit_dist(rng));
                double ang = angle_dist(rng);
                init[i]    = Eigen::Vector2d(r * std::cos(ang), r * std::sin(ang));
            }
        }
        auto res = system.find_equilibrium(init, 1e-10);
        if (!res.success) continue;
        bool any_nan = false;
        for (auto& p : res.eq_pos)
            if (!std::isfinite(p.x()) || !std::isfinite(p.y())) { any_nan = true; break; }
        if (any_nan) continue;
        double rms = rms_hungarian(res.eq_pos, target_positions);
        if (rms < best_rms) {
            best_rms = rms;
            best_res = res;
        }
    }
    return best_res;
}
struct OuterData {
    int M, M_master, N, K;   
    double R, m, m_ext, c0;
    const std::vector<Eigen::Vector2d>* initial_positions;
    const std::vector<Eigen::Vector2d>* target_positions;
    int N_RESTARTS;
    std::mt19937* rng;
    long long total_evals       = 0;
    long long successful_solves = 0;
    double best_rms_seen        = std::numeric_limits<double>::infinity();
    std::vector<double> best_x;              
    MagnetSystem::OptResult best_result;     
};
static double outer_objective(unsigned /*ndim*/, const double* x,
                              double* /*grad*/, void* data) {
    auto* d = reinterpret_cast<OuterData*>(data);
    d->total_evals++;
    const double c0 = d->c0;
    const int K     = d->K;
    std::vector<double> c_cos(K + 1), c_sin(K);
    c_cos[0] = c0;
    for (int k = 0; k < K; ++k) c_cos[k + 1] = x[k];
    for (int k = 0; k < K; ++k) c_sin[k]     = x[K + k];
    const int N_CHECK = 720;
    double min_n = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_CHECK; ++i) {
        double th  = (2.0 * PI * i) / N_CHECK;
        double n_th = c0;
        for (int k = 1; k <= K; ++k)
            n_th += c_cos[k] * std::cos(k * th) + c_sin[k - 1] * std::sin(k * th);
        if (n_th < min_n) min_n = n_th;
    }
    const double PENALTY_WEIGHT = 500.0;
    double density_penalty = 0.0;
    if (min_n <= 1e-3) {
        density_penalty = PENALTY_WEIGHT * (1e-3 - min_n);
        if (min_n < -0.5) return density_penalty;  
    }
    int m_grid = std::max(8000, 50 * d->M_master);
    std::vector<Eigen::Vector2d> fixed_master;
    try {
        fixed_master = MagnetSystem::create_fixed_magnets_from_density(
            d->M_master, d->R, c_cos, c_sin, m_grid);
    } catch (...) { return density_penalty + 1e6; }
    std::vector<Eigen::Vector2d> fixed_used;
    try {
        fixed_used = subsample_fixed(fixed_master, d->M);
    } catch (...) { return density_penalty + 1e6; }
    MagnetSystem system(d->M, d->N, d->R, d->m, d->m_ext, c_cos, c_sin, fixed_used);
    std::size_t seed = 0;
    for (int i = 0; i < 2 * K; ++i) {
        std::size_t h = std::hash<double>{}(x[i]);
        seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    std::mt19937 local_rng(static_cast<uint32_t>(seed));
    auto res = find_best_equilibrium(system,
                                     *d->initial_positions,
                                     *d->target_positions,
                                     d->N_RESTARTS,
                                     local_rng);
    if (!res.success || res.eq_pos.empty()) return density_penalty + 1e6;
    d->successful_solves++;
    double rms = rms_hungarian(res.eq_pos, *d->target_positions) + density_penalty;
    if (rms < d->best_rms_seen) {
        d->best_rms_seen = rms;
        d->best_x.assign(x, x + 2 * K);
        d->best_result = res;   
        std::cout << "  [eval " << d->total_evals << "] new best RMS = "
                  << std::fixed << std::setprecision(6) << rms << "\n";
    }
    return rms;
}
int main() {
    int M        = 640;
    int M_master = 640;
    int N        = 5;
    double R     = 1.0;
    double m     = 1.75;
    double m_ext = 0.125;
//	 std::vector<Eigen::Vector2d> target_positions = {
  //      { 0.2920,  0.3744},
//{ 0.0000,  0.5616},
//{-0.2920,  0.3744},
//{-0.2920,  0.1872},
//{ 0.0000,  0.0000},
//{ 0.2920, -0.1872},
//{ 0.2920, -0.3744},
//{ 0.0000, -0.5616},
//{-0.2920, -0.3744},
  //  };
   // std::vector<Eigen::Vector2d> initial_positions = {
   //     {-0.30,  0.50},
     //   { 0.010, 0.40},
       // {-0.10,  0.20},
       // { 0.00, -0.15},
       // {-0.15,  0.00},
       // {0.1,0.2},
       // {-0.1,0.4},
       // {0.0,0.2},
       // {0.3,0.3}
   // };
    std::vector<Eigen::Vector2d> target_positions = {
        {-0.35, -0.20},
        {-0.20,  0.25},
        { 0.00, -0.10},
        { 0.20,  0.25},
        {0.35, -0.20}
        };
    std::vector<Eigen::Vector2d> initial_positions = {
        {-0.30,  0.50},
        { 0.010, 0.40},
        {-0.10,  0.20},
        { 0.00, -0.15},
        {-0.15,  0.00},
    };
    if ((int)target_positions.size() != N || (int)initial_positions.size() != N) {
        std::cerr << "Error: target_positions and initial_positions must both have size N=" << N << "\n";
        return 1;
    }
    if (M_master % M != 0) {
        std::cerr << "Error: M_master (" << M_master << ") must be divisible by M (" << M << ").\n";
        return 1;
    }
    double c0        = 1.0;
    double tolerance = 0.2;
    const int K = 1;
    const int NDIM = 2 * K;
    std::vector<double> lb(NDIM, -1.0);
    std::vector<double> ub(NDIM,  1.0);
    std::mt19937 rng(3);
    const int N_RESTARTS = 10;
    OuterData outer_data;
    outer_data.K                 = K;
    outer_data.M                 = M;
    outer_data.M_master          = M_master;
    outer_data.N                 = N;
    outer_data.R                 = R;
    outer_data.m                 = m;
    outer_data.m_ext             = m_ext;
    outer_data.c0                = c0;
    outer_data.initial_positions = &initial_positions;
    outer_data.target_positions  = &target_positions;
    outer_data.N_RESTARTS        = N_RESTARTS;
    outer_data.rng               = &rng;
    std::cout << "CRS2-LM, maxeval=20000\n";
    nlopt_opt opt_global = nlopt_create(NLOPT_GN_CRS2_LM, NDIM);
    nlopt_set_lower_bounds(opt_global, lb.data());
    nlopt_set_upper_bounds(opt_global, ub.data());
    nlopt_set_min_objective(opt_global, outer_objective, &outer_data);
    nlopt_set_maxeval(opt_global, 20000);
    nlopt_set_ftol_abs(opt_global, 1e-6);
    std::vector<double> x_best(NDIM, 0.0); 
    double f_best = 1e99;
    nlopt_result r1 = nlopt_optimize(opt_global, x_best.data(), &f_best);
    nlopt_destroy(opt_global);
    std::cout << "Phase 1 done. nlopt code=" << (int)r1
              << "  best RMS so far=" << std::fixed << std::setprecision(6) << f_best << "\n\n";
    std::cout << "BOBYQA, maxeval=500\n";
    nlopt_opt opt_local = nlopt_create(NLOPT_LN_BOBYQA, NDIM);
    nlopt_set_lower_bounds(opt_local, lb.data());
    nlopt_set_upper_bounds(opt_local, ub.data());
    nlopt_set_min_objective(opt_local, outer_objective, &outer_data);
    nlopt_set_maxeval(opt_local, 500);
    nlopt_set_xtol_rel(opt_local, 1e-8);
    nlopt_set_ftol_abs(opt_local, 1e-8);
    nlopt_result r2 = nlopt_optimize(opt_local, x_best.data(), &f_best);
    nlopt_destroy(opt_local);
    std::cout << "Phase 2 done. nlopt code=" << (int)r2
              << "  final best RMS=" << std::fixed << std::setprecision(6) << f_best << "\n\n";
    if (outer_data.best_rms_seen < f_best && !outer_data.best_x.empty()) {
        x_best  = outer_data.best_x;
        f_best  = outer_data.best_rms_seen;
        std::cout << "Note: Phase 2 worse than Phase 1. "
                  << "Restored Phase 1 best (RMS=" << f_best << ").\n\n";
    }
    std::vector<double> c_cos_best(K + 1), c_sin_best(K);
    c_cos_best[0] = c0;
    for (int k = 0; k < K; ++k) c_cos_best[k + 1] = x_best[k];
    for (int k = 0; k < K; ++k) c_sin_best[k]     = x_best[K + k];
    std::cout << "Best coefficients:\n";
    for (int k = 1; k <= K; ++k)
        std::cout << "  c" << k << "=" << c_cos_best[k]
                  << "  s" << k << "=" << c_sin_best[k - 1] << "\n";
    int m_grid_best = std::max(8000, 50 * M_master);
    auto fixed_master_best = MagnetSystem::create_fixed_magnets_from_density(
        M_master, R, c_cos_best, c_sin_best, m_grid_best);
    auto fixed_used_best = subsample_fixed(fixed_master_best, M);
    MagnetSystem system_best(M, N, R, m, m_ext, c_cos_best, c_sin_best, fixed_used_best);
    auto& res_best  = outer_data.best_result;
    double best_rms = outer_data.best_rms_seen;
    std::cout << "\nEquilibrium from best:\n" << std::setprecision(6);
    for (int i = 0; i < N; ++i)
        std::cout << "Floater " << (i+1) << ": ("
                  << res_best.eq_pos[i].x() << ", "
                  << res_best.eq_pos[i].y() << ")\n";
    std::cout << "Best RMS error: " << best_rms << "\n";
    write_best_summary_csv("best_summary.csv",
                           c_cos_best, c_sin_best,
                           best_rms, res_best.energy, res_best.success, res_best.message);
    write_equilibrium_csv("best_equilibrium.csv", res_best.eq_pos, target_positions);
    write_fixed_csv("best_fixed.csv", system_best.fixed_positions);
    write_density_csv("best_density.csv", c_cos_best, c_sin_best, 600);
    std::cout << "\nWrote CSV files:\n"
              << "  best_summary.csv\n"
              << "  best_equilibrium.csv\n"
              << "  best_fixed.csv\n"
              << "  best_density.csv\n";
    if (best_rms <= tolerance)
        std::cout << "Best RMS is within tolerance (" << tolerance << ").\n";
    else
        std::cout << "Best RMS is NOT within tolerance (" << tolerance << ").\n";
    return 0;
}
