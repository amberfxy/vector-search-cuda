#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

// Online / offline latency statistics over measured samples (milliseconds).
// Warm-up samples must NOT be pushed into this object.
struct BenchStats {
    std::vector<double> samples_ms;

    void add(double ms) { samples_ms.push_back(ms); }
    void clear() { samples_ms.clear(); }
    size_t count() const { return samples_ms.size(); }

    double mean() const {
        require_nonempty();
        double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
        return sum / static_cast<double>(samples_ms.size());
    }

    double min() const {
        require_nonempty();
        return *std::min_element(samples_ms.begin(), samples_ms.end());
    }

    double max() const {
        require_nonempty();
        return *std::max_element(samples_ms.begin(), samples_ms.end());
    }

    // Nearest-rank percentile on a sorted copy. p in [0, 100].
    double percentile(double p) const {
        require_nonempty();
        if (p < 0.0 || p > 100.0) {
            throw std::invalid_argument("percentile must be in [0, 100]");
        }
        std::vector<double> sorted = samples_ms;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() == 1) return sorted[0];
        const double rank = (p / 100.0) * static_cast<double>(sorted.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(rank));
        const size_t hi = static_cast<size_t>(std::ceil(rank));
        if (lo == hi) return sorted[lo];
        const double w = rank - static_cast<double>(lo);
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }

    double p50() const { return percentile(50.0); }
    double p95() const { return percentile(95.0); }
    double p99() const { return percentile(99.0); }

    // Queries per second from mean latency in milliseconds (single-stream).
    double qps_from_mean() const {
        const double m = mean();
        if (m <= 0.0) return 0.0;
        return 1000.0 / m;
    }

private:
    void require_nonempty() const {
        if (samples_ms.empty()) {
            throw std::runtime_error("BenchStats: no samples (did warm-up exclude all iterations?)");
        }
    }
};
