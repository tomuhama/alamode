/* Retarded resolvent of a linearly interpolated tetrahedron and projector. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace PHON_NS {
namespace tmatrix_kernel {

using ComplexLong = std::complex<long double>;

inline void average_degenerate_weights(const double *frequency, const unsigned int ns,
                                       std::complex<double> *weight)
{
    // Match Conductivity's tolerance; only the whole degenerate projector is basis invariant.
    const double tolerance = 1.0e-7;
    for (unsigned int first = 0; first < ns;) {
        auto last = first + 1;
        while (last < ns && std::abs(frequency[last] - frequency[first]) < tolerance) ++last;
        std::complex<double> sum = 0.0;
        for (auto s = first; s < last; ++s) sum += weight[s];
        for (auto s = first; s < last; ++s) weight[s] = sum / static_cast<double>(last - first);
        first = last;
    }
}

// F'''(x)=log(z-x). Repeating vertex i in its fourth divided difference
// gives integral(lambda_i/(z-e)), with the tetrahedron volume normalized to one.
inline ComplexLong log_primitive_derivative(const long double x,
                                            const ComplexLong z,
                                            const int n)
{
    const auto d = x - z;
    const auto logarithm = std::log(-d);
    if (n == 0) return d * d * d * (logarithm - 11.0L / 6.0L) / 6.0L;
    if (n == 1) return d * d * (logarithm - 1.5L) / 2.0L;
    if (n == 2) return d * (logarithm - 1.0L);
    if (n == 3) return logarithm;
    ComplexLong result = 1.0L / d;
    for (int j = 4; j < n; ++j) result *= -static_cast<long double>(j - 3) / d;
    return result;
}

inline ComplexLong confluent_difference(const long double *x, const int order,
                                         const ComplexLong z)
{
    const auto center = (x[0] + x[order]) / 2.0L;
    // Complete homogeneous polynomials yield the Taylor divided difference.
    std::array<long double, 9> h{};
    h[0] = 1.0L;
    for (int i = 0; i <= order; ++i)
        for (int k = 1; k < 9; ++k) h[k] += (x[i] - center) * h[k - 1];
    long double factorial = 1.0L;
    for (int k = 2; k <= order; ++k) factorial *= k;
    ComplexLong result = 0.0L;
    for (int k = 0; k < 9; ++k) {
        if (k) factorial *= order + k;
        result += h[k] * log_primitive_derivative(center, z, order + k) / factorial;
    }
    return result;
}

inline std::array<std::complex<double>, 4>
tetrahedron_resolvent(const std::array<double, 4> &energy,
                      const std::complex<double> z_in)
{
    if (!(z_in.imag() > 0.0) || !std::isfinite(z_in.real()) || !std::isfinite(z_in.imag())
        || !std::all_of(energy.begin(), energy.end(), [](double e) { return std::isfinite(e); })) {
        throw std::runtime_error("Retarded tetrahedron requires finite energies and Im(z)>0");
    }
    const auto bounds = std::minmax_element(energy.begin(), energy.end());
    const long double center = (*bounds.first + *bounds.second) / 2.0L;
    const long double span = *bounds.second - *bounds.first;
    const ComplexLong dz = ComplexLong(z_in.real(), z_in.imag()) - center;
    std::array<std::complex<double>, 4> result;
    if (span == 0.0L) {
        result.fill(static_cast<std::complex<double>>(0.25L / dz));
        return result;
    }
    std::array<long double, 4> x;
    for (int i = 0; i < 4; ++i) x[i] = (energy[i] - center) / span;
    if (std::abs(dz) > 2.0L * span) {
        // Far from the tetrahedron, the moment series avoids log cancellation.
        std::array<long double, 21> h{};
        h[0] = 1.0L;
        for (const auto v: x)
            for (int k = 1; k < 21; ++k) h[k] += v * h[k - 1];
        for (int i = 0; i < 4; ++i) {
            long double repeated = 0.0L;
            ComplexLong sum = 0.0L, power = 1.0L / dz;
            for (int k = 0; k < 21; ++k) {
                repeated = h[k] + x[i] * repeated;
                sum += power * (6.0L * repeated / ((k + 1.L) * (k + 2.L) * (k + 3.L) * (k + 4.L)));
                power *= span / dz;
            }
            result[i] = static_cast<std::complex<double>>(sum);
        }
        return result;
    }
    const auto z = dz / span;
    for (int vertex = 0; vertex < 4; ++vertex) {
        std::array<long double, 5> nodes{x[0], x[1], x[2], x[3], x[vertex]};
        std::sort(nodes.begin(), nodes.end());
        std::array<ComplexLong, 5> divided;
        for (int i = 0; i < 5; ++i) divided[i] = log_primitive_derivative(nodes[i], z, 0);
        for (int order = 1; order < 5; ++order) {
            for (int i = 0; i < 5 - order; ++i) {
                const auto width = nodes[i + order] - nodes[i];
                const auto midpoint = (nodes[i + order] + nodes[i]) / 2.0L;
                if (width < 1.0e-3L * std::abs(z - midpoint)) {
                    divided[i] = confluent_difference(nodes.data() + i, order, z);
                } else {
                    divided[i] = (divided[i + 1] - divided[i]) / width;
                }
            }
        }
        result[vertex] = static_cast<std::complex<double>>(-6.0L * divided[0] / span);
    }
    return result;
}

} // namespace tmatrix_kernel
} // namespace PHON_NS
