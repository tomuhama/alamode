/* Finite-cluster retarded Green function and T matrix, in ANPHON units. */
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace PHON_NS {
namespace tmatrix_kernel {

inline void accumulate_green(Eigen::MatrixXcd &green,
                             const Eigen::MatrixXcd &amplitude,
                             const double *frequency,
                             const double omega,
                             const double eta)
{
    const auto z2 = std::complex<double>(omega, eta) * std::complex<double>(omega, eta);
    for (Eigen::Index s = 0; s < amplitude.cols(); ++s) {
        green.noalias() += (amplitude.col(s) * amplitude.col(s).adjoint())
                          / (z2 - frequency[s] * frequency[s]);
    }
}

struct Solution {
    Eigen::MatrixXcd t;
    double residual;
    double rcond;
};

inline Solution solve(const Eigen::MatrixXcd &v, const Eigen::MatrixXcd &green)
{
    if (v.rows() != v.cols() || green.rows() != v.rows() || green.cols() != v.cols()
        || !v.allFinite() || !green.allFinite()) {
        throw std::runtime_error("Invalid finite-cluster T-matrix dimensions or values");
    }
    if (v.isZero(0.0)) return {Eigen::MatrixXcd::Zero(v.rows(), v.cols()), 0.0, 1.0};
    const Eigen::MatrixXcd a = Eigen::MatrixXcd::Identity(v.rows(), v.cols()) - v * green;
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(a);
    Solution result{lu.solve(v), 0.0, lu.rcond()};
    const auto scale = a.norm() * result.t.norm() + v.norm();
    result.residual = (a * result.t - v).norm() / scale;
    if (!result.t.allFinite() || !std::isfinite(result.residual)
        || !std::isfinite(result.rcond) || result.residual > 1.0e-10) {
        throw std::runtime_error("T-matrix solve failed its backward-residual check");
    }
    return result;
}

inline double halfwidth(const std::complex<double> selfenergy, const double omega)
{
    if (!(omega > 0.0) || !std::isfinite(omega)
        || !std::isfinite(selfenergy.real()) || !std::isfinite(selfenergy.imag())) {
        throw std::runtime_error("Invalid frequency or selfenergy in T-matrix halfwidth");
    }
    const auto gamma = -selfenergy.imag() / (2.0 * omega);
    const auto tolerance = 1.0e-15 + 1.0e-10 * std::abs(selfenergy) / (2.0 * omega);
    if (gamma < -tolerance) throw std::runtime_error("Negative T-matrix halfwidth");
    return std::max(0.0, gamma);
}

} // namespace tmatrix_kernel
} // namespace PHON_NS
