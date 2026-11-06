/* Independent finite-cluster defect scattering. See docs/source/anphondir/tmatrix.rst. */
#pragma once

#include "phonon.h"
#include "ndarray.h"
#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace PHON_NS {
class System; class Symmetry; class Kpoint; class Dos; class Dynamical; class Isotope;
class Tmatrix {
public:
    Tmatrix(const RunInfo &run, const System *system, const Symmetry *symmetry,
            const Kpoint *kpoint, const Dos *dos, const Dynamical *dynamical, const Isotope *isotope);
    ~Tmatrix() = default;

    std::string file;
    double epsilon_kayser = -1.0;
    bool epsilon_specified = false;
    int integration_method = 0;
    NDArray<double, 2> gamma_tmatrix;

    bool enabled() const { return !file.empty(); }
    void setup();
    void calculate();
    void write_selfenergy() const;

private:
    const RunInfo &run;
    const System *system;
    const Symmetry *symmetry;
    const Kpoint *kpoint;
    const Dos *dos;
    const Dynamical *dynamical;
    const Isotope *isotope;
    struct Site {
        int atom;
        std::array<int, 3> cell;
        bool operator==(const Site &other) const { return atom == other.atom && cell == other.cell; }
    };
    struct Defect {
        std::string name;
        double density;
        std::vector<Site> sites;
        Eigen::MatrixXd k;
        Eigen::VectorXd mass_ratio;
    };
    std::vector<Defect> models;
    double eta = 0.0;

    void read_models();
    void broadcast_models();
    void amplitudes(const Defect &model, unsigned int iq, Eigen::MatrixXcd &b) const;
};
} // namespace PHON_NS
