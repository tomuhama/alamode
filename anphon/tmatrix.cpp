#include "mpi_common.h"
#include "tmatrix.h"
#include "tmatrix_kernel.h"
#include "tmatrix_tetrahedron.h"
#include "constants.h"
#include "dynamical.h"
#include "isotope.h"
#include "kpoint.h"
#include "memory.h"
#include "phonon_dos.h"
#include "symmetry_core.h"
#include "system.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>

using namespace PHON_NS;
namespace {
using boost::property_tree::ptree;

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void children(const ptree &node, const std::set<std::string> &allowed)
{
    for (const auto &entry: node) {
        require(allowed.count(entry.first), "Unknown T-matrix XML element: " + entry.first);
    }
}

void attributes(const ptree &node, const std::set<std::string> &allowed)
{
    if (const auto attrs = node.get_child_optional("<xmlattr>")) {
        for (const auto &entry: *attrs) {
            require(allowed.count(entry.first), "Unknown T-matrix XML attribute: " + entry.first);
        }
    }
}

template<typename T, size_t N>
std::array<T, N> numbers(const std::string &text)
{
    std::istringstream input(text);
    std::array<T, N> values{};
    for (auto &value: values) require(bool(input >> value), "Invalid T-matrix numeric list: " + text);
    std::string extra;
    require(!(input >> extra), "Extra T-matrix numeric fields: " + text);
    return values;
}

[[noreturn]] void abort_tmatrix(const std::string &message, const int rank, MPI_Comm comm)
{
    std::cerr << "ERROR in Tmatrix (rank " << rank << "): " << message << std::endl;
    MPI_Abort(comm, EXIT_FAILURE);
    std::abort();
}
} // namespace

Tmatrix::Tmatrix(const RunInfo &run_in, const System *system_in, const Symmetry *symmetry_in,
                 const Kpoint *kpoint_in, const Dos *dos_in, const Dynamical *dynamical_in,
                 const Isotope *isotope_in) :
    run(run_in), system(system_in), symmetry(symmetry_in), kpoint(kpoint_in), dos(dos_in),
    dynamical(dynamical_in), isotope(isotope_in)
{}

void Tmatrix::setup()
{
    MPI_Bcast_string(file, 0, run.comm);
    MPI_Bcast(&epsilon_kayser, 1, MPI_DOUBLE, 0, run.comm);
    MPI_Bcast(&integration_method, 1, MPI_INT, 0, run.comm);
    std::string error;
    if (run.my_rank == 0) {
        try {
            if (!enabled()) {
                require(!epsilon_specified, "TMATRIX_EPSILON requires TMATRIX_FILE");
                require(integration_method == 0, "TMATRIX_INTEGRATION requires TMATRIX_FILE");
            } else {
                require(run.mode == "KAPPA" && kpoint->kpoint_mode == 2,
                        "TMATRIX_FILE requires MODE=kappa, SOLVER=RTA and KPOINT mode 2");
                require(!isotope->include_isotope, "TMATRIX_FILE and ISOTOPE cannot be combined (mass double counting)");
                require(std::isfinite(epsilon_kayser) && epsilon_kayser > 0.0,
                        "TMATRIX_FILE requires a finite, positive TMATRIX_EPSILON [cm^-1]");
                require(integration_method == 0 || integration_method == 1, "TMATRIX_INTEGRATION must be 0 or 1");
                read_models();
            }
        } catch (const std::exception &e) {
            error = e.what();
        }
    }
    MPI_Bcast_string(error, 0, run.comm);
    if (!error.empty()) abort_tmatrix(error, run.my_rank, run.comm);
    if (!enabled()) return;
    eta = epsilon_kayser * time_ry / Hz_to_kayser;
    try {
        broadcast_models();
    } catch (const std::exception &e) {
        abort_tmatrix(e.what(), run.my_rank, run.comm);
    }
}

void Tmatrix::read_models()
{
    models.clear();
    const auto &primitive = system->get_primcell();
    const auto &masses = system->get_mass_prim();
    ptree tree;
    boost::property_tree::read_xml(file, tree, boost::property_tree::xml_parser::trim_whitespace
                                             | boost::property_tree::xml_parser::no_comments);
    children(tree, {"Data"});
    require(tree.count("Data") == 1 && tree.get_child("Data").count("Tmatrix") == 1,
            "Expected a single Data.Tmatrix XML element");
    children(tree.get_child("Data"), {"Tmatrix"});
    const auto &root = tree.get_child("Data.Tmatrix");
    children(root, {"<xmlattr>", "Host", "Defects"});
    attributes(root, {"version", "fc2_unit", "mass_unit", "density_unit"});
    require(root.count("Host") == 1 && root.count("Defects") == 1, "Expected one Host and one Defects element");
    require(root.get<int>("<xmlattr>.version") == 1, "Unsupported T-matrix XML version");
    require(root.get<std::string>("<xmlattr>.fc2_unit") == "Ry/Bohr^2"
            && root.get<std::string>("<xmlattr>.mass_unit") == "amu"
            && root.get<std::string>("<xmlattr>.density_unit") == "defects/primitive_cell",
            "T-matrix XML units must be Ry/Bohr^2, amu, defects/primitive_cell");
    const auto &host = root.get_child("Host");
    children(host, {"<xmlattr>", "LatticeVector", "Basis"});
    attributes(host, {"natmin", "fcsxml", "fc2xml"});
    require(host.count("LatticeVector") == 1 && host.count("Basis") == 1, "Expected one host lattice and basis");
    require(host.get<unsigned int>("<xmlattr>.natmin") == primitive.number_of_atoms, "T-matrix host natmin mismatch");
    children(host.get_child("LatticeVector"), {"a1", "a2", "a3"});
    for (int j = 0; j < 3; ++j) {
        require(host.get_child("LatticeVector").count("a" + std::to_string(j + 1)) == 1,
                "Duplicate/missing host lattice vector");
        const auto a = numbers<double, 3>(host.get<std::string>("LatticeVector.a" + std::to_string(j + 1)));
        for (int i = 0; i < 3; ++i) {
            require(std::isfinite(a[i]) && std::abs(a[i] - primitive.lattice_vector(i, j))
                    <= 1.0e-10 + symmetry->tolerance * std::max(1.0, std::abs(primitive.lattice_vector(i, j))),
                    "T-matrix host lattice mismatch (vectors must be in Bohr)");
        }
    }
    const auto &basis = host.get_child("Basis");
    children(basis, {"Atom"});
    std::set<int> basis_seen;
    for (const auto &entry: basis) {
        const auto &atom = entry.second;
        attributes(atom, {"index", "kind", "fractional", "mass"});
        children(atom, {"<xmlattr>"});
        const auto p = atom.get<int>("<xmlattr>.index") - 1;
        require(p >= 0 && p < primitive.number_of_atoms && basis_seen.insert(p).second, "Invalid/duplicate T-matrix host atom");
        const auto s = p;
        require(atom.get<unsigned int>("<xmlattr>.kind") == primitive.kind[s] + 1, "T-matrix host kind mismatch");
        const auto mass = atom.get<double>("<xmlattr>.mass");
        require(std::isfinite(mass) && mass > 0.0
                && std::abs(mass * amu_ry - masses[s]) <= 1.0e-8 * masses[s],
                "T-matrix host mass mismatch (check MASS)");
        const auto x = numbers<double, 3>(atom.get<std::string>("<xmlattr>.fractional"));
        for (int i = 0; i < 3; ++i) {
            const auto diff = x[i] - primitive.x_fractional(s, i);
            require(std::isfinite(x[i]) && std::abs(diff - std::round(diff)) <= symmetry->tolerance,
                    "T-matrix host basis position mismatch");
        }
    }
    require(basis_seen.size() == primitive.number_of_atoms, "Incomplete T-matrix host basis");

    const auto &operations = symmetry->SymmListWithMap;
    require(!operations.empty(), "Missing host symmetry mappings");
    const auto mesh = dos->kmesh_dos.get();
    for (const auto &op: operations) {
        for (unsigned int iq = 0; iq < mesh->nk; ++iq) {
            double q[3]{};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) q[i] += op.rot_reciprocal[3 * i + j] * mesh->xk[iq][j];
                q[i] -= std::round(q[i]);
            }
            require(mesh->get_knum(q) >= 0, "T-matrix q mesh is not closed under host symmetry");
        }
    }

    const auto &defects = root.get_child("Defects");
    children(defects, {"Defect"});
    std::set<std::string> names;
    for (const auto &entry: defects) {
        const auto &node = entry.second;
        attributes(node, {"name", "density"});
        children(node, {"<xmlattr>", "Sites", "DeltaFC2"});
        require(node.count("Sites") == 1 && node.count("DeltaFC2") == 1, "Expected one Sites and one DeltaFC2 element");
        Defect original;
        original.name = node.get<std::string>("<xmlattr>.name");
        original.density = node.get<double>("<xmlattr>.density");
        require(!original.name.empty() && names.insert(original.name).second, "Empty/duplicate T-matrix defect name");
        require(std::isfinite(original.density) && original.density >= 0.0, "Invalid T-matrix defect density");
        const auto &sites = node.get_child("Sites");
        children(sites, {"Site"});
        require(!sites.empty() && sites.size() <= INT_MAX / 3, "Invalid T-matrix cluster size");
        const auto n = static_cast<int>(sites.size());
        const auto d = 3 * n;
        require(static_cast<long long>(d) * d <= INT_MAX, "T-matrix cluster exceeds MPI matrix count limits");
        original.sites.resize(n);
        original.mass_ratio.resize(n);
        std::set<int> site_seen;
        std::set<std::array<int, 4>> locations;
        for (const auto &site_entry: sites) {
            const auto &site = site_entry.second;
            attributes(site, {"index", "primitive_atom", "cell", "mass"});
            children(site, {"<xmlattr>"});
            const auto index = site.get<int>("<xmlattr>.index") - 1;
            const auto atom = site.get<int>("<xmlattr>.primitive_atom") - 1;
            require(index >= 0 && index < n && site_seen.insert(index).second, "Invalid/duplicate cluster site index");
            require(atom >= 0 && atom < primitive.number_of_atoms, "Invalid cluster primitive_atom");
            const auto cell = numbers<int, 3>(site.get<std::string>("<xmlattr>.cell"));
            require(locations.insert({atom, cell[0], cell[1], cell[2]}).second, "Duplicate cluster lattice site");
            original.sites[index] = {atom, cell};
            const auto m0 = masses[atom] / amu_ry;
            const auto mass = site.get<double>("<xmlattr>.mass", m0);
            require(std::isfinite(mass) && mass > 0.0, "Cluster masses must be finite and positive");
            original.mass_ratio(index) = (mass - m0) / m0;
        }
        Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(d, d);
        const auto &fc2 = node.get_child("DeltaFC2");
        children(fc2, {"FC2"});
        std::set<std::pair<int, int>> pairs;
        for (const auto &fc_entry: fc2) {
            const auto &fc = fc_entry.second;
            attributes(fc, {"pair1", "pair2"});
            children(fc, {"<xmlattr>"});
            const auto p1 = numbers<int, 2>(fc.get<std::string>("<xmlattr>.pair1"));
            const auto p2 = numbers<int, 2>(fc.get<std::string>("<xmlattr>.pair2"));
            require(p1[0] >= 1 && p1[0] <= n && p2[0] >= 1 && p2[0] <= n
                    && p1[1] >= 1 && p1[1] <= 3 && p2[1] >= 1 && p2[1] <= 3,
                    "DeltaFC2 pair index out of range");
            const auto i = 3 * (p1[0] - 1) + p1[1] - 1;
            const auto j = 3 * (p2[0] - 1) + p2[1] - 1;
            require(pairs.insert({i, j}).second, "Duplicate DeltaFC2 component");
            phi(i, j) = fc.get_value<double>();
            require(std::isfinite(phi(i, j)), "Nonfinite DeltaFC2 component");
        }
        const auto scale = phi.cwiseAbs().rowwise().sum().maxCoeff();
        const auto transpose_error = (phi - phi.transpose()).cwiseAbs().maxCoeff();
        require(transpose_error <= 1.0e-12 + 1.0e-8 * scale, "DeltaFC2 is not symmetric");
        double asr_error = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int beta = 0; beta < 3; ++beta) {
                double sum = 0.0, row_scale = 0.0;
                for (int b = 0; b < n; ++b) {
                    sum += phi(i, 3 * b + beta);
                    row_scale += std::abs(phi(i, 3 * b + beta));
                }
                asr_error = std::max(asr_error, std::abs(sum));
                require(std::abs(sum) <= 1.0e-12 + 1.0e-8 * row_scale,
                        "DeltaFC2 violates the acoustic sum rule (include complete affected bonds)");
            }
        }
        original.k = phi;
        for (int i = 0; i < d; ++i) {
            const auto mi = masses[original.sites[i / 3].atom];
            for (int j = 0; j < d; ++j) {
                const auto mj = masses[original.sites[j / 3].atom];
                original.k(i, j) /= std::sqrt(mi * mj);
            }
        }
        std::ostringstream info;
        info << std::scientific << std::setprecision(8)
                  << " Tmatrix defect " << original.name << ": density=" << original.density
                  << " defects/primitive_cell, sites=" << n << ", orientations=" << operations.size()
                  << ", transpose/ASR residual=" << transpose_error << "/" << asr_error << " Ry/Bohr^2\n";
        std::cout << info.str();
        if (original.density == 0.0) continue;
        for (const auto &op: operations) {
            Defect model = original;
            model.density /= static_cast<double>(operations.size());
            Eigen::Matrix3d rotation;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) rotation(i, j) = op.rot[3 * i + j];
            for (int a = 0; a < n; ++a) {
                const auto atom = original.sites[a].atom;
                const auto mapped = op.mapping[atom];
                model.sites[a].atom = mapped;
                const auto sa = atom;
                const auto sm = mapped;
                for (int i = 0; i < 3; ++i) {
                    auto x = op.shift[i] - primitive.x_fractional(sm, i);
                    for (int j = 0; j < 3; ++j) x += op.rot_real[3 * i + j]
                            * (primitive.x_fractional(sa, j) + original.sites[a].cell[j]);
                    require(std::isfinite(x) && std::abs(x - std::round(x)) <= symmetry->tolerance
                            && std::abs(x) < INT_MAX / 2, "Cannot map T-matrix site under host symmetry");
                    model.sites[a].cell[i] = static_cast<int>(std::llround(x));
                }
                for (int b = 0; b < n; ++b) {
                    model.k.block<3, 3>(3 * a, 3 * b) = rotation
                        * original.k.block<3, 3>(3 * a, 3 * b) * rotation.transpose();
                }
            }
            // A common integer translation is a global Bloch phase and cancels in the width.
            const auto origin = model.sites.front().cell;
            for (auto &site: model.sites) for (int i = 0; i < 3; ++i) site.cell[i] -= origin[i];
            const auto identical = std::find_if(models.begin(), models.end(), [&](const Defect &other) {
                return other.name == model.name && other.sites == model.sites
                       && (other.k.array() == model.k.array()).all()
                       && (other.mass_ratio.array() == model.mass_ratio.array()).all();
            });
            if (identical == models.end()) models.push_back(std::move(model));
            else identical->density += model.density;
        }
    }
    require(!names.empty(), "T-matrix XML contains no defects");
    std::cout << " Tmatrix distinct clusters after exact translation-equivalent merging: " << models.size() << '\n';
}

void Tmatrix::broadcast_models()
{
    int count = static_cast<int>(models.size());
    MPI_Bcast(&count, 1, MPI_INT, 0, run.comm);
    if (run.my_rank) models.resize(count);
    for (auto &model: models) {
        MPI_Bcast_string(model.name, 0, run.comm);
        int n = static_cast<int>(model.sites.size());
        MPI_Bcast(&n, 1, MPI_INT, 0, run.comm);
        MPI_Bcast(&model.density, 1, MPI_DOUBLE, 0, run.comm);
        std::vector<int> sites(4 * n);
        if (run.my_rank == 0) {
            for (int i = 0; i < n; ++i) {
                sites[4 * i] = model.sites[i].atom;
                for (int j = 0; j < 3; ++j) sites[4 * i + 1 + j] = model.sites[i].cell[j];
            }
        } else {
            model.sites.resize(n);
            model.k.resize(3 * n, 3 * n);
            model.mass_ratio.resize(n);
        }
        MPI_Bcast(sites.data(), 4 * n, MPI_INT, 0, run.comm);
        if (run.my_rank) {
            for (int i = 0; i < n; ++i) {
                model.sites[i].atom = sites[4 * i];
                for (int j = 0; j < 3; ++j) model.sites[i].cell[j] = sites[4 * i + 1 + j];
            }
        }
        MPI_Bcast(model.k.data(), 9 * n * n, MPI_DOUBLE, 0, run.comm);
        MPI_Bcast(model.mass_ratio.data(), n, MPI_DOUBLE, 0, run.comm);
    }
}

void Tmatrix::amplitudes(const Defect &model, const unsigned int iq, Eigen::MatrixXcd &b) const
{
    const auto mesh = dos->kmesh_dos.get();
    const auto evec = dos->dymat_dos->get_eigenvectors();
    for (size_t a = 0; a < model.sites.size(); ++a) {
        double phase = 0.0;
        for (int j = 0; j < 3; ++j) phase += mesh->xk[iq][j] * model.sites[a].cell[j];
        const auto factor = std::exp(im * (2.0 * pi * phase));
        for (unsigned int s = 0; s < dynamical->neval; ++s) {
            for (int alpha = 0; alpha < 3; ++alpha) {
                b(3 * a + alpha, s) = factor * evec[iq][s][3 * model.sites[a].atom + alpha];
            }
        }
    }
}

void Tmatrix::calculate()
{
    if (!enabled()) return;
    const auto mesh = dos->kmesh_dos.get();
    const auto ns = dynamical->neval;
    const auto eval = dos->dymat_dos->get_eigenvalues();
    const auto nks_size = static_cast<size_t>(mesh->nk_irred) * ns;
    if (nks_size > INT_MAX) abort_tmatrix("T-matrix mode count exceeds MPI limits", run.my_rank, run.comm);
    const auto nks = static_cast<int>(nks_size);
    for (unsigned int iq = 0; iq < mesh->nk; ++iq) {
        for (unsigned int s = 0; s < ns; ++s) {
            if (!std::isfinite(eval[iq][s]) || eval[iq][s] < 0.0) {
                abort_tmatrix("Unstable/nonfinite host frequency at full q=" + std::to_string(iq + 1)
                              + ", branch=" + std::to_string(s + 1), run.my_rank, run.comm);
            }
        }
    }
    std::vector<double> local(nks, 0.0), total(nks, 0.0), local_shift(nks, 0.0), total_shift(nks, 0.0);
    double max_residual = 0.0, min_rcond = 1.0;
    std::string error, context;
    try {
        std::unique_ptr<TetraNodes> tetra;
        std::vector<std::complex<double>> weights;
        if (integration_method == 1) {
            tetra.reset(new TetraNodes(mesh->nk_i[0], mesh->nk_i[1], mesh->nk_i[2]));
            tetra->setup();
            weights.resize(static_cast<size_t>(mesh->nk) * ns);
        }
        for (int job = run.my_rank; job < nks; job += run.nprocs) {
            const auto iq = mesh->kpoint_irred_all[job / ns][0].knum;
            const auto s = job % ns;
            const auto omega = eval[iq][s];
            if (omega <= eps15) continue;
            if (tetra) {
                std::fill(weights.begin(), weights.end(), 0.0);
                const auto nodes = tetra->get_tetras();
                const auto ntetra = tetra->get_ntetra();
                const auto z = std::complex<double>(omega, eta) * std::complex<double>(omega, eta);
                for (unsigned int it = 0; it < ntetra; ++it) {
                    for (unsigned int branch = 0; branch < ns; ++branch) {
                        std::array<double, 4> energy;
                        for (int a = 0; a < 4; ++a) energy[a] = std::pow(eval[nodes[it][a]][branch], 2);
                        const auto integral = tmatrix_kernel::tetrahedron_resolvent(energy, z);
                        for (int a = 0; a < 4; ++a)
                            weights[static_cast<size_t>(nodes[it][a]) * ns + branch] += integral[a] / static_cast<double>(ntetra);
                    }
                }
                for (unsigned int jq = 0; jq < mesh->nk; ++jq)
                    tmatrix_kernel::average_degenerate_weights(eval[jq], ns,
                                                               weights.data() + static_cast<size_t>(jq) * ns);
            }
            for (const auto &model: models) {
                if (model.k.isZero(0.0) && model.mass_ratio.isZero(0.0)) continue;
                const auto d = static_cast<int>(3 * model.sites.size());
                Eigen::MatrixXcd b(d, ns), green = Eigen::MatrixXcd::Zero(d, d);
                context = "defect=" + model.name + ", irreducible_q=" + std::to_string(job / ns + 1)
                          + ", branch=" + std::to_string(s + 1);
                for (unsigned int jq = 0; jq < mesh->nk; ++jq) {
                    amplitudes(model, jq, b);
                    if (tetra) {
                        for (unsigned int branch = 0; branch < ns; ++branch)
                            green.noalias() += (b.col(branch) * b.col(branch).adjoint())
                                              * weights[static_cast<size_t>(jq) * ns + branch];
                    } else {
                        tmatrix_kernel::accumulate_green(green, b, eval[jq], omega, eta);
                    }
                }
                if (!tetra) green /= static_cast<double>(mesh->nk);
                Eigen::MatrixXcd v = model.k.cast<std::complex<double>>();
                for (int a = 0; a < d; ++a) v(a, a) -= omega * omega * model.mass_ratio(a / 3);
                const auto solution = tmatrix_kernel::solve(v, green);
                amplitudes(model, iq, b);
                const auto projected = b.col(s).dot(solution.t * b.col(s));
                const auto width = tmatrix_kernel::halfwidth(projected, omega);
                local[job] += model.density * width;
                local_shift[job] += model.density * projected.real() / (2.0 * omega);
                max_residual = std::max(max_residual, solution.residual);
                min_rcond = std::min(min_rcond, solution.rcond);
            }
        }
    } catch (const std::exception &e) {
        error = context + ": " + e.what();
    }
    int failed = !error.empty(), any_failed;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, run.comm);
    if (any_failed) abort_tmatrix(error.empty() ? "A peer failed during T-matrix calculation" : error, run.my_rank, run.comm);
    MPI_Reduce(local.data(), total.data(), nks, MPI_DOUBLE, MPI_SUM, 0, run.comm);
    MPI_Reduce(local_shift.data(), total_shift.data(), nks, MPI_DOUBLE, MPI_SUM, 0, run.comm);
    double diagnostics[2]{max_residual, -min_rcond};
    double global[2];
    MPI_Reduce(diagnostics, global, 2, MPI_DOUBLE, MPI_MAX, 0, run.comm);
    if (run.my_rank == 0) {
        gamma_tmatrix.resize(mesh->nk_irred, ns);
        double max_width_ratio = 0.0, max_shift_ratio = 0.0;
        for (int job = 0; job < nks; ++job) {
            gamma_tmatrix[job / ns][job % ns] = total[job];
            const auto omega = eval[mesh->kpoint_irred_all[job / ns][0].knum][job % ns];
            if (omega > eps15) {
                max_width_ratio = std::max(max_width_ratio, total[job] / omega);
                max_shift_ratio = std::max(max_shift_ratio, std::abs(total_shift[job]) / omega);
            }
        }
        std::ostringstream info;
        info << std::scientific << std::setprecision(8)
                  << " Tmatrix: max backward residual=" << global[0] << ", min rcond=" << -global[1]
                  << ", max total gamma/omega=" << max_width_ratio
                  << ", max total |shift|/omega=" << max_shift_ratio << '\n';
        std::cout << info.str();
        if (-global[1] < 1.0e-12) std::cout << " WARNING: ill-conditioned T matrix; converge TMATRIX_EPSILON and q mesh.\n";
        std::cout << " Tmatrix uses independent defects and a fixed host; no selfconsistent frequency shift.\n";
    }
}

void Tmatrix::write_selfenergy() const
{
    if (!enabled() || run.my_rank) return;
    std::ofstream out(run.job_title + ".self_tmatrix");
    if (!out) abort_tmatrix("Cannot open .self_tmatrix output", run.my_rank, run.comm);
    out << "# Finite-cluster T-matrix HWHM, independent defects, degeneracy averaged\n"
        << "# Irred. knum, mode num, frequency [cm^-1], Gamma_T [cm^-1]\n\n";
    const auto mesh = dos->kmesh_dos.get();
    const auto eval = dos->dymat_dos->get_eigenvalues();
    const auto factor = Hz_to_kayser / time_ry;
    out << std::scientific << std::setprecision(15);
    for (unsigned int i = 0; i < mesh->nk_irred; ++i) {
        const auto iq = mesh->kpoint_irred_all[i][0].knum;
        out << "# Irreducible k point : " << i + 1 << " (" << mesh->kpoint_irred_all[i].size() << ")\n## xk =";
        for (int j = 0; j < 3; ++j) out << ' ' << mesh->xk[iq][j];
        out << '\n';
        for (unsigned int s = 0; s < dynamical->neval; ++s) {
            out << i + 1 << ' ' << s + 1 << ' ' << eval[iq][s] * factor << ' '
                << gamma_tmatrix[i][s] * factor << '\n';
        }
        out << '\n';
    }
    if (!out) abort_tmatrix("Failed writing .self_tmatrix output", run.my_rank, run.comm);
}
