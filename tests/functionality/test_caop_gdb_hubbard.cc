#include "sbd/sbd.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr std::size_t bit_length = 64;

struct BasisLayout {
  sbd::det_vector<std::size_t> determinants;
  std::vector<std::size_t> logical_keys;
  std::vector<double> phases;
};

int population(std::size_t bits) {
  return __builtin_popcountll(static_cast<unsigned long long>(bits));
}

double block_to_interleaved_phase(std::size_t alpha, std::size_t beta,
                                  int sites) {
  int inversions = 0;
  for(int beta_site = 0; beta_site < sites; ++beta_site) {
    if(!(beta & (std::size_t{1} << beta_site))) continue;
    for(int alpha_site = beta_site + 1; alpha_site < sites; ++alpha_site)
      if(alpha & (std::size_t{1} << alpha_site)) ++inversions;
  }
  return inversions % 2 == 0 ? 1.0 : -1.0;
}

BasisLayout make_basis(int sites, int particles_per_spin, bool interleaved) {
  using Entry = std::tuple<std::size_t, std::size_t, double>;
  std::vector<Entry> entries;
  const std::size_t states = std::size_t{1} << sites;
  for(std::size_t beta = 0; beta < states; ++beta) {
    if(population(beta) != particles_per_spin) continue;
    for(std::size_t alpha = 0; alpha < states; ++alpha) {
      if(population(alpha) != particles_per_spin) continue;
      std::size_t encoded = 0;
      for(int site = 0; site < sites; ++site) {
        if(alpha & (std::size_t{1} << site))
          encoded |= std::size_t{1} << (interleaved ? 2 * site : site);
        if(beta & (std::size_t{1} << site))
          encoded |= std::size_t{1} <<
                     (interleaved ? 2 * site + 1 : sites + site);
      }
      const std::size_t key = alpha | (beta << sites);
      entries.emplace_back(
          encoded, key, block_to_interleaved_phase(alpha, beta, sites));
    }
  }
  std::sort(entries.begin(), entries.end());

  BasisLayout result;
  for(const auto& [encoded, key, phase] : entries) {
    result.determinants.push_back(std::vector<std::size_t>{encoded});
    result.logical_keys.push_back(key);
    result.phases.push_back(phase);
  }
  return result;
}

std::vector<std::size_t> positions_by_key(const BasisLayout& basis,
                                          int sites) {
  std::vector<std::size_t> positions(
      std::size_t{1} << (2 * sites), basis.logical_keys.size());
  for(std::size_t index = 0; index < basis.logical_keys.size(); ++index)
    positions[basis.logical_keys[index]] = index;
  return positions;
}

std::size_t determinant_key(const sbd::det_vector<std::size_t>::row& det,
                            int sites, bool interleaved) {
  std::size_t alpha = 0;
  std::size_t beta = 0;
  for(int site = 0; site < sites; ++site) {
    const int alpha_bit = interleaved ? 2 * site : site;
    const int beta_bit = interleaved ? 2 * site + 1 : sites + site;
    if(det[0] & (std::size_t{1} << alpha_bit))
      alpha |= std::size_t{1} << site;
    if(det[0] & (std::size_t{1} << beta_bit))
      beta |= std::size_t{1} << site;
  }
  return alpha | (beta << sites);
}

std::vector<std::size_t> candidate_keys(
    const sbd::det_vector<std::size_t>& candidates,
    int sites, bool interleaved) {
  std::vector<std::size_t> keys;
  keys.reserve(candidates.size());
  for(const auto& determinant : candidates)
    keys.push_back(determinant_key(determinant, sites, interleaved));
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

void compare_expansion(
    int sites, const BasisLayout& gdb_basis,
    const BasisLayout& ca_op_basis,
    const sbd::oneInt<double>& one_body,
    const sbd::twoInt<double>& two_body,
    double scalar, const sbd::GeneralOp<double>& hamiltonian,
    MPI_Comm comm) {
  const auto gdb_positions = positions_by_key(gdb_basis, sites);
  const auto ca_op_positions = positions_by_key(ca_op_basis, sites);
  const std::size_t parent_count = sites == 4 ? 9 : 41;
  const std::size_t stride = sites == 4 ? 5 : 37;
  sbd::det_vector<std::size_t> gdb_parents;
  sbd::det_vector<std::size_t> ca_op_parents;
  std::vector<double> coefficients;
  for(std::size_t parent = 0; parent < parent_count; ++parent) {
    const std::size_t full_index =
        (parent * stride) % gdb_basis.logical_keys.size();
    const std::size_t key = gdb_basis.logical_keys[full_index];
    gdb_parents.push_back(gdb_basis.determinants[gdb_positions[key]]);
    ca_op_parents.push_back(ca_op_basis.determinants[ca_op_positions[key]]);
    const double magnitude =
        0.15 + 0.8 * static_cast<double>(parent + 1) /
                   static_cast<double>(parent_count + 1);
    coefficients.push_back(parent % 2 == 0 ? magnitude : -magnitude);
  }

  const auto flat_data = sbd::MakeKetSideGeneralOpFlatData<double>(
      hamiltonian, 1, bit_length,
      [](double coefficient) { return std::abs(coefficient); });
  for(const double cutoff : {0.2, 0.6}) {
    sbd::det_vector<std::size_t> ca_op_candidates;
    sbd::caop::HeatbathExpansion(
        ca_op_parents, coefficients, flat_data, cutoff, 7,
        ca_op_candidates, comm, comm, comm);
    const auto ca_op_keys = candidate_keys(ca_op_candidates, sites, false);
    for(int gdb_type = 0; gdb_type <= 1; ++gdb_type) {
      sbd::det_vector<std::size_t> gdb_candidates;
      sbd::gdb::HeatbathExpansion(
          gdb_parents, coefficients, bit_length,
          static_cast<std::size_t>(sites), scalar, one_body, two_body,
          gdb_type, cutoff, 7, gdb_candidates, comm, comm);
      const auto gdb_keys = candidate_keys(gdb_candidates, sites, true);
      if(gdb_keys != ca_op_keys)
        throw std::runtime_error(
            "CAOP and GDB heatbath candidates differ at cutoff " +
            std::to_string(cutoff));
      std::cout << "CAOP_GDB_HUBBARD_EXPANSION_TEST PASS sites=" << sites
                << " parents=" << parent_count
                << " cutoff=" << cutoff
                << " gdb_type=" << gdb_type
                << " candidates=" << gdb_keys.size() << std::endl;
    }
  }
}

void compare_model(const std::string& filename, int sites,
                   int particles_per_spin, MPI_Comm comm) {
  const BasisLayout gdb_basis = make_basis(sites, particles_per_spin, true);
  const BasisLayout ca_op_basis = make_basis(sites, particles_per_spin, false);
  const auto ca_op_positions = positions_by_key(ca_op_basis, sites);

  std::vector<double> gdb_input(gdb_basis.logical_keys.size());
  std::vector<double> ca_op_input(ca_op_basis.logical_keys.size());
  for(std::size_t index = 0; index < gdb_input.size(); ++index) {
    const std::size_t key = gdb_basis.logical_keys[index];
    const double value = std::sin(0.37 * static_cast<double>(key + 1)) +
                         0.25 * std::cos(
                             0.19 * static_cast<double>(index + 1));
    gdb_input[index] = value;
    const std::size_t ca_op_index = ca_op_positions[key];
    ca_op_input[ca_op_index] = ca_op_basis.phases[ca_op_index] * value;
  }

  const auto fcidump = sbd::LoadFCIDump(filename);
  int norb = 0;
  int nelec = 0;
  double scalar = 0.0;
  sbd::oneInt<double> one_body;
  sbd::twoInt<double> two_body;
  sbd::SetupIntegrals(fcidump, norb, nelec, scalar, one_body, two_body);

  sbd::gdb::DetIndexMap index_map;
  std::vector<sbd::gdb::ExcitationLookup> excitation_lookup;
  sbd::gdb::MakeHelpers(gdb_basis.determinants, bit_length, norb, index_map,
                        excitation_lookup, comm, comm, comm);
  std::vector<double> gdb_diagonal;
  sbd::gdb::makeQChamDiagTerms(
      gdb_basis.determinants, bit_length, norb, index_map, excitation_lookup,
      scalar, one_body, two_body, gdb_diagonal, comm, comm, comm);
  std::vector<double> gdb_result(gdb_input.size(), 0.0);
  sbd::gdb::mult(gdb_diagonal, gdb_input, gdb_result, bit_length, norb,
                 gdb_basis.determinants, index_map, excitation_lookup,
                 scalar, one_body, two_body, comm, comm, comm);

  int ca_op_norb = 0;
  int ca_op_nelec = 0;
  sbd::GeneralOp<double> hamiltonian;
  sbd::GeneralOp_From_FCIDump(
      filename, comm, comm, ca_op_norb, ca_op_nelec, hamiltonian);
  const std::vector<int> slide{0};
  std::vector<double> ca_op_diagonal;
  sbd::makeCAOpHamDiagTerms(ca_op_basis.determinants, bit_length, slide,
                            hamiltonian, ca_op_diagonal);
  std::vector<double> ca_op_result(ca_op_input.size(), 0.0);
  sbd::mult(ca_op_diagonal, ca_op_input, ca_op_result,
            ca_op_basis.determinants, bit_length, slide, hamiltonian, true,
            comm, comm, comm);
  std::vector<double> ca_op_without_sign(ca_op_input.size(), 0.0);
  sbd::mult(ca_op_diagonal, ca_op_input, ca_op_without_sign,
            ca_op_basis.determinants, bit_length, slide, hamiltonian, false,
            comm, comm, comm);

  double maximum_error = 0.0;
  double missing_sign_error = 0.0;
  for(std::size_t gdb_index = 0; gdb_index < gdb_result.size(); ++gdb_index) {
    const std::size_t key = gdb_basis.logical_keys[gdb_index];
    const std::size_t ca_op_index = ca_op_positions[key];
    const double phase = ca_op_basis.phases[ca_op_index];
    maximum_error = std::max(
        maximum_error,
        std::abs(phase * ca_op_result[ca_op_index] - gdb_result[gdb_index]));
    missing_sign_error = std::max(
        missing_sign_error,
        std::abs(phase * ca_op_without_sign[ca_op_index] -
                 gdb_result[gdb_index]));
  }
  if(missing_sign_error < 1.0e-6 || maximum_error > 1.0e-11)
    throw std::runtime_error(
        "CAOP and GDB fermionic mult results differ for " + filename);

  std::cout << "CAOP_GDB_HUBBARD_FERMION_MULT_TEST PASS sites=" << sites
            << " basis=" << gdb_result.size()
            << " max_error=" << maximum_error
            << " no_sign_error=" << missing_sign_error << std::endl;

  compare_expansion(sites, gdb_basis, ca_op_basis, one_body, two_body,
                    scalar, hamiltonian, comm);
}

void test_caop_matches_gdb(MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  if(size != 1) return;

  sbd::det_vector<std::size_t>::init_elem_size(1);
  sbd::det_vector<std::size_t, sbd::det_kind::half>::init_elem_size(1);
  compare_model("caop_gdb_hubbard4_fcidump.txt", 4, 2, comm);
  compare_model("caop_gdb_hubbard8_open_fcidump.txt", 8, 4, comm);
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    test_caop_matches_gdb(MPI_COMM_WORLD);
  } catch(const std::exception& error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) std::cerr << error.what() << std::endl;
    result = 1;
  }
  MPI_Finalize();
  return result;
}
