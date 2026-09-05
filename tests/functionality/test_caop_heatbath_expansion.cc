#include "sbd/caop/basic/expansion.h"
#include "sbd/caop/basic/arithmetic.h"
#include "sbd/caop/basic/helper.h"
#include "sbd/caop/basic/loadmodel.h"
#include "sbd/framework/determinant_distribution_round_robin.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Geometry {
  int h = 1;
  int b = 1;
  int t = 1;
  int distribution = 0;
};

Geometry parse_geometry(int argc, char** argv) {
  Geometry geometry;
  for(int i = 1; i + 1 < argc; ++i) {
    const std::string argument(argv[i]);
    if(argument == "--h") geometry.h = std::stoi(argv[++i]);
    else if(argument == "--b") geometry.b = std::stoi(argv[++i]);
    else if(argument == "--t") geometry.t = std::stoi(argv[++i]);
    else if(argument == "--distribution")
      geometry.distribution = std::stoi(argv[++i]);
  }
  return geometry;
}

std::vector<sbd::GeneralOp<double>> make_heisenberg_terms() {
  constexpr int sites = 8;
  std::vector<sbd::GeneralOp<double>> terms;
  for(int left = 0; left < sites; ++left) {
    const int right = (left + 1) % sites;
    terms.push_back(0.5 * sbd::Cr(left) * sbd::An(right));
    terms.push_back(0.5 * sbd::Cr(right) * sbd::An(left));
  }
  return terms;
}

sbd::GeneralOp<double> distribute_terms(
    const std::vector<sbd::GeneralOp<double>>& terms,
    int h_rank, int h_size) {
  sbd::GeneralOp<double> result;
  for(std::size_t index = 0; index < terms.size(); ++index)
    if(static_cast<int>(index % static_cast<std::size_t>(h_size)) == h_rank)
      result += terms[index];
  return result;
}

void make_heisenberg_parents(std::vector<std::size_t>& parents,
                             std::vector<double>& coefficients) {
  std::vector<std::size_t> sector;
  for(std::size_t state = 0; state < 256; ++state)
    if(__builtin_popcountll(static_cast<unsigned long long>(state)) == 4)
      sector.push_back(state);
  std::mt19937_64 generator(20260903);
  std::shuffle(sector.begin(), sector.end(), generator);
  parents.assign(sector.begin(), sector.begin() + 16);
  std::uniform_real_distribution<double> distribution(0.15, 1.0);
  coefficients.resize(parents.size());
  for(double& coefficient : coefficients) coefficient = distribution(generator);
  const double norm = std::sqrt(std::inner_product(
      coefficients.begin(), coefficients.end(), coefficients.begin(), 0.0));
  for(double& coefficient : coefficients) coefficient /= norm;
}

std::vector<std::size_t> heisenberg_reference(
    const std::vector<std::size_t>& parents,
    const std::vector<double>& coefficients,
    double cutoff) {
  constexpr int sites = 8;
  std::vector<std::size_t> expected(parents);
  for(std::size_t parent_index = 0; parent_index < parents.size();
      ++parent_index) {
    const std::size_t parent = parents[parent_index];
    if(!(0.5 * std::abs(coefficients[parent_index]) > cutoff)) continue;
    for(int left = 0; left < sites; ++left) {
      const int right = (left + 1) % sites;
      const std::size_t left_mask = std::size_t(1) << left;
      const std::size_t right_mask = std::size_t(1) << right;
      if(((parent & left_mask) != 0) != ((parent & right_mask) != 0))
        expected.push_back(parent ^ left_mask ^ right_mask);
    }
  }
  std::sort(expected.begin(), expected.end());
  expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
  return expected;
}

void require_world(bool local_condition, const char* message, MPI_Comm comm) {
  int local_ok = local_condition ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
  if(!global_ok) throw std::runtime_error(message);
}

template <typename FlatCoeffT>
void test_flat_data_layout(
    const sbd::GeneralOpFlatData<FlatCoeffT>& flat_data,
    MPI_Comm world) {
  const std::size_t terms = flat_data.coefficients.size();
  bool valid = flat_data.creation_masks.size() == terms &&
               flat_data.annihilation_masks.size() == terms &&
               flat_data.creation_counts.size() == terms &&
               flat_data.offsets.size() == terms + 1 &&
               !flat_data.offsets.empty() && flat_data.offsets.front() == 0 &&
               flat_data.offsets.back() == flat_data.orbitals.size();
  for(std::size_t term = 0; valid && term < terms; ++term) {
    valid = flat_data.offsets[term] <= flat_data.offsets[term + 1] &&
            flat_data.creation_counts[term] <=
                flat_data.offsets[term + 1] - flat_data.offsets[term] &&
            (term == 0 ||
             std::abs(flat_data.coefficients[term - 1]) >=
                 std::abs(flat_data.coefficients[term]));
    std::vector<std::size_t> creation(
        flat_data.creation_masks[term].size(), 0);
    std::vector<std::size_t> annihilation(
        flat_data.annihilation_masks[term].size(), 0);
    const std::size_t middle =
        flat_data.offsets[term] + flat_data.creation_counts[term];
    for(std::size_t index = flat_data.offsets[term];
        index < flat_data.offsets[term + 1]; ++index) {
      const std::size_t site =
          static_cast<std::size_t>(flat_data.orbitals[index]);
      auto& mask = index < middle ? creation : annihilation;
      mask[site / 64] |= std::size_t(1) << (site % 64);
    }
    valid = valid && creation == flat_data.creation_masks[term] &&
            annihilation == flat_data.annihilation_masks[term];
  }
  require_world(valid, "GeneralOp flat data layout test failed", world);
}

void test_cutoff_boundary(int h_rank, int b_rank,
                          MPI_Comm h_comm, MPI_Comm t_comm,
                          MPI_Comm world) {
  sbd::det_vector<std::size_t> parents;
  std::vector<double> coefficients;
  if(b_rank == 0) {
    parents.resize(1);
    parents[0][0] = 1;
    coefficients.push_back(0.5);
  }
  sbd::GeneralOp<double> hamiltonian;
  if(h_rank == 0) hamiltonian += 0.5 * sbd::Cr(1) * sbd::An(0);
  const auto flat_data = sbd::MakeKetSideGeneralOpFlatData<double>(
      hamiltonian, 1, 64,
      [](double coefficient) { return std::abs(coefficient); });
  test_flat_data_layout(flat_data, world);

  const double equal = 0.25;
  const double cutoffs[] = {
      std::nextafter(equal, 0.0), equal,
      std::nextafter(equal, 1.0)};
  const std::size_t expected_counts[] = {2, 1, 1};
  for(std::size_t test = 0; test < 3; ++test) {
    sbd::det_vector<std::size_t> candidates;
    sbd::caop::HeatbathExpansionStats stats;
    sbd::caop::HeatbathExpansionProfile profile;
    sbd::caop::HeatbathExpansion(
        parents, coefficients, flat_data, cutoffs[test], 1, candidates,
        h_comm, t_comm, world, &stats, &profile);
    std::size_t local_count = candidates.size();
    std::size_t global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, SBD_MPI_SIZE_T,
                  MPI_SUM, world);
    bool valid = global_count == expected_counts[test] && stats.parents == 1 &&
                 stats.terms_visited == (test == 0 ? 1 : 0) &&
                 stats.mask_rejections == 0 &&
                 stats.accepted_before_unique == (test == 0 ? 1 : 0) &&
                 profile.generation_seconds >= 0.0 &&
                 profile.global_sort_unique_seconds >= 0.0 &&
                 profile.redistribution_seconds >= 0.0;
    for(const auto& candidate : candidates)
      valid = valid && (candidate[0] == 1 ||
                        (test == 0 && candidate[0] == 2));
    require_world(valid, "heatbath cutoff boundary test failed", world);
  }
}

void test_parent_truncation(MPI_Comm world) {
  sbd::det_vector<std::size_t> parents(3);
  parents[0][0] = 1;
  parents[1][0] = 2;
  parents[2][0] = 4;
  std::vector<double> coefficients = {1.0, 0.5, 0.25};
  sbd::caop::TruncateHeatbathParents(parents, coefficients, 0.25);
  require_world(parents.size() == 1 && coefficients.size() == 1 &&
                    parents[0][0] == 1 && coefficients[0] == 1.0,
                "heatbath parent truncation boundary test failed", world);
}

void test_complex_coefficient_lookup(MPI_Comm world) {
  using Complex = std::complex<double>;
  const Complex coefficient(-0.25, 0.75);
  sbd::GeneralOp<Complex> hamiltonian;
  hamiltonian += coefficient * sbd::Cr(1) * sbd::An(0);
  const auto flat_data = sbd::MakeKetSideGeneralOpFlatData<Complex>(
      hamiltonian, 1, 64,
      [](const Complex& value) { return value; });
  test_flat_data_layout(flat_data, world);
  require_world(flat_data.coefficients.size() == 1 &&
                    flat_data.coefficients[0] == coefficient,
                "complex GeneralOp flat coefficient was not preserved",
                world);
}

void test_round_robin_pairs(int b_rank, int b_size,
                            MPI_Comm b_comm, MPI_Comm world) {
  const std::vector<double> all_coefficients = {
      0.1, 0.7, 0.3, 0.6, 0.2, 0.5, 0.4};
  std::size_t begin = 0;
  std::size_t end = all_coefficients.size();
  sbd::get_mpi_range(b_size, b_rank, begin, end);
  sbd::det_vector<std::size_t> parents(end - begin);
  std::vector<double> coefficients(end - begin);
  for(std::size_t index = begin; index < end; ++index) {
    parents[index - begin][0] = index + 1;
    coefficients[index - begin] = all_coefficients[index];
  }
  sbd::redistribute_determinants_weight_round_robin(
      parents, coefficients, b_comm);

  std::vector<std::size_t> ranking(all_coefficients.size());
  std::iota(ranking.begin(), ranking.end(), 0);
  std::sort(ranking.begin(), ranking.end(), [&](std::size_t lhs,
                                                std::size_t rhs) {
    return all_coefficients[lhs] > all_coefficients[rhs];
  });
  bool valid = parents.size() == coefficients.size();
  for(std::size_t local = 0; local < parents.size(); ++local) {
    const std::size_t original = parents[local][0] - 1;
    const auto position = std::find(ranking.begin(), ranking.end(), original);
    valid = valid && original < all_coefficients.size() &&
            coefficients[local] == all_coefficients[original] &&
            static_cast<int>((position - ranking.begin()) % b_size) == b_rank;
  }
  std::size_t local_count = parents.size();
  std::size_t global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, SBD_MPI_SIZE_T,
                MPI_SUM, b_comm);
  valid = valid && global_count == all_coefficients.size();
  require_world(valid, "heatbath round-robin pair test failed", world);
}

void test_fermion_sign_broadcast(MPI_Comm h_comm, MPI_Comm b_comm,
                                 MPI_Comm t_comm, MPI_Comm world) {
  sbd::GeneralOp<double> hamiltonian;
  bool fermion_sign = false;
  sbd::load_GeneralOp_from_file(
      "caop_fermion_hamiltonian.txt", hamiltonian, fermion_sign,
      h_comm, b_comm, t_comm);
  const auto flat_data = sbd::MakeKetSideGeneralOpFlatData<double>(
      hamiltonian, 1, 64, [](double coefficient) { return coefficient; });
  const double local_coefficient =
      flat_data.coefficients.empty() ? 0.0 : flat_data.coefficients.front();
  double coefficient = 0.0;
  MPI_Allreduce(&local_coefficient, &coefficient, 1, MPI_DOUBLE,
                MPI_SUM, h_comm);
  require_world(fermion_sign && std::abs(coefficient + 1.0) < 1.0e-12,
                "CAOP fermion sign was not applied during normal ordering",
                world);
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  int result = 0;
  MPI_Comm h_comm = MPI_COMM_NULL;
  MPI_Comm b_comm = MPI_COMM_NULL;
  MPI_Comm t_comm = MPI_COMM_NULL;
  try {
    const Geometry geometry = parse_geometry(argc, argv);
    if(geometry.h * geometry.b * geometry.t != world_size)
      throw std::invalid_argument("MPI size must equal h*b*t");
    sbd::setup_communicator(world, geometry.h, geometry.b, geometry.t,
                            h_comm, b_comm, t_comm);
    int h_rank = 0;
    int b_rank = 0;
    int b_size = 1;
    MPI_Comm_rank(h_comm, &h_rank);
    MPI_Comm_rank(b_comm, &b_rank);
    MPI_Comm_size(b_comm, &b_size);

    if(geometry.distribution != 0 && geometry.distribution != 1)
      throw std::invalid_argument("distribution must be 0 or 1");
    sbd::det_vector<std::size_t>::init_elem_size(1);
    test_cutoff_boundary(h_rank, b_rank, h_comm, t_comm, world);
    test_parent_truncation(world);
    test_complex_coefficient_lookup(world);
    test_round_robin_pairs(b_rank, b_size, b_comm, world);
    test_fermion_sign_broadcast(h_comm, b_comm, t_comm, world);
    std::vector<std::size_t> all_parents;
    std::vector<double> all_coefficients;
    make_heisenberg_parents(all_parents, all_coefficients);
    std::size_t parent_begin = 0;
    std::size_t parent_end = all_parents.size();
    sbd::get_mpi_range(geometry.b, b_rank, parent_begin, parent_end);
    sbd::det_vector<std::size_t> parents(parent_end - parent_begin);
    std::vector<double> coefficients(parent_end - parent_begin);
    for(std::size_t index = parent_begin; index < parent_end; ++index) {
      parents[index - parent_begin][0] = all_parents[index];
      coefficients[index - parent_begin] = all_coefficients[index];
    }
    if(geometry.distribution == 1)
      sbd::redistribute_determinants_weight_round_robin(
          parents, coefficients, b_comm);

    const auto hamiltonian = distribute_terms(
        make_heisenberg_terms(), h_rank, geometry.h);
    const auto flat_data = sbd::MakeKetSideGeneralOpFlatData<double>(
        hamiltonian, 1, 64,
        [](double coefficient) { return std::abs(coefficient); });
    test_flat_data_layout(flat_data, world);
    sbd::det_vector<std::size_t> candidates;
    constexpr double cutoff = 0.12;
    sbd::caop::HeatbathExpansion(parents, coefficients, flat_data,
                                 cutoff, 3, candidates,
                                 h_comm, t_comm, world);

    const auto expected = heisenberg_reference(
        all_parents, all_coefficients, cutoff);
    std::size_t local_count = candidates.size();
    std::size_t global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, SBD_MPI_SIZE_T,
                  MPI_SUM, world);
    int local_ok = global_count == expected.size() ? 1 : 0;
    for(const auto& candidate : candidates)
      if(!std::binary_search(expected.begin(), expected.end(), candidate[0]))
        local_ok = 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, world);
    if(!global_ok) {
      if(world_rank == 0)
        std::cerr << "candidate count=" << global_count
                  << " expected=" << expected.size() << '\n';
      throw std::runtime_error("unexpected D-HB candidate set");
    }

    if(world_rank == 0)
      std::cout << "CAOP_HEISENBERG_HEATBATH_TEST PASS h=" << geometry.h
                << " b=" << geometry.b << " t=" << geometry.t
                << " distribution=" << geometry.distribution
                << " candidates=" << expected.size() << '\n';
  } catch(const std::exception& error) {
    if(world_rank == 0) std::cerr << error.what() << '\n';
    result = 1;
  }

  if(h_comm != MPI_COMM_NULL) MPI_Comm_free(&h_comm);
  if(b_comm != MPI_COMM_NULL) MPI_Comm_free(&b_comm);
  if(t_comm != MPI_COMM_NULL) MPI_Comm_free(&t_comm);
  MPI_Finalize();
  return result;
}
