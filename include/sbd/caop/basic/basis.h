/**
@file sbd/caop/basic/basis.h
@brief function to setup the basis
 */
#ifndef SBD_CAOP_BASIC_BASIS_H
#define SBD_CAOP_BASIC_BASIS_H

#include <sys/stat.h>
#include <iomanip>
#include <atomic>
#include <cstdio>
#include <type_traits>

#include <omp.h>

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {
  
  template <typename Container>
  void redistribution(Container & config,
		      size_t bit_length,
		      size_t total_bit_length,
		      MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);

    // Fast-path: if every rank already holds its target count of records AND
    // its slice is locally sorted, the shuffle and re-sort inside
    // mpi_redistribution are both no-ops.  Skip them.
    //
    // This fires at r=1 (trivially: mpi_size==1), and also at r>1 when
    // load_basis_from_files has already k-way-merged the files into sorted
    // per-rank slices of equal size (the common case with equal-size shards).
    //
    // Cost of the check: one MPI_Allreduce (size_t) + one MPI_Allreduce (int)
    // + an O(n/r) sequential sorted scan — negligible vs the sort it avoids.
    {
      size_t local_n = static_cast<size_t>(config.size());
      size_t total_n = 0;
      MPI_Allreduce(&local_n, &total_n, 1, SBD_MPI_SIZE_T, MPI_SUM, comm);

      // Compute the target [want_begin, want_end) range for this rank using the
      // same formula that mpi_redistribution uses internally (get_mpi_range).
      size_t want_begin = 0, want_end = total_n;
      get_mpi_range(mpi_size, mpi_rank, want_begin, want_end);
      size_t want_n = want_end - want_begin;

      // Local checks: correct count AND locally sorted.
      bool local_ok = (local_n == want_n);
      if (local_ok && local_n > 1) {
        for (size_t i = 1; i < local_n && local_ok; ++i)
          if (less_from_back(config[i], config[i-1])) local_ok = false;
      }

      int local_ok_i = static_cast<int>(local_ok);
      int all_ok = 0;
      MPI_Allreduce(&local_ok_i, &all_ok, 1, MPI_INT, MPI_LAND, comm);
      if (all_ok) return;
    }

    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_redistribution(config,config_begin,config_end,index_begin,index_end,
		       total_bit_length,bit_length,comm);

  }
  


  // Sort idx[lo..hi) so that (a[idx[i]][elem] & Mask) is in ascending order,
  // recursing on equal-valued runs through decreasing elements.
  // Mask is a template parameter so the compiler specialises each instantiation:
  // Mask=~0 eliminates the AND entirely; Mask=0x5555... is baked into code.
  template<size_t Mask = ~size_t(0), typename Container>
  void idx_sort_from_back(std::vector<size_t>& idx,
                          const Container& a,
                          size_t lo, size_t hi, int elem) {
    if (hi - lo <= 1 || elem < 0) return;
    std::sort(idx.begin() + lo, idx.begin() + hi,
              [&a, elem](size_t x, size_t y) {
                return (a[x][elem] & Mask) < (a[y][elem] & Mask);
              });
    if (elem == 0) return;
    size_t run_lo = lo;
    for (size_t i = lo + 1; i <= hi; i++) {
      if (i == hi || (a[idx[i]][elem] & Mask) != (a[idx[run_lo]][elem] & Mask)) {
        if (i - run_lo > 1)
          idx_sort_from_back<Mask>(idx, a, run_lo, i, elem - 1);
        run_lo = i;
      }
    }
  }

  // Equal-bra_a redistribution: partition dets so each rank owns a disjoint
  // contiguous range of alpha strings, giving equal bra_a across ranks.
  // Alpha occupation is at even bit positions (0,2,4,...) of the det bitstring,
  // interleaved with beta at odd positions.  Works for any clen.
  void redistribution_equal_bra_a(det_vector<size_t> & config,
                                   size_t bit_length,
                                   size_t total_bit_length,
                                   MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);

    const int clen = static_cast<int>((total_bit_length + bit_length - 1) / bit_length);
    // Alpha bits sit at even positions (0,2,4,...) of each interleaved word.
    // Masking with ALPHA_MASK zeroes beta bits in place; the remaining even-position
    // bits compare correctly with < because bit position order is preserved.
    constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;

    // Step 1: sort local dets alpha-primary by masking beta bits out of each word.
    // Det order within each alpha group is irrelevant; Step 7 re-sorts beta-primary.
    std::vector<size_t> idx(config.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    idx_sort_from_back<ALPHA_MASK>(idx, config, 0, idx.size(), clen - 1);
    {
      det_vector<size_t> tmp(config.size());
      for (size_t i = 0; i < config.size(); i++) tmp[i] = config[idx[i]];
      config = std::move(tmp);
    }

    // Step 2: collect local unique alpha keys from sorted config (mask on the fly).
    det_vector<size_t> local_alphas;
    {
      std::vector<size_t> prev(clen, ~size_t(0));
      for (size_t j = 0; j < config.size(); j++) {
        bool diff = false;
        for (int k = 0; k < clen && !diff; k++)
          diff = ((config[j][k] & ALPHA_MASK) != prev[k]);
        if (diff) {
          for (int k = 0; k < clen; k++)
            prev[k] = config[j][k] & ALPHA_MASK;
          local_alphas.emplace_back(prev);
        }
      }
    }

    // Step 3: allgather unique alpha keys → global sorted unique list.
    // Each alpha key is clen words; exchange as flat size_t arrays.
    int local_n = static_cast<int>(local_alphas.size());
    std::vector<int> all_counts(mpi_size);
    MPI_Allgather(&local_n, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
    int total_n = 0;
    std::vector<int> ag_displs_w(mpi_size, 0), ag_counts_w(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
      ag_displs_w[r] = total_n * clen;
      ag_counts_w[r] = all_counts[r] * clen;
      total_n += all_counts[r];
    }
    std::vector<size_t> flat_local(local_n * clen), flat_all(total_n * clen);
    for (int i = 0; i < local_n; i++)
      for (int k = 0; k < clen; k++) flat_local[i * clen + k] = local_alphas[i][k];
    MPI_Allgatherv(flat_local.data(), local_n * clen, SBD_MPI_SIZE_T,
                   flat_all.data(), ag_counts_w.data(), ag_displs_w.data(), SBD_MPI_SIZE_T, comm);
    det_vector<size_t> all_alphas(static_cast<size_t>(total_n));
    for (int i = 0; i < total_n; i++)
      for (int k = 0; k < clen; k++) all_alphas[i][k] = flat_all[i * clen + k];
    sort_bitarray(all_alphas);
    size_t global_bra_a = all_alphas.size();

    // Step 4: assign alpha strings to ranks with equal bra_a.
    size_t chunk = global_bra_a / static_cast<size_t>(mpi_size);
    size_t rem   = global_bra_a % static_cast<size_t>(mpi_size);
    std::vector<size_t> alpha_start(mpi_size + 1, 0);
    for (int r = 0; r < mpi_size; r++)
      alpha_start[r+1] = alpha_start[r] + chunk + (static_cast<size_t>(r) < rem ? 1 : 0);

    // Step 5: for each local det compute destination rank.
    // all_alphas entries are already masked; apply ALPHA_MASK to config[j] inline.
    std::vector<int> dest_per_det(config.size());
    std::vector<int> sendcounts(mpi_size, 0);
    for (size_t j = 0; j < config.size(); j++) {
      size_t pos = static_cast<size_t>(
        std::lower_bound(all_alphas.begin(), all_alphas.end(), config[j],
          [clen](const auto& alpha, const auto& det) {
            constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;
            for (int k = clen - 1; k >= 0; k--) {
              if (alpha[k] < (det[k] & ALPHA_MASK)) return true;
              if (alpha[k] > (det[k] & ALPHA_MASK)) return false;
            }
            return false;
          })
        - all_alphas.begin());
      int dest = static_cast<int>(
        std::upper_bound(alpha_start.begin(), alpha_start.end(), pos)
        - alpha_start.begin()) - 1;
      if (dest < 0) dest = 0;
      if (dest >= mpi_size) dest = mpi_size - 1;
      dest_per_det[j] = dest;
      sendcounts[dest]++;
    }

    // Step 6: MPI_Alltoallv (clen words per det).
    std::vector<int> sendcounts_w(mpi_size), sdispls(mpi_size, 0);
    for (int r = 0; r < mpi_size; r++) sendcounts_w[r] = sendcounts[r] * clen;
    for (int r = 1; r < mpi_size; r++) sdispls[r] = sdispls[r-1] + sendcounts_w[r-1];
    std::vector<int> recvcounts_w(mpi_size), rdispls(mpi_size, 0);
    MPI_Alltoall(sendcounts_w.data(), 1, MPI_INT, recvcounts_w.data(), 1, MPI_INT, comm);
    for (int r = 1; r < mpi_size; r++) rdispls[r] = rdispls[r-1] + recvcounts_w[r-1];
    int total_recv_w = rdispls[mpi_size-1] + recvcounts_w[mpi_size-1];

    std::vector<size_t> sendbuf(config.size() * clen);
    {
      std::vector<int> fill(sdispls);
      for (size_t j = 0; j < config.size(); j++) {
        int d = dest_per_det[j];
        for (int k = 0; k < clen; k++) sendbuf[fill[d]++] = config[j][k];
      }
    }
    std::vector<size_t> recvbuf(total_recv_w);
    MPI_Alltoallv(sendbuf.data(), sendcounts_w.data(), sdispls.data(), SBD_MPI_SIZE_T,
                  recvbuf.data(), recvcounts_w.data(), rdispls.data(), SBD_MPI_SIZE_T, comm);

    // Step 7: unpack and restore kernel sort order (beta-primary = less_from_back).
    size_t n_recv = static_cast<size_t>(total_recv_w) / static_cast<size_t>(clen);
    config.resize(n_recv);
    for (size_t i = 0; i < n_recv; i++)
      for (int k = 0; k < clen; k++) config[i][k] = recvbuf[i * clen + k];
    sort_bitarray(config);
  }

  template <typename Container>
  void reordering(Container & config,
		  size_t bit_length,
		  size_t total_bit_length,
		  MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,
		      total_bit_length,bit_length,comm);
  }
  
  // I/O for basis
  template<typename Container>
  void load_basis_from_file(const std::string & filename,
			    Container & config,
			    size_t bit_length,
			    size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      // Fast path: text is regular fixed-width records (total_bit_length chars of
      // '0'/'1' + '\n').  Read the whole file in one shot and parse in-place.
      int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // --- open ---
      double t0 = omp_get_wtime();
      std::ifstream ifs(filename, std::ios::binary);
      if( !ifs.is_open() )
	throw std::runtime_error("Failed to open basis bit-string file.");
      double t_open = omp_get_wtime();

      ifs.seekg(0, std::ios::end);
      const size_t file_size = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      const size_t record_size = total_bit_length + 1;  // L chars + '\n'
      const size_t num_records = (file_size + 1) / record_size;
      const size_t num_words   = (total_bit_length + bit_length - 1) / bit_length;

      // --- alloc read buffer (zero-init touches all pages) ---
      std::vector<char> buf(file_size);
      double t_alloc = omp_get_wtime();

      // --- read ---
      ifs.read(buf.data(), file_size);
      if( !ifs )
	throw std::runtime_error("Failed to read basis bit-string file.");
      double t_read = omp_get_wtime();

      // --- alloc output container (single flat allocation of num_records * num_words * 8B) ---
      config.resize(num_records);
      double t_resize = omp_get_wtime();

      // --- parse: each thread processes its own row range in the flat buffer ---
      std::atomic<bool> parse_error{false};
      #pragma omp parallel for schedule(static)
      for(size_t i = 0; i < num_records; i++) {
        config[i].resize(num_words);
        std::fill(config[i].begin(), config[i].end(), size_t(0));
        const char* rec = buf.data() + i * record_size;
        for(size_t j = 0; j < total_bit_length; j++) {
          if( rec[j] == '1' ) {
            size_t bit_idx = total_bit_length - 1 - j;
            config[i][bit_idx / bit_length] |= (size_t(1) << (bit_idx % bit_length));
          } else if( rec[j] != '0' ) {
            parse_error.store(true, std::memory_order_relaxed);
          }
        }
      }
      if( parse_error.load() )
        throw std::runtime_error(
          "Unexpected character in basis file: expected '0' or '1'");
      double t_parse = omp_get_wtime();

      // Report per-rank, per-file breakdown to stderr.
      const double read_dt = t_read - t_alloc;
      const double read_bw = (read_dt > 1e-6) ? file_size / read_dt / 1e6 : 0.0;
      fprintf(stderr,
        "basis-timing rank=%d file=%s size=%.0fMB "
        "open=%.3fs buf_alloc=%.3fs read=%.3fs(%.0fMB/s) "
        "cfg_alloc=%.3fs parse=%.3fs total=%.3fs\n",
        rank, filename.c_str(), file_size / 1.0e6,
        t_open   - t0,
        t_alloc  - t_open,
        read_dt, read_bw,
        t_resize - t_read,
        t_parse  - t_resize,
        t_parse  - t0);
      fflush(stderr);
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ifstream ifs(filename, std::ios::binary);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string binary file.");
      }

      size_t inner_size = (total_bit_length+bit_length-1)/bit_length;
      ifs.seekg(0, std::ios::end);
      std::streampos file_size = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      size_t bytes_per_line = inner_size * sizeof(size_t);

      if( file_size % bytes_per_line != 0 ) {
	throw std::runtime_error("Binary file size mismatch");
      }

      size_t num_lines = file_size / bytes_per_line;

      config.resize(num_lines);
      for(size_t i=0; i < num_lines; i++) {
	config[i].resize(inner_size);
	ifs.read(reinterpret_cast<char*>(config[i].data()),bytes_per_line);
	if (!ifs) {
	  throw std::runtime_error("Failed to read binary basis data.");
	}
      }
    }
  }
  
  template<typename Container>
  void save_basis_to_file(const std::string & filename,
			  Container & config,
			  size_t bit_length,
			  size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ofstream ofs(filename);
      for(size_t i=0; i < config.size(); i++) {
	ofs << makestring(config[i],bit_length,total_bit_length) << std::endl;
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ofstream ofs(filename,std::ios::binary);
      for(auto & b : config) {
	ofs.write(reinterpret_cast<char*>(b.data()),sizeof(size_t)*b.size());
      }
    }
  }

  // basis file name for multiple nodes
  std::string basisfilename(const std::string & basisname, int index, int filetype) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename;
    if( filetype == 0 ) {
      filename = basisname + tag + ".txt";
    } else if ( filetype == 1 ) {
      filename = basisname + tag + ".bin";
    }
    return filename;
  }

  // ---- Helper: is_sorted_fromback -----------------------------------------
  // O(n) check: is container a sorted in less_from_back order?
  // Works for det_vector<size_t> (rows have size() and operator[]).
  template<typename Container>
  bool is_sorted_fromback(const Container& a) {
    const size_t n = a.size();
    if (n <= 1) return true;
    for (size_t i = 1; i < n; ++i) {
      if (less_from_back(a[i], a[i-1]))  // a[i] < a[i-1] → out of order
        return false;
    }
    return true;
  }

  // ---- Helper: kway_merge_detvec -------------------------------------------
  // K-way merge of k already-sorted det_vector<size_t> sources into dest.
  // Uses a min-heap over (row-pointer, source-index, row-index).
  // Deduplicates identical rows (consistent with sort_bitarray semantics).
  // Precondition: every sources[i] is sorted in less_from_back order.
  template<det_kind Kind>
  void kway_merge_detvec(std::vector<det_vector<size_t, Kind>>& sources,
                         det_vector<size_t, Kind>& dest) {
    const int k = static_cast<int>(sources.size());
    if (k == 0) { dest.clear(); return; }
    if (k == 1) { dest = std::move(sources[0]); return; }

    size_t total = 0;
    for (auto& s : sources) total += s.size();
    dest.resize(total);  // over-allocate; trimmed to unique_n at end

    const size_t row_len = sources[0].elem_size();

    struct Entry { size_t* ptr; int src; size_t idx; };
    // STL heap is a max-heap; comparator inverted so smallest row is at top.
    const auto heap_cmp = [row_len](const Entry& a, const Entry& b) noexcept {
      for (int w = static_cast<int>(row_len) - 1; w >= 0; --w) {
        if (b.ptr[w] < a.ptr[w]) return true;   // b < a → a should not be at top
        if (b.ptr[w] > a.ptr[w]) return false;
      }
      return false;
    };

    std::vector<Entry> heap;
    heap.reserve(k);
    for (int i = 0; i < k; ++i)
      if (!sources[i].empty())
        heap.push_back({sources[i][0].data(), i, 0});
    std::make_heap(heap.begin(), heap.end(), heap_cmp);

    size_t out = 0;
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), heap_cmp);
      Entry e = heap.back(); heap.pop_back();

      // Dedup: skip if equal to last written row.
      if (out == 0 || std::memcmp(dest[out-1].data(), e.ptr,
                                   row_len * sizeof(size_t)) != 0) {
        std::memcpy(dest[out].data(), e.ptr, row_len * sizeof(size_t));
        ++out;
      }

      if (e.idx + 1 < sources[e.src].size()) {
        e.idx++;
        e.ptr = sources[e.src][e.idx].data();
        heap.push_back(e);
        std::push_heap(heap.begin(), heap.end(), heap_cmp);
      }
    }
    dest.resize(out);  // trim to unique count
  }

  // ---- load_basis_from_files -----------------------------------------------
  // Loads the rank's assigned shard files sequentially, checks each is sorted
  // (aborts with an error if not — sort files in advance with
  // scripts/sort-basis-shards.py), then k-way merges the sorted runs.
  // The sort_bitarray() call that previously dominated (~28 s at r=1 for 100M
  // records) is eliminated: the merge is O(n log k) with k = files-per-rank.
  template<typename Container>
  void load_basis_from_files(const std::vector<std::string> & all_filenames,
			     Container & config,
			     size_t bit_length,
			     size_t total_bit_length,
			     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);

    const int num_files = static_cast<int>(all_filenames.size());
    config.clear();
    if (num_files == 0) return;

    const int base = num_files / mpi_size;
    const int rem  = num_files % mpi_size;
    int my_first = 0, my_count = 0;
    if (mpi_rank < rem) {
      my_count = base + 1;
      my_first = mpi_rank * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1) + (mpi_rank - rem) * base;
    }

    if constexpr (std::is_same_v<Container, det_vector<size_t>>) {
      // Fast path for det_vector<size_t>: check-sorted + k-way merge.
      double t0 = omp_get_wtime();

      // Load files sequentially (concurrent reads contend on shared FS).
      std::vector<Container> per_file(my_count);
      for (int i = 0; i < my_count; ++i)
        load_basis_from_file(all_filenames[my_first + i], per_file[i],
                             bit_length, total_bit_length);
      double t_load = omp_get_wtime();

      // Require each file to be sorted; abort otherwise.
      // Sort files once in advance with: python3 scripts/sort-basis-shards.py FILE...
      for (int i = 0; i < my_count; ++i) {
        if (!is_sorted_fromback(per_file[i])) {
          fprintf(stderr,
            "load_basis_from_files rank=%d: ERROR file not sorted: %s\n"
            "  Sort shard files in advance with:\n"
            "    python3 scripts/sort-basis-shards.py FILE...\n",
            mpi_rank, all_filenames[my_first + i].c_str());
          fflush(stderr);
          MPI_Abort(comm, 1);
        }
      }
      double t_check = omp_get_wtime();

      // K-way merge of sorted runs, or plain concatenation if globally ordered.
      bool globally_ordered = (my_count <= 1);
      if (!globally_ordered) {
        globally_ordered = true;
        for (int i = 0; i + 1 < my_count && globally_ordered; ++i) {
          if (!per_file[i].empty() && !per_file[i+1].empty()) {
            // last of per_file[i] must be <= first of per_file[i+1]
            if (less_from_back(per_file[i+1][0],
                               per_file[i][per_file[i].size()-1]))
              globally_ordered = false;
          }
        }
      }

      if (globally_ordered) {
        for (int i = 0; i < my_count; ++i)
          config.insert(config.end(), per_file[i].begin(), per_file[i].end());
      } else {
        kway_merge_detvec(per_file, config);
      }
      double t_done = omp_get_wtime();

      fprintf(stderr,
        "load-files rank=%d files=%d: load=%.3fs check=%.3fs "
        "merge=%.3fs(globally_ordered=%d) total=%.3fs\n",
        mpi_rank, my_count,
        t_load - t0, t_check - t_load,
        t_done - t_check, (int)globally_ordered,
        t_done - t0);
      fflush(stderr);

      // ---- Global sorted-and-no-cross-shard-dup check -------------------------
      // Send this rank's last element to rank+1; rank+1 verifies its first
      // element is strictly greater.  Cost: one MPI_Sendrecv of row_len*8 bytes.
      // Catches overlapping shard ranges (e.g. round-robin gen_bits.py output)
      // that are individually sorted but collectively non-monotone across ranks.
      if (mpi_size > 1) {
        const size_t row_len = (config.empty() ? 0 : config.elem_size());
        size_t max_row_len = 0;
        MPI_Allreduce(&row_len, &max_row_len, 1, SBD_MPI_SIZE_T, MPI_MAX, comm);

        if (max_row_len > 0) {
          // Send last element (or all-zeros sentinel if this rank is empty).
          std::vector<size_t> send_buf(max_row_len, 0);
          if (!config.empty()) {
            const auto& last = config[config.size() - 1];
            std::memcpy(send_buf.data(), last.data(), max_row_len * sizeof(size_t));
          }
          std::vector<size_t> recv_buf(max_row_len, 0);

          const int send_to   = (mpi_rank + 1) % mpi_size;
          const int recv_from = (mpi_rank - 1 + mpi_size) % mpi_size;
          MPI_Sendrecv(send_buf.data(), static_cast<int>(max_row_len), SBD_MPI_SIZE_T,
                       send_to,   98,
                       recv_buf.data(), static_cast<int>(max_row_len), SBD_MPI_SIZE_T,
                       recv_from, 98, comm, MPI_STATUS_IGNORE);

          // Rank 0's recv_buf is rank (mpi_size-1)'s last element — wrap-around,
          // not a continuity constraint.  Only check ranks 1..mpi_size-1.
          bool local_ok = true;
          if (mpi_rank > 0 && !config.empty()) {
            const auto& my_first = config[0];
            bool is_dup = (std::memcmp(my_first.data(), recv_buf.data(),
                                       max_row_len * sizeof(size_t)) == 0);
            bool out_of_order = less_from_back(my_first, recv_buf);
            local_ok = !is_dup && !out_of_order;
          }

          int local_ok_i = static_cast<int>(local_ok);
          int all_ok = 0;
          MPI_Allreduce(&local_ok_i, &all_ok, 1, MPI_INT, MPI_LAND, comm);

          if (!all_ok) {
            if (mpi_rank == 0) {
              fprintf(stderr,
                "sbd: ERROR: basis shards are not globally sorted or contain\n"
                "  cross-shard duplicate records.  Each shard must cover a\n"
                "  non-overlapping sorted range (as produced by gdet).\n"
                "  Fix: regenerate shards with:\n"
                "    python3 sbd/apps/caop_selected_basis_diagonalization/gen_bits.py"
                " --sorted-split ...\n"
                "  or run gdet on the alpha-det file.\n");
              fflush(stderr);
            }
            MPI_Abort(comm, 1);
          }
        }
      }

    } else {
      // Fallback for other Container types (e.g. std::vector<std::vector<size_t>>):
      // original sequential load + sort_bitarray.
      for (int i = my_first; i < my_first + my_count; ++i) {
        Container local;
        load_basis_from_file(all_filenames[i], local, bit_length, total_bit_length);
        config.insert(config.end(),
                      std::make_move_iterator(local.begin()),
                      std::make_move_iterator(local.end()));
      }
      sort_bitarray(config);
    }
  }
  
  // load single file
  void load_basis_from_single_binary(const std::string & filename,
				     std::vector<std::vector<size_t>> & config,
				     size_t bit_length,
				     size_t total_bit_length,
				     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const size_t inner_size    = (total_bit_length + bit_length - 1) / bit_length;
    const size_t bytes_per_line = inner_size * sizeof(size_t);
    
    std::uint64_t num_lines_u64 = 0;
    
    if (mpi_rank == 0) {
      std::ifstream ifs(filename, std::ios::binary);
      if (!ifs.is_open()) {
	throw std::runtime_error("Failed to open basis binary file: " + filename);
      }
      
      ifs.seekg(0, std::ios::end);
      std::streampos file_size_pos = ifs.tellg();
      ifs.seekg(0, std::ios::beg);
      
    if (file_size_pos < 0) {
      throw std::runtime_error("tellg() failed for file: " + filename);
    }
    
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_size_pos);
    
    if (file_size % bytes_per_line != 0) {
      throw std::runtime_error("Binary file size mismatch in " + filename);
    }
    
    num_lines_u64 = file_size / bytes_per_line;
    }
    
    MPI_Bcast(&num_lines_u64, 1, MPI_UINT64_T, 0, comm);

    if (num_lines_u64 == 0) {
      config.clear();
      return;
    }
    
    const std::size_t num_lines = static_cast<std::size_t>(num_lines_u64);

    const std::size_t base = num_lines / mpi_size;
    const std::size_t rem  = num_lines % mpi_size;
    
    std::size_t my_first = 0;
    std::size_t my_count = 0;
    if (static_cast<std::size_t>(mpi_rank) < rem) {
      my_count = base + 1;
      my_first = static_cast<std::size_t>(mpi_rank) * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1)
	+ (static_cast<std::size_t>(mpi_rank) - rem) * base;
    }
    const std::size_t my_last = my_first + my_count;
    
    config.clear();
    config.resize(my_count);
    for (auto & row : config) {
      row.resize(inner_size);
    }
    
    if (my_count == 0) {
      return;
    }
    
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open basis binary file (per rank): " + filename);
    }

    const std::uint64_t my_offset_bytes =
      static_cast<std::uint64_t>(my_first) * bytes_per_line;
    
    ifs.seekg(static_cast<std::streamoff>(my_offset_bytes), std::ios::beg);
    if (!ifs) {
      throw std::runtime_error("seekg failed for basis binary file: " + filename);
    }
    
    for (std::size_t i = 0; i < my_count; ++i) {
      ifs.read(reinterpret_cast<char*>(config[i].data()), bytes_per_line);
      if (!ifs) {
	throw std::runtime_error("Failed to read basis data from: " + filename);
      }
    }
    sort_bitarray(config);
  }

  inline void mpi_bcast_string_vector(std::vector<std::string> & vec,
				      int root,
				      MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int count = static_cast<int>(vec.size());
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    
    if (rank != root) {
      vec.resize(count);
    }
    
    std::vector<int> lengths(count);
    if (rank == root) {
      for (int i = 0; i < count; i++) {
	lengths[i] = static_cast<int>(vec[i].size());
      }
    }
    MPI_Bcast(lengths.data(), count, MPI_INT, root, comm);
    
    for (int i = 0; i < count; i++) {
      if (rank != root) {
	vec[i].resize(lengths[i]);
      }
      if (lengths[i] > 0) {
#ifdef SBD_TRADMODE
	char * ptr = &vec[i][0];
	MPI_Bcast(ptr, lengths[i], MPI_CHAR, root, comm);
#else
	MPI_Bcast(vec[i].data(), lengths[i], MPI_CHAR, root, comm);
#endif
      }
    }
  }
  
  
}

#endif
