#!/usr/bin/env python3
"""Generate fixed-Hamming-weight bitstrings under U(1) symmetry.

Two output modes:

sorted-split (default when multiple -o files are given):
    Generate all strings in parallel (one process per --workers N workers),
    sort and dedup each chunk in parallel, merge-sort all chunks into a single
    sorted stream, then range-split into output files so each file holds a
    contiguous non-overlapping sorted range — matching gdet's output format.
    The diag executable enforces this layout at startup via a global-sort
    check; round-robin files will cause an immediate abort.

round-robin (--round-robin, or default when zero/one -o files are given):
    Bitstrings are generated one at a time and written to output files in
    round-robin order (bitstring i → file i % n_files).  Files cover
    overlapping value ranges — only useful for single-file output or testing.

Usage (4-shard diag run — sorted-split is automatic with multiple -o files):
    python3 gen_bits.py --bitlength 48 --numones 24 --num 100000000 --seed 42 \\
        --workers 32 \\
        -o basis48-25M-00.txt basis48-25M-01.txt basis48-25M-02.txt basis48-25M-03.txt
"""

import argparse
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Worker (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _worker_generate_sorted(args):
    """Generate ``count`` random bitstrings, sort+dedup in-process, write to
    ``out_path``.  Returns ``out_path`` on success.

    Each worker uses its own seeded RNG so output is deterministic and
    workers are fully independent (no shared state, no locking).
    The in-worker sort means the main process only needs ``sort --merge``
    (linear-time merge of pre-sorted runs) rather than a full re-sort.
    """
    worker_id, bit_length, num_ones, count, seed, out_path = args
    rng = random.Random(seed)
    base_bits = [1] * num_ones + [0] * (bit_length - num_ones)

    # Generate directly into a list, then sort+dedup in Python (faster than
    # writing to a temp file and shelling out for small per-worker counts).
    strings = []
    for _ in range(count):
        rng.shuffle(base_bits)
        strings.append("".join(str(b) for b in base_bits))

    strings.sort()

    # Dedup (strings is sorted so duplicates are adjacent)
    with open(out_path, "w") as f:
        prev = None
        for s in strings:
            if s != prev:
                f.write(s + "\n")
                prev = s

    return out_path


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def method_0_random(bit_length: int, num_ones: int, num_bitstrings: int,
                    rng: random.Random):
    """Method 0: random Hamming-weight-preserving shuffle, one string at a time."""
    base_bits = [1] * num_ones + [0] * (bit_length - num_ones)
    for _ in range(num_bitstrings):
        rng.shuffle(base_bits)
        yield "".join(str(b) for b in base_bits)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _mpi_range(n_files: int, rank: int, total: int):
    """(i_begin, i_end) for shard ``rank`` — identical to SBD's get_mpi_range."""
    base  = total // n_files
    extra = total % n_files
    i_begin = rank * base + min(rank, extra)
    i_end   = i_begin + base + (1 if rank < extra else 0)
    return i_begin, i_end


def write_roundrobin(bitstrings, outfiles):
    """Write bitstrings to ``outfiles`` in round-robin order (legacy mode)."""
    if not outfiles:
        for bs in bitstrings:
            print(bs)
        return

    n_files = len(outfiles)
    handles = [open(name, "w") for name in outfiles]
    try:
        for idx, bs in enumerate(bitstrings):
            handles[idx % n_files].write(bs + "\n")
    finally:
        for f in handles:
            f.close()


def write_sorted_split_parallel(bit_length: int, num_ones: int, total_n: int,
                                seed, outfiles: list, n_workers: int = None):
    """Generate, sort, and range-split into non-overlapping shard files.

    Pipeline:
      1. N workers each generate (total_n / N) strings, sort+dedup in-process,
         write sorted chunks to temp files — all in parallel.
      2. ``sort --merge --unique`` merges the pre-sorted chunks (linear time).
      3. Count unique lines; compute get_mpi_range boundaries.
      4. Stream through sorted file, write each line to its shard by index.

    Wall time is dominated by the parallel generation step (~total_n / N / shuffle_rate).
    """
    if not outfiles:
        # No output files: fall back to printing sorted unique strings.
        master_rng = random.Random(seed)
        strings = list(method_0_random(bit_length, num_ones, total_n, master_rng))
        strings.sort()
        prev = None
        for s in strings:
            if s != prev:
                print(s)
                prev = s
        return

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 64)

    n_files   = len(outfiles)
    dirpath   = os.path.dirname(os.path.abspath(outfiles[0]))

    # Derive deterministic per-worker seeds from the main seed.
    master_rng   = random.Random(seed)
    worker_seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(n_workers)]

    # Distribute total_n across workers (each gets base or base+1 items).
    base_count = total_n // n_workers
    extra      = total_n % n_workers
    counts     = [base_count + (1 if i < extra else 0) for i in range(n_workers)]

    # Temp paths for per-worker sorted chunks.
    tmp_chunks = [
        tempfile.mktemp(dir=dirpath, suffix=f".w{i:03d}.chunk_tmp")
        for i in range(n_workers)
    ]

    print(f"  [gen_bits] {n_workers} workers × ~{base_count:,} strings each "
          f"(total {total_n:,}) ...", file=sys.stderr, flush=True)

    # ---- Step 1: parallel generate + sort ----------------------------------
    worker_args = [
        (i, bit_length, num_ones, counts[i], worker_seeds[i], tmp_chunks[i])
        for i in range(n_workers)
    ]
    try:
        with mp.Pool(n_workers) as pool:
            pool.map(_worker_generate_sorted, worker_args)

        # ---- Step 2: merge pre-sorted chunks --------------------------------
        tmp_merged = tempfile.mktemp(dir=dirpath, suffix=".merged.gen_tmp")
        try:
            print(f"  [gen_bits] sort --merge --unique ({n_workers} chunks) ...",
                  file=sys.stderr, flush=True)
            subprocess.run(
                ["sort", "--merge", "--unique", "-o", tmp_merged] + tmp_chunks,
                check=True,
            )
            for p in tmp_chunks:
                if os.path.exists(p):
                    os.unlink(p)

            # ---- Step 3: count unique lines --------------------------------
            result = subprocess.run(
                ["wc", "-l", tmp_merged],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            unique_total = int(result.stdout.split()[0])
            print(f"  [gen_bits] {unique_total:,} unique strings after dedup",
                  file=sys.stderr, flush=True)

            # Precompute range boundaries
            boundaries = [_mpi_range(n_files, r, unique_total)[0]
                          for r in range(n_files)] + [unique_total]

            # ---- Step 4: stream-split to output files ----------------------
            handles = [open(fn, "w") for fn in outfiles]
            try:
                cur = 0
                with open(tmp_merged) as sf:
                    for i, line in enumerate(sf):
                        while cur + 1 < n_files and i >= boundaries[cur + 1]:
                            cur += 1
                        handles[cur].write(line)
            finally:
                for fh in handles:
                    fh.close()

            for r, fn in enumerate(outfiles):
                count = boundaries[r + 1] - boundaries[r]
                print(f"  [gen_bits] {fn}: {count:,} records",
                      file=sys.stderr, flush=True)

        finally:
            if os.path.exists(tmp_merged):
                os.unlink(tmp_merged)

    finally:
        # Best-effort cleanup of any leftover chunk files.
        for p in tmp_chunks:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bitlength", type=int, required=True,
                        help="Total bitstring length (system size).")
    parser.add_argument("--numones", type=int, required=True,
                        help="Number of 1-bits (particle number).")
    parser.add_argument("--num", type=int, default=10,
                        help="How many bitstrings to generate (default: 10).")
    parser.add_argument("--method", type=int, default=0,
                        help="Generation method.  Currently only 0 (random).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (optional).")
    parser.add_argument("--unique", action="store_true",
                        help="(Round-robin mode only) output only unique bitstrings.")
    parser.add_argument("--sorted-split", action="store_true",
                        help=(
                            "Force sorted-split mode even for single-file or stdout output."
                        ))
    parser.add_argument("--round-robin", action="store_true",
                        help=(
                            "Force round-robin mode even when multiple -o files are given.  "
                            "Produces overlapping ranges; diag will reject these files at "
                            "startup.  Useful only for testing or single-file output."
                        ))
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for sorted-split mode "
                             "(default: min(cpu_count, 64)).")
    parser.add_argument("-o", "--outfile", nargs="+",
                        help="Output file(s).  Multiple files: sorted-split by default "
                             "(use --round-robin to override).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.bitlength <= 0:
        print("Error: --bitlength must be positive.", file=sys.stderr)
        sys.exit(1)
    if not (0 <= args.numones <= args.bitlength):
        print("Error: --numones must satisfy 0 <= numones <= bitlength.",
              file=sys.stderr)
        sys.exit(1)
    if args.num <= 0:
        print("Error: --num must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.sorted_split and args.round_robin:
        print("Error: --sorted-split and --round-robin are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    # Default: sorted-split when multiple output files are given (required by
    # diag's global-sort check); round-robin otherwise.
    use_sorted_split = (
        args.sorted_split
        or (not args.round_robin and len(args.outfile or []) > 1)
    )

    if use_sorted_split:
        write_sorted_split_parallel(
            args.bitlength, args.numones, args.num,
            args.seed, args.outfile or [],
            n_workers=args.workers,
        )
    elif args.unique:
        strings = sorted(set(method_0_random(args.bitlength, args.numones,
                                             args.num, rng)))
        write_roundrobin(iter(strings), args.outfile)
    else:
        generator = method_0_random(args.bitlength, args.numones, args.num, rng)
        write_roundrobin(generator, args.outfile)


if __name__ == "__main__":
    main()
