mpirun -np 16 -x OMP_NUM_THREADS=1 ./diag --fcidump fcidump_Fe4S4.txt \
       --detfiles det0.txt,det1.txt,det2.txt,det3.txt \
       --do_redist_alpha_eq 1 \
       --method 0 --block 10 --iteration 4 --tolerance 1.0e-4 \
       --b_comm_size 2 --t_comm_size 1 \
       --init 0 --shuffle 0 \
       --carryovername carryover.txt \
       --rdm 1 --carryover_type 1 --carryover_ratio 0.1
