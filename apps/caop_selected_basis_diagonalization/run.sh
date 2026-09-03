mpirun -np 16 -x OMP_NUM_THREADS=1  ./diag --hamfile hamiltonian.txt \
       --basisfiles basis.txt --do_sort_basis 0 --do_redist_basis 1 \
       --t_comm_size 4 --b_comm_size 4 \
       --method 1 --iteration 10 --block 10 --tolerance 1.0e-4 \
       --system_size 8
