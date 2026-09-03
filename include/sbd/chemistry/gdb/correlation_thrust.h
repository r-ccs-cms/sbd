/**
@file sbd/chemistry/tpb/correlation_thrust.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_GDB_CORRELATION_THRUST_H
#define SBD_CHEMISTRY_GDB_CORRELATION_THRUST_H

namespace sbd {
namespace gdb {

template <typename ElemT>
class CorrelationKernelBase : public MultKernelBase<ElemT> {
protected:
    CorrelationKernels<ElemT> correlation;
public:
    CorrelationKernelBase() {}

    CorrelationKernelBase(
                        const thrust::device_vector<ElemT>& v_wb,
                        const thrust::device_vector<ElemT>& v_t,
						const MultGDBThrust<ElemT>& data,
                        thrust::device_vector<ElemT>& b1,
                        thrust::device_vector<ElemT>& b2)
                         : MultKernelBase<ElemT>(v_wb, v_t, data),
                           correlation(data.bit_length(), data.norbs(),  data.I0(), data.I1(), data.I2(), b1, b2)
    {
    }

};

template <typename ElemT>
class CorrelationInit : public CorrelationKernelBase<ElemT>
{
protected:
public:
    CorrelationInit(
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2) : CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2)
    {
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        this->correlation.ZeroDiffCorrelation(this->det + i * this->D_size, this->wb[i]);
    }
};

template <typename ElemT>
class CorrelationSingleAlphaKernel : public CorrelationKernelBase<ElemT>
{
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	CorrelationSingleAlphaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		const uint32_t self_lo = exidx.SelfFromBdetOffset[ibst];
		const uint32_t self_hi = exidx.SelfFromBdetOffset[ibst + 1];
		const uint32_t ja_lo   = exidx.SinglesFromAdetOffset[ia];
		const uint32_t ja_hi   = exidx.SinglesFromAdetOffset[ia + 1];
		if (self_lo != self_hi && ja_lo != ja_hi) {
			uint32_t jbst = exidx.SelfFromBdetSM[self_lo];
			const uint32_t bdet_lo = tidxmap.BdetToDetOffset[jbst];
			const uint32_t bdet_hi = tidxmap.BdetToDetOffset[jbst + 1];
			// single alpha excitations
			for (uint32_t ja = ja_lo; ja < ja_hi; ja++) {
				uint32_t jast = exidx.SinglesFromAdetSM[ja];
				auto [found_a, idxa] = tidxmap.bdet_lower_bound(bdet_lo, bdet_hi, jast);
				if (found_a) {
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					this->correlation.OneDiffCorrelation(this->det + idet * this->D_size,
											this->wb[idet], this->twk[jdet],
											exidx.SinglesAdetCrAnSM[ja],
											exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet]);
				}
			}
		} // if there is same beta string
	}
};

template <typename ElemT>
class CorrelationDoubleAlphaKernel : public CorrelationKernelBase<ElemT>
{
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	CorrelationDoubleAlphaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		const uint32_t self_lo = exidx.SelfFromBdetOffset[ibst];
		const uint32_t self_hi = exidx.SelfFromBdetOffset[ibst + 1];
		const uint32_t ja_lo   = exidx.DoublesFromAdetOffset[ia];
		const uint32_t ja_hi   = exidx.DoublesFromAdetOffset[ia + 1];
		if (self_lo != self_hi && ja_lo != ja_hi) {
			uint32_t jbst = exidx.SelfFromBdetSM[self_lo];
			const uint32_t bdet_lo = tidxmap.BdetToDetOffset[jbst];
			const uint32_t bdet_hi = tidxmap.BdetToDetOffset[jbst + 1];
			// double alpha excitations
			for (uint32_t ja = ja_lo; ja < ja_hi; ja++) {
				uint32_t jast = exidx.DoublesFromAdetSM[ja];
				auto [found_a, idxa] = tidxmap.bdet_lower_bound(bdet_lo, bdet_hi, jast);
				if (found_a) {
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					this->correlation.TwoDiffCorrelation(this->det + idet * this->D_size,
											this->wb[idet], this->twk[jdet],
											exidx.DoublesAdetCrAnSM[ja],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 2],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 3]);
				}
			}
		} // if there is same beta string
	}
};


template <typename ElemT>
class CorrelationSingleBetaKernel : public CorrelationKernelBase<ElemT>
{
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	CorrelationSingleBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;

		const uint32_t self_lo = exidx.SelfFromAdetOffset[iast];
		const uint32_t self_hi = exidx.SelfFromAdetOffset[iast + 1];
		const uint32_t jb_lo   = exidx.SinglesFromBdetOffset[ib];
		const uint32_t jb_hi   = exidx.SinglesFromBdetOffset[ib + 1];
		if (self_lo != self_hi && jb_lo != jb_hi) {
			uint32_t jast = exidx.SelfFromAdetSM[self_lo];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];
			// single beta excitations
			for (uint32_t jb = jb_lo; jb < jb_hi; jb++) {
				uint32_t jbst = exidx.SinglesFromBdetSM[jb];
				auto [found_b, idxb] = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				if (found_b) {
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					this->correlation.OneDiffCorrelation(this->det + idet * this->D_size,
											this->wb[idet], this->twk[jdet],
											exidx.SinglesBdetCrAnSM[jb],
											exidx.SinglesBdetCrAnSM[jb + exidx.size_single_bdet]);
				}
			}
		} // if there is same beta string
	}
};

template <typename ElemT>
class CorrelationDoubleBetaKernel : public CorrelationKernelBase<ElemT>
{
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	CorrelationDoubleBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;

		const uint32_t self_lo = exidx.SelfFromAdetOffset[iast];
		const uint32_t self_hi = exidx.SelfFromAdetOffset[iast + 1];
		const uint32_t jb_lo   = exidx.DoublesFromBdetOffset[ib];
		const uint32_t jb_hi   = exidx.DoublesFromBdetOffset[ib + 1];
		if (self_lo != self_hi && jb_lo != jb_hi) {
			uint32_t jast = exidx.SelfFromAdetSM[self_lo];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];
			// double beta excitations
			for (uint32_t jb = jb_lo; jb < jb_hi; jb++) {
				uint32_t jbst = exidx.DoublesFromBdetSM[jb];
				auto [found_b, idxb] = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				if (found_b) {
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					this->correlation.TwoDiffCorrelation(this->det + idet * this->D_size,
											this->wb[idet], this->twk[jdet],
											exidx.DoublesBdetCrAnSM[jb],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 2],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 3]);
				}
			}
		} // if there is same beta string
	}
};


template <typename ElemT>
class CorrelationAlphaBetaKernel : public CorrelationKernelBase<ElemT>
{
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	CorrelationAlphaBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: CorrelationKernelBase<ElemT>(v_wb, v_t, data, b1, b2), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		// alpha-beta two-particle excitations
		const uint32_t ja_lo = exidx.SinglesFromAdetOffset[ia];
		const uint32_t ja_hi = exidx.SinglesFromAdetOffset[ia + 1];
		const uint32_t k_lo  = exidx.SinglesFromBdetOffset[ibst];
		const uint32_t k_hi  = exidx.SinglesFromBdetOffset[ibst + 1];
		if (k_lo != k_hi) for (uint32_t ja = ja_lo; ja < ja_hi; ja++) {
			uint32_t jast = exidx.SinglesFromAdetSM[ja];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];
			for (uint32_t k = k_lo; k < k_hi; k++) {
				uint32_t jbst = exidx.SinglesFromBdetSM[k];
				auto [found_b, idxb] = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				if (found_b) {
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					this->correlation.TwoDiffCorrelation(this->det + idet * this->D_size,
										this->wb[idet], this->twk[jdet],
											exidx.SinglesAdetCrAnSM[ja],
											exidx.SinglesBdetCrAnSM[k],
											exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet],
											exidx.SinglesBdetCrAnSM[k + exidx.size_single_bdet]);
				}
			}
		}
	}
};


template <typename ElemT>
void MultGDBThrust<ElemT>::correlation(const std::vector<ElemT> & w_in,
				std::vector<std::vector<ElemT>> & onebody_out,
				std::vector<std::vector<ElemT>> & twobody_out)
{
    thrust::device_vector<ElemT> onebody(this->norbs() * this->norbs() * 2, ElemT(0.0));
    thrust::device_vector<ElemT> twobody(this->norbs() * this->norbs() * this->norbs() * this->norbs() * 4, ElemT(0.0));

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(this->h_comm(), &mpi_rank_h);
    MPI_Comm_size(this->h_comm(), &mpi_size_h);

    int mpi_size_b;
    MPI_Comm_size(this->b_comm(), &mpi_size_b);
    int mpi_rank_b;
    MPI_Comm_rank(this->b_comm(), &mpi_rank_b);
    int mpi_size_t;
    MPI_Comm_size(this->t_comm(), &mpi_size_t);
    int mpi_rank_t;
    MPI_Comm_rank(this->t_comm(), &mpi_rank_t);

	thrust::device_vector<ElemT> w(w_in.size());
	thrust::device_vector<ElemT> tw(w_in.size());
	thrust::device_vector<ElemT> rw;
	thrust::device_vector<uint32_t> tidxmap_storage;
	DetIndexMapThrust tidxmap;

    thrust::copy_n(w_in.begin(), w_in.size(), w.begin());

	if (exidx[0].slide != 0) {
		sbd::gdb::MpiSlide(idxmap, idxmap_storage, tidxmap, tidxmap_storage, -exidx[0].slide, this->b_comm());
		sbd::MpiSlide(w, tw, -exidx[0].slide, this->b_comm());
	} else {
		tidxmap.copy(tidxmap_storage, idxmap, idxmap_storage);
		thrust::copy_n(w.begin(), w.size(), tw.begin());
	}

    if (mpi_rank_t == 0) {
		CorrelationInit kernel(w, tw, *this, onebody, twobody);
		kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci, tw.size(), kernel);
	}

	for (size_t task = 0; task < exidx.size(); task++) {
		// single alpha excitations
		CorrelationSingleAlphaKernel kernel_single_alpha(w, tw, *this, onebody, twobody, idxmap, tidxmap, exidx[task]);
		kernel_single_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_sa = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_sa, idxmap.size_adet, kernel_single_alpha);

		// double alpha excitations
		CorrelationDoubleAlphaKernel kernel_double_alpha(w, tw, *this, onebody, twobody, idxmap, tidxmap, exidx[task]);
		kernel_double_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_da = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_da, idxmap.size_adet, kernel_double_alpha);

		// single beta excitations
		CorrelationSingleBetaKernel kernel_single_beta(w, tw, *this, onebody, twobody, idxmap, tidxmap, exidx[task]);
		kernel_single_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_sb = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_sb, idxmap.size_bdet, kernel_single_beta);

		// double beta excitations
		CorrelationDoubleBetaKernel kernel_double_beta(w, tw, *this, onebody, twobody, idxmap, tidxmap, exidx[task]);
		kernel_double_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_db = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_db, idxmap.size_bdet, kernel_double_beta);

		// alpha-beta excitations
		CorrelationAlphaBetaKernel kernel_alpha_beta(w, tw, *this, onebody, twobody, idxmap, tidxmap, exidx[task]);
		kernel_alpha_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_ab = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_ab, idxmap.size_adet, kernel_alpha_beta);

		if (task != exidx.size() - 1) {
			int slide = exidx[task].slide - exidx[task + 1].slide;

			rw = tw;
			sbd::MpiSlide(rw, tw, slide, this->b_comm());
			// Slide the CURRENT index map together with the wavefunction, as the CPU path
			// (correlation.h) and the Hamiltonian loop (mult_thrust.h) do. The previous code
			// re-slid the ORIGINAL idxmap by -exidx[0].slide here before applying `slide`,
			// which left tidxmap one or more tasks behind tw for every task >= 2 (the third
			// task on) when b_comm_size >= 3: wrong 1-/2-RDM values when all ranks hold equal alpha counts,
			// a hang (size mismatch in the correlation kernels) when they do not.
			thrust::device_vector<uint32_t> ridxmap_storage;
			DetIndexMapThrust ridxmap;
			ridxmap.copy(ridxmap_storage, tidxmap, tidxmap_storage);
			sbd::gdb::MpiSlide(ridxmap, ridxmap_storage, tidxmap, tidxmap_storage, slide, this->b_comm());
		}
	} // end for(size_t task=0; task < exidx.size(); task++)

    if (mpi_size_b > 1)
        MpiAllreduce(onebody, MPI_SUM, this->b_comm());
    if (mpi_size_t > 1)
        MpiAllreduce(onebody, MPI_SUM, this->t_comm());
    if (mpi_size_h > 1)
        MpiAllreduce(onebody, MPI_SUM, this->h_comm());

    if (mpi_size_b > 1)
        MpiAllreduce(twobody, MPI_SUM, this->b_comm());
    if (mpi_size_t > 1)
        MpiAllreduce(twobody, MPI_SUM, this->t_comm());
    if (mpi_size_h > 1)
        MpiAllreduce(twobody, MPI_SUM, this->h_comm());

    // copy out onebody, twobody
    onebody_out.resize(2);
    size_t size = this->norbs() * this->norbs();
    size_t offset = 0;
    for(int s=0; s < 2; s++) {
        onebody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(onebody.begin() + offset, size, onebody_out[s].begin());
        offset += size;
    }

    twobody_out.resize(4);
    size = this->norbs() * this->norbs() * this->norbs() * this->norbs();
    offset = 0;
    for(int s=0; s < 4; s++) {
        twobody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(twobody.begin() + offset, size, twobody_out[s].begin());
        offset += size;
    }
} // end function Correlation

} // end namespace gdb
} // end namespace sbd

#endif
