/**
@file sbd/chemistry/basic/correlation_thrust.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/

#ifndef SBD_CHEMISTRY_BASIC_CORRELATION_THRUST_H
#define SBD_CHEMISTRY_BASIC_CORRELATION_THRUST_H

namespace sbd
{

template <typename ElemT>
class CorrelationKernels : public DeterminantKernels<ElemT> {
protected:
    ElemT* onebody;
    ElemT* twobody;
    size_t onebody_size;
    size_t twobody_size;
public:
    CorrelationKernels() {}

    CorrelationKernels(const size_t bit_length_in, const size_t norbs_in,
                        const ElemT zero_in,
                        const oneInt_Thrust<ElemT> one_in,
                        const twoInt_Thrust<ElemT> two_in,
                        thrust::device_vector<ElemT>& b1,
                        thrust::device_vector<ElemT>& b2
                    ) : DeterminantKernels<ElemT>(bit_length_in, norbs_in, zero_in, one_in, two_in)
    {
        onebody = (ElemT*)thrust::raw_pointer_cast(b1.data());
        twobody = (ElemT*)thrust::raw_pointer_cast(b2.data());
        onebody_size = this->norbs * this->norbs;
        twobody_size = this->norbs * this->norbs * this->norbs * this->norbs;
    }

    /**
         Function for adding diagonal contribution
    */
    __device__ __host__ void ZeroDiffCorrelation(const size_t* det, ElemT WeightI)
    {
        for (int i = 0; i < 2 * this->norbs; i++) {
            if (this->getocc(det, i)) {
                int oi = i / 2;
                int si = i % 2;
                atomicAdd(onebody + si * onebody_size + oi + this->norbs * oi, Conjugate(WeightI) * WeightI);
                for (int j = i + 1; j < 2 * this->norbs; j++) {
                    if (this->getocc(det, j)) {
                        int oj = j / 2;
                        int sj = j % 2;
                        atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oi + this->norbs * oj + this->norbs * this->norbs * oi + this->norbs * this->norbs * this->norbs * oj), Conjugate(WeightI) * WeightI);
                        atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + this->norbs * oi + this->norbs * this->norbs * oj + this->norbs * this->norbs * this->norbs * oi), Conjugate(WeightI) * WeightI);
                        if (si == sj) {
                            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oi + this->norbs * oj + this->norbs * this->norbs * oj + this->norbs * this->norbs * this->norbs * oi), -Conjugate(WeightI) * WeightI);
                            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + this->norbs * oi + this->norbs * this->norbs * oi + this->norbs * this->norbs * this->norbs * oj), -Conjugate(WeightI) * WeightI);
                        }
                    }
                }
            }
        }
    }

    /**
        Function for adding one-occupation different contribution
    */
    __device__ __host__ void OneDiffCorrelation(const size_t* det,
                            const ElemT WeightI,
                            const ElemT WeightJ,
                            int i,
                            int a)
    {
        double sgn = 1.0;
        this->parity(det, std::min(i, a), std::max(i, a), sgn);
        int oi = i / 2;
        int si = i % 2;
        int oa = a / 2;
        int sa = a % 2;
        atomicAdd(onebody + si * onebody_size + (oi + this->norbs * oa), Conjugate(WeightI) * WeightJ * ElemT(sgn));
        size_t one = 1;
        for (int x = 0; x < this->D_size; x++) {
            size_t bits = det[x];
            for (int pos = 0; pos < this->bit_length; pos++) {
                if ((bits & 1ULL) == 1ULL) {
                    int soj = x * this->bit_length + pos;
                    int oj = soj / 2;
                    int sj = soj % 2;

                    atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + oj * this->norbs + oi * this->norbs * this->norbs + oj * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(sgn));
                    atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + oa * this->norbs + oj * this->norbs * this->norbs + oi * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(sgn));

                    if (si == sj) {
                        atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + oj * this->norbs + oj * this->norbs * this->norbs + oi * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(-sgn));
                        atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + oa * this->norbs + oi * this->norbs * this->norbs + oj * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(-sgn));
                    }
                }
                bits >>= 1;
            }
        }
    }

    /**
        Function for adding two-occupation different contribution
    */
    __device__ __host__ void TwoDiffCorrelation(const size_t* det,
                            const ElemT WeightI,
                            const ElemT WeightJ,
                            int i,
                            int j,
                            int a,
                            int b)
    {
        double sgn = 1.0;
        int I = std::min(i, j);
        int J = std::max(i, j);
        int A = std::min(a, b);
        int B = std::max(a, b);
        this->parity(det, std::min(I, A), std::max(I, A), sgn);
        this->parity(det, std::min(J, B), std::max(J, B), sgn);
        if (A > J || B < I)
            sgn *= -1.0;
        int oi = I / 2;
        int si = I % 2;
        int oa = A / 2;
        int sa = A % 2;
        int oj = J / 2;
        int sj = J % 2;
        int ob = B / 2;
        int sb = B % 2;

        if (si == sa) {
            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + this->norbs * ob + this->norbs * this->norbs * (oi + this->norbs * oj)), ElemT(sgn) * Conjugate(WeightI) * WeightJ);
            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (ob + this->norbs * oa + this->norbs * this->norbs * (oj + this->norbs * oi)), ElemT(sgn) * Conjugate(WeightI) * WeightJ);
        }

        if (si == sb) {
            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + this->norbs * ob + this->norbs * this->norbs * (oj + this->norbs * oi)), ElemT(-sgn) * Conjugate(WeightI) * WeightJ);
            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (ob + this->norbs * oa + this->norbs * this->norbs * (oi + this->norbs * oj)), ElemT(-sgn) * Conjugate(WeightI) * WeightJ);
        }
    }
};

}

#endif
