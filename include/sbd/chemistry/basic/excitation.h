/**
@file sbd/chemistry/basic/excitation.h
@brief function to find excitation from a determinant
 */
#ifndef SBD_CHEMISTRY_BASIC_EXCITATION_H
#define SBD_CHEMISTRY_BASIC_EXCITATION_H
namespace sbd {

  void single_from_hdet(const std::vector<size_t> & hdet_base,
			size_t bit_length,
			size_t norb,
			size_t num_closed,
			const std::vector<int> & open_base,
			const std::vector<int> & closed_base,
			std::vector<std::vector<size_t>> & hdet_ex) {
    // supporsed that open and closed are obtained priory by using getOpenClosed function for hdet_base.
    size_t num_ex = num_closed * (norb - num_closed);
    hdet_ex.resize(num_ex);
    std::vector<size_t> base = hdet_base;
    size_t ex_count = 0;
    for(size_t j=0; j < num_closed; j++) {
      setocc(base,bit_length,closed_base[j],false);
      for(size_t k=0; k < norb-num_closed; k++) {
	setocc(base,bit_length,open_base[k],true);
	hdet_ex[ex_count++] = base;
	setocc(base,bit_length,open_base[k],false);
      }
      setocc(base,bit_length,closed_base[j],true);
    }
  }
  
  void single_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			std::vector<std::vector<size_t>> & edet) {
    std::vector<int> open_base(norb);
    std::vector<int> closed_base(norb);
    int nc = getOpenClosed(hdet,bit_length,norb,open_base,closed_base);
    size_t numc = static_cast<size_t>(nc);
    single_from_hdet(hdet,bit_length,norb,numc,open_base,closed_base,edet);
  }
  
  void double_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			size_t num_closed,
			const std::vector<int> & open_base,
			const std::vector<int> & closed_base,
			std::vector<std::vector<size_t>> & edet) {
    // supporsed that open and closed are obtained priory by using getOpenClosed function for hdet_base.
    size_t num_ex = num_closed * (num_closed-1) * (norb - num_closed) * (norb - num_closed-1) / 4;
    edet.resize(num_ex);
    size_t ex_count = 0;
    std::vector<size_t> base = hdet;
    for(size_t i=0; i < num_closed; i++) {
      setocc(base,bit_length,closed_base[i],false);
      for(size_t j=i+1; j < num_closed; j++) {
	setocc(base,bit_length,closed_base[j],false);
	for(size_t k=0; k < norb-num_closed; k++) {
	  setocc(base,bit_length,open_base[k],true);
	  for(size_t l=k+1; l < norb-num_closed; l++) {
	    setocc(base,bit_length,open_base[l],true);
	    edet[ex_count++] = base;
	    setocc(base,bit_length,open_base[l],false);
	  }
	  setocc(base,bit_length,open_base[k],false);
	}
	setocc(base,bit_length,closed_base[j],true);
      }
      setocc(base,bit_length,closed_base[i],true);
    }
  }
  
  void double_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			std::vector<std::vector<size_t>> & edet) {
    std::vector<int> open_base(norb);
    std::vector<int> closed_base(norb);
    int nc = getOpenClosed(hdet,bit_length,norb,open_base,closed_base);
    size_t numc = static_cast<size_t>(nc);
    double_from_hdet(hdet,bit_length,norb,numc,open_base,closed_base,edet);
  }
  
}

#endif
