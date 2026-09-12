#pragma once
#include "sbd/caop/stat/stat_evaluator.h"
#include "sbd/caop/basic/restart.h"
#include "sbd/caop/basic/arithmetic.h"
#include <fstream>
#include <iostream>
#include <array>

namespace cs = sbd::ca_stat;
using Dets = sbd::det_vector<std::size_t>;
struct Factor { bool create; int q; };
struct Term { double c; std::vector<Factor> ops; };
void near(double x, double y) {
  if(!std::isfinite(x) || !std::isfinite(y) || std::abs(x-y)>1e-11*std::max({1.0,std::abs(x),std::abs(y)}))
    throw std::runtime_error("numeric mismatch: "+std::to_string(x)+" vs "+std::to_string(y));
}
// Independent small Fock-space application, without SBD masks/sign helpers.
std::array<double,16> column(int ket, const std::vector<Term>& terms, bool fermion) {
  std::array<double,16> result{};
  for(const auto& term: terms) {
    int state=ket; double v=term.c;
    for(auto op=term.ops.rbegin(); op!=term.ops.rend(); ++op) {
      bool occupied = (state>>op->q)&1;
      if(occupied==op->create) { v=0; break; }
      if(fermion) for(int k=0;k<op->q;++k) if((state>>k)&1) v=-v;
      state ^= 1<<op->q;
    }
    result[state]+=v;
  }
  return result;
}
sbd::GeneralOp<double> build(const std::vector<Term>& terms, bool sign) {
  sbd::GeneralOp<double> h;
  for(const auto& t:terms) {
    sbd::GeneralOp<double> term(sbd::ProductOp{});
    for(const auto& op:t.ops) term *= op.create ? sbd::Cr(op.q) : sbd::An(op.q);
    h += t.c*term;
  }
  sbd::NormalOrdering(h,sign); sbd::Simplify(h); return h;
}
