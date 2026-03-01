#include "grne_id.hpp"
#include <stdexcept>
#include <algorithm>

namespace grne {

void GrneIdRunner::setModel(
    int N,
    const std::vector<std::vector<int>>& children,
    const std::vector<int>& parent,
    const Mat4& A0_1,
    const std::vector<Mat4>& Ap_j,
    const std::vector<Vec6>& Y,
    const std::vector<Mat6>& Mb,
    int rmaxMax) {

  if (N <= 0) throw std::runtime_error("N must be > 0");
  if ((int)children.size() != N) throw std::runtime_error("children size mismatch");
  if ((int)parent.size() != N) throw std::runtime_error("parent size mismatch");
  if ((int)Ap_j.size() != N) throw std::runtime_error("Ap_j size mismatch");
  if ((int)Y.size() != N) throw std::runtime_error("Y size mismatch");
  if ((int)Mb.size() != N) throw std::runtime_error("Mb size mismatch");
  if (rmaxMax < 0) throw std::runtime_error("rmaxMax must be >= 0");

  N_ = N;
  rmaxMax_ = rmaxMax;
  c_ = children;
  p_ = parent;
  A0_1_ = A0_1;
  Ap_ = Ap_j;
  Y_  = Y;
  Mb_ = Mb;

  buildOrders();
  buildBinom();
  resizeWorkspace();
}

void GrneIdRunner::buildOrders() {
  preOrder_.clear();
  postOrder_.clear();

  std::vector<int> stk;
  stk.reserve(N_);
  stk.push_back(0);
  while (!stk.empty()) {
    int j = stk.back(); stk.pop_back();
    preOrder_.push_back(j);
    const auto& kids = c_[j];
    for (int i=(int)kids.size()-1;i>=0;--i) stk.push_back(kids[i]);
  }

  std::vector<int> stk1, stk2;
  stk1.reserve(N_); stk2.reserve(N_);
  stk1.push_back(0);
  while (!stk1.empty()) {
    int j = stk1.back(); stk1.pop_back();
    stk2.push_back(j);
    for (int k : c_[j]) stk1.push_back(k);
  }
  postOrder_.assign(stk2.rbegin(), stk2.rend());
}

void GrneIdRunner::buildBinom() {
  const int nMax = rmaxMax_ + 1;
  Bdim_ = nMax + 3;
  B_.assign(Bdim_*Bdim_, 0.0);

  auto B = [&](int n, int k)->double& { return B_[n*Bdim_ + k]; };

  B(0,0) = 1.0;
  for (int n=1; n<=nMax+1; ++n) {
    B(n,0) = 1.0;
    B(n,n) = 1.0;
    for (int k=1; k<=n-1; ++k) B(n,k) = B(n-1,k-1) + B(n-1,k);
  }
}

void GrneIdRunner::resizeWorkspace() {
  K_S_ = rmaxMax_ + 2;
  K_M_ = rmaxMax_ + 2;
  K_W_ = rmaxMax_ + 1;

  A0w_.assign(N_, Mat4::Identity());
  Fw_.assign(N_,  Mat4::Identity());
  C0w_.assign(N_, Mat4::Identity());

  Sder_.assign(N_*K_S_, Vec6::Zero());
  Vder_.assign(N_*K_S_, Vec6::Zero());

  Mder_.assign(K_M_*N_, Mat6::Zero());
  Pider_.assign(K_M_*N_, Vec6::Zero());
  Wgrav_.assign(K_W_*N_, Vec6::Zero());
  Wder_.assign(K_W_*N_, Vec6::Zero());
}

void GrneIdRunner::compute(
    const Mat4& C0_1,
    const Eigen::Ref<const Eigen::MatrixXd>& V0_1_derivs,
    const Eigen::Ref<const Eigen::MatrixXd>& q_derivs,
    const Eigen::Ref<const Eigen::MatrixXd>& Wapp_flat,
    int r,
    double g,
    IdOut* out) {

  if (!out) throw std::runtime_error("out is null");
  if (r < 0 || r > rmaxMax_) throw std::runtime_error("r out of range");

  if (V0_1_derivs.rows()!=6 || V0_1_derivs.cols() < (r+2)) throw std::runtime_error("V0_1_derivs size mismatch");
  if (q_derivs.rows()!=N_  || q_derivs.cols()  < (r+3)) throw std::runtime_error("q_derivs size mismatch");
  if (Wapp_flat.rows()!=6*N_ || Wapp_flat.cols() < (r+1)) throw std::runtime_error("Wapp_flat size mismatch");

  if (out->rmax_alloc != rmaxMax_ || out->Qjoint.rows()!=N_) out->allocate(N_, rmaxMax_);
  out->r_used = r;

  // we overwrite all needed entries => no need to clear whole buffers
  out->Qjoint.row(0).setZero(); // base row unused

  const Vec6 G0 = (Vec6() << 0,0,0, 0,0,-g).finished();

  // -----------------------
  // Forward: poses + S^(0..r+1), V^(0..r+1)
  // -----------------------
  A0w_[0] = A0_1_;
  Fw_[0]  = C0_1;
  C0w_[0] = Fw_[0] * A0w_[0];

  for (int k=0;k<=r+1;++k) {
    V(0,k) = V0_1_derivs.col(k);
    S(0,k).setZero();
  }

  for (size_t ii=1; ii<preOrder_.size(); ++ii) {
    const int j  = preOrder_[ii];
    const int pj = p_[j];

    A0w_[j] = A0w_[pj] * Ap_[j];
    const double q0 = q_derivs(j,0);
    Fw_[j]  = Fw_[pj] * expSE3_fast(Y_[j], q0);
    C0w_[j] = Fw_[j] * A0w_[j];

    S(j,0) = Ad_mul_Twist(Fw_[j], Y_[j]);
    V(j,0) = V(pj,0) + S(j,0) * q_derivs(j,1);

    for (int k=1; k<=r+1; ++k) {
      Vec6 Sk; Sk.setZero();
      for (int a=0; a<=k-1; ++a) {
        const double coeff = binom(k-1, a);
        Sk += coeff * ad_mul_Twist(V(j,a), S(j, k-1-a));
      }
      S(j,k) = Sk;

      Vec6 Vk = V(pj,k);
      for (int a=0; a<=k; ++a) {
        const double coeff = binom(k, a);
        const int qord = k - a + 1; // q^(k-a+1)
        Vk += coeff * (S(j,a) * q_derivs(j, qord));
      }
      V(j,k) = Vk;
    }
  }

  // -----------------------
  // Per-body dynamics: M^(0..r+1), Pi^(0..r+1), Wgrav^(0..r)
  // -----------------------
  for (int j=0;j<N_;++j) {
    M(j,0) = transformInertiaToWorld(C0w_[j], Mb_[j]);

    for (int rr=1; rr<=r+1; ++rr) {
      Mat6 Mrr = Mat6::Zero();
      for (int k=0; k<=rr-1; ++k) {
        const double coeff = binom(rr-1, k);
        const Mat6& Mprev  = M(j, rr-1-k);
        const Mat6 A = ad_mat(V(j,k));
        Mrr -= coeff * (Mprev*A + A.transpose()*Mprev);
      }
      M(j,rr) = Mrr;
    }

    for (int m=0; m<=r+1; ++m) {
      Vec6 Pim; Pim.setZero();
      for (int k=0; k<=m; ++k) {
        const double coeff = binom(m, k);
        Pim += coeff * ( M(j, m-k) * V(j,k) );
      }
      Pi(j,m) = Pim;
    }

    for (int k=0;k<=r;++k) Wg(j,k) = M(j,k) * G0;
  }

  // -----------------------
  // Backward: W^(0..r)
  // -----------------------
  for (int jj : postOrder_) {
    for (int k=0;k<=r;++k) {
      Vec6 sumChild; sumChild.setZero();
      for (int ch : c_[jj]) sumChild += W(ch,k);

      const Vec6 Pi_kp1 = Pi(jj, k+1);
      const Vec6 Wapp_k = Wapp_flat.block<6,1>(6*jj, k);

      W(jj,k) = Pi_kp1 - Wapp_k - Wg(jj,k) + sumChild;
    }
  }

  // -----------------------
  // Project generalized forces
  // -----------------------
  for (int k=0;k<=r;++k) out->Qbase.col(k) = W(0,k);

  for (int j=1;j<N_;++j) {
    for (int rr=0; rr<=r; ++rr) {
      double qrr = 0.0;
      for (int k=0; k<=rr; ++k) {
        const double coeff = binom(rr, k);
        qrr += coeff * ( S(j, rr-k).dot( W(j,k) ) );
      }
      out->Qjoint(j, rr) = qrr;
    }
  }
}

} // namespace grne
