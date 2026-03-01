#include "gabi_fbs.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace gabi {

Mat4 expSE3_fast(const Vec6& Y, double theta) {
  const Vec3 w = Y.head<3>();
  const Vec3 v = Y.tail<3>();
  const double wn = w.norm();

  Mat3 R = Mat3::Identity();
  Vec3 p = Vec3::Zero();

  if (wn < 1e-12) {
    p = v * theta;
  } else {
    const double phi = wn * theta;
    const double s = std::sin(phi);
    const double c = std::cos(phi);

    const Mat3 What  = skew3(w);
    const Mat3 What2 = What * What;

    R = Mat3::Identity()
      + (s/wn) * What
      + ((1.0 - c)/(wn*wn)) * What2;

    const double A = (1.0 - c) / (wn*wn);
    const double B = (phi - s) / (wn*wn*wn);

    const Mat3 Vmat = Mat3::Identity()*theta + A*What + B*What2;
    p = Vmat * v;
  }

  Mat4 T = Mat4::Identity();
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = p;
  return T;
}

Vec6 Ad_mul_Twist(const Mat4& T, const Vec6& Y) {
  const Mat3 R = T.block<3,3>(0,0);
  const Vec3 p = T.block<3,1>(0,3);
  const Vec3 w = Y.head<3>();
  const Vec3 v = Y.tail<3>();

  const Vec3 Rw = R*w;
  Vec6 S;
  S.head<3>() = Rw;
  S.tail<3>() = p.cross(Rw) + R*v;
  return S;
}

Vec6 ad_mul_Twist(const Vec6& V, const Vec6& S) {
  const Vec3 w  = V.head<3>();
  const Vec3 v  = V.tail<3>();
  const Vec3 Sw = S.head<3>();
  const Vec3 Sv = S.tail<3>();

  Vec6 y;
  y.head<3>() = w.cross(Sw);
  y.tail<3>() = v.cross(Sw) + w.cross(Sv);
  return y;
}

Vec6 minus_adT_mul(const Vec6& V, const Vec6& x) {
  const Vec3 w  = V.head<3>();
  const Vec3 v  = V.tail<3>();
  const Vec3 xw = x.head<3>();
  const Vec3 xv = x.tail<3>();

  Vec6 y;
  y.head<3>() = w.cross(xw) + v.cross(xv);
  y.tail<3>() = w.cross(xv);
  return y;
}

Mat6 ad_mat(const Vec6& V) {
  const Vec3 w = V.head<3>();
  const Vec3 v = V.tail<3>();

  Mat6 A = Mat6::Zero();
  A.block<3,3>(0,0) = skew3(w);
  A.block<3,3>(3,0) = skew3(v);
  A.block<3,3>(3,3) = skew3(w);
  return A;
}

Mat6 transformInertiaToWorld(const Mat4& C0, const Mat6& Mb) {
  const Mat3 R = C0.block<3,3>(0,0);
  const Vec3 p = C0.block<3,1>(0,3);

  const Mat3 Rinv = R.transpose();
  const Vec3 pinv = -Rinv*p;

  Mat6 Ad = Mat6::Zero();
  Ad.block<3,3>(0,0) = Rinv;
  Ad.block<3,3>(3,0) = skew3(pinv) * Rinv;
  Ad.block<3,3>(3,3) = Rinv;

  return Ad.transpose() * Mb * Ad;
}

void GabiFBSRunner::setModel(
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
  c_ = children;
  p_ = parent;
  A0_1_ = A0_1;
  Ap_ = Ap_j;
  Y_ = Y;
  Mb_ = Mb;
  rmaxMax_ = rmaxMax;

  buildOrders();
  buildBinom();
  resizeWorkspace();
}

void GabiFBSRunner::buildOrders() {
  preOrder_.clear();
  postOrder_.clear();

  std::vector<int> stk;
  stk.reserve(N_);
  stk.push_back(0);
  while (!stk.empty()) {
    int j = stk.back(); stk.pop_back();
    preOrder_.push_back(j);
    const auto& kids = c_[j];
    for (int i = (int)kids.size()-1; i >= 0; --i) stk.push_back(kids[i]);
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

void GabiFBSRunner::buildBinom() {
  const int nMax = rmaxMax_ + 1;
  Bdim_ = nMax + 3;
  B_.assign(Bdim_ * Bdim_, 0.0);

  auto B = [&](int n, int k) -> double& { return B_[n * Bdim_ + k]; };

  B(0,0) = 1.0;
  for (int n = 1; n <= nMax+1; ++n) {
    B(n,0) = 1.0;
    B(n,n) = 1.0;
    for (int k = 1; k <= n-1; ++k) B(n,k) = B(n-1,k-1) + B(n-1,k);
  }
}

void GabiFBSRunner::resizeWorkspace() {
  K_S_ = rmaxMax_ + 2;
  K_W_ = rmaxMax_ + 1;
  K_M_ = rmaxMax_ + 2;

  A0w_.assign(N_, Mat4::Identity());
  Fw_.assign(N_,  Mat4::Identity());
  C0w_.assign(N_, Mat4::Identity());

  M0w_.assign(N_, Mat6::Zero());
  MAw_.assign(N_, Mat6::Zero());
  Dsc_.assign(N_, 0.0);

  Sder_.assign(N_ * K_S_, Vec6::Zero());
  Vder_.assign(N_ * K_S_, Vec6::Zero());
  WA_.assign(N_ * K_W_,   Vec6::Zero());

  Mder_.assign(K_M_ * N_, Mat6::Zero());

  Vbias_.assign(N_, Vec6::Zero());
  Pi_bias_.assign(N_, Vec6::Zero());

  qder_.resize(N_, rmaxMax_ + 3);
  qhigh_.resize(N_, rmaxMax_ + 1);
  qtil_.resize(N_, rmaxMax_ + 1);
}

void GabiFBSRunner::compute(
    const Mat4& C0_1,
    const Vec6& V0_1,
    const std::vector<double>& q0,
    const std::vector<double>& qdot,
    const Eigen::Ref<const Eigen::MatrixXd>& Wprop,
    const Eigen::Ref<const Eigen::MatrixXd>& tau_bar,
    const Eigen::Ref<const Eigen::MatrixXd>& Wapp_flat,
    int rmax,
    double g,
    GabiOut* out) {

  if (!out) throw std::runtime_error("out is null");
  if (rmax < 0 || rmax > rmaxMax_) throw std::runtime_error("rmax out of range");
  if ((int)q0.size() != N_ || (int)qdot.size() != N_) throw std::runtime_error("q0/qdot size mismatch");
  if (Wprop.rows() != 6 || Wprop.cols() < rmax+1) throw std::runtime_error("Wprop size mismatch");
  if (tau_bar.rows() != N_ || tau_bar.cols() < rmax+1) throw std::runtime_error("tau_bar size mismatch");
  if (Wapp_flat.rows() != 6*N_ || Wapp_flat.cols() < rmax+1) throw std::runtime_error("Wapp_flat size mismatch");

  if (out->rmax_alloc != rmaxMax_ || out->qder.rows() != N_ || out->Vder_flat.rows() != 6*N_) {
    out->allocate(N_, rmaxMax_);
  }
  out->rmax_used = rmax;

  const Vec6 G0 = (Vec6() << 0,0,0, 0,0,-g).finished();

  // Clear only needed parts
  qder_.leftCols(rmax+3).setZero();
  qhigh_.leftCols(rmax+1).setZero();
  qtil_.leftCols(rmax+1).setZero();

  // q^(0), q^(1)
  for (int j=0;j<N_;++j) {
    qder_(j,0) = q0[j];
    qder_(j,1) = qdot[j];
  }

  const int ScolsNeeded = rmax + 2; // S^(0..rmax+1)
  const int WcolsNeeded = rmax + 1; // WA^(0..rmax)

  // clear only needed Vec6 entries
  for (int i=0;i<N_*ScolsNeeded;++i) { Sder_[i].setZero(); Vder_[i].setZero(); }
  for (int i=0;i<N_*WcolsNeeded;++i) { WA_[i].setZero(); }

  // Forward pose + S^(0) + V^(0)
  A0w_[0] = A0_1_;
  Fw_[0]  = C0_1;
  C0w_[0] = Fw_[0] * A0w_[0];
  S(0,0).setZero();
  V(0,0) = V0_1;

  for (size_t ii = 1; ii < preOrder_.size(); ++ii) {
    const int j  = preOrder_[ii];
    const int pj = p_[j];

    A0w_[j] = A0w_[pj] * Ap_[j];
    Fw_[j]  = Fw_[pj]  * expSE3_fast(Y_[j], q0[j]);
    C0w_[j] = Fw_[j] * A0w_[j];

    S(j,0) = Ad_mul_Twist(Fw_[j], Y_[j]);
    V(j,0) = V(pj,0) + S(j,0) * qdot[j];
  }

  // M0 and M^(0)
  for (int j = 0; j < N_; ++j) {
    M0w_[j] = transformInertiaToWorld(C0w_[j], Mb_[j]);
    M(0,j)  = M0w_[j];
  }

  // MA (postorder)
  for (int jj : postOrder_) {
    Mat6 MAj = M0w_[jj];
    for (int i : c_[jj]) {
      const Vec6& Si = S(i,0);
      const Mat6& Mi = MAw_[i];
      const Vec6 Ui  = Mi * Si;
      const double Di = Si.dot(Ui);
      Dsc_[i] = Di;
      MAj += (Mi - (Ui * (Ui.transpose() / Di)));
    }
    MAw_[jj] = MAj;
  }

  llt_.compute(MAw_[0]);
  if (llt_.info() != Eigen::Success) throw std::runtime_error("LLT failed (MA base not SPD?)");

  int Smax = 0;
  int Mmax = 0;

  for (int r = 0; r <= rmax; ++r) {
    // ensure S up to r+1
    const int kNeed = r + 1;
    for (int k = Smax+1; k <= kNeed; ++k) {
      for (size_t ii = 1; ii < preOrder_.size(); ++ii) {
        const int j = preOrder_[ii];
        Vec6 Sk; Sk.setZero();
        for (int l = 0; l <= k-1; ++l) {
          const double coeff = binom(k-1, l);
          Sk += coeff * ad_mul_Twist(V(j,l), S(j, k-1-l));
        }
        S(j,k) = Sk;
      }
    }
    Smax = std::max(Smax, kNeed);

    // ensure M up to r+1 if r>0
    if (r > 0) {
      const int rrNeed = r + 1;
      for (int rr = Mmax+1; rr <= rrNeed; ++rr) {
        for (int j = 0; j < N_; ++j) {
          Mat6 Mrr = Mat6::Zero();
          for (int k = 0; k <= rr-1; ++k) {
            const double coeff = binom(rr-1, k);
            const Mat6& Mprev  = M(rr-1-k, j);
            const Mat6 A = ad_mat(V(j,k));
            Mrr -= coeff * (Mprev*A + A.transpose()*Mprev);
          }
          M(rr,j) = Mrr;
        }
      }
      Mmax = std::max(Mmax, rrNeed);
    }

    // Vbias
    Vbias_[0].setZero();
    for (size_t ii = 1; ii < preOrder_.size(); ++ii) {
      const int j = preOrder_[ii];
      Vec6 vb; vb.setZero();
      for (int a = 1; a <= r+1; ++a) {
        const double coeff = binom(r+1, a);
        const int qord = r - a + 2;
        vb += coeff * (S(j,a) * qder_(j, qord));
      }
      Vbias_[j] = vb;
    }

    // Pi_bias
    if (r > 0) {
      const int m = r + 1;
      for (int j = 0; j < N_; ++j) {
        Vec6 pib; pib.setZero();
        for (int k = 0; k <= r; ++k) {
          const double coeff = binom(m, k);
          pib += coeff * (M(m-k, j) * V(j,k));
        }
        Pi_bias_[j] = pib;
      }
    }

    // Backward WA
    for (int jj : postOrder_) {
      const Vec6 V0j = V(jj,0);

      if (r == 0) {
        const Vec6 MV = M0w_[jj] * V0j;
        Vec6 WAj = minus_adT_mul(V0j, MV)
                 - Wapp_flat.block<6,1>(6*jj, 0)
                 - (M0w_[jj] * G0);

        for (int i : c_[jj]) {
          const Vec6& Si  = S(i,0);
          const Vec6& Sdi = S(i,1);
          const Mat6& MiA = MAw_[i];
          const Vec6  WiA = WA(i,0);
          const double Di = Dsc_[i];
          const double qd = qder_(i,1);

          const double qtil_i = (tau_bar(i,0) - Si.dot(MiA*(Sdi*qd) + WiA)) / Di;
          qtil_(i,0) = qtil_i;
          WAj += WiA + MiA*(Si*qtil_i + Sdi*qd);
        }
        WA(jj,0) = WAj;

      } else {
        const Mat6& Mr = M(r, jj);
        Vec6 WAjr = -Wapp_flat.block<6,1>(6*jj, r) - (Mr * G0) + Pi_bias_[jj];

        for (int i : c_[jj]) {
          const Mat6& MiA = MAw_[i];
          const Vec6& Si  = S(i,0);
          const double Di = Dsc_[i];
          const Vec6& Vb  = Vbias_[i];

          double tau_tilde = 0.0;
          for (int k = 0; k <= r-1; ++k) {
            const double coeff = binom(r, k);
            tau_tilde += coeff * ( S(i, r-k).dot(MiA*V(i,k+1) + WA(i,k)) );
          }

          const double qtil_i = (tau_bar(i,r) - tau_tilde - Si.dot(MiA*Vb + WA(i,r))) / Di;
          qtil_(i,r) = qtil_i;

          WAjr += WA(i,r) + MiA*(Si*qtil_i + Vb);
        }
        WA(jj,r) = WAjr;
      }
    }

    // Forward solve
    V(0, r+1) = llt_.solve(Wprop.block<6,1>(0, r) - WA(0,r));

    for (size_t ii = 1; ii < preOrder_.size(); ++ii) {
      const int j = preOrder_[ii];
      const int pj = p_[j];

      const Vec6 Vp = V(pj, r+1);
      const Vec6& Sj = S(j,0);
      const double Di = Dsc_[j];

      const double qhigh_j = -(Sj.dot(MAw_[j] * Vp))/Di + qtil_(j,r);
      qhigh_(j,r) = qhigh_j;
      V(j, r+1) = Vp + Vbias_[j] + Sj*qhigh_j;
    }

    qder_.col(r+2) = qhigh_.col(r);
  }

  // Pack outputs (no resize), only needed columns
  for (int j=0;j<N_;++j) {
    for (int k=0;k<=rmax+1;++k) {
      out->Vder_flat.block<6,1>(6*j, k) = V(j,k);
    }
  }
  out->qder.leftCols(rmax+3)  = qder_.leftCols(rmax+3);
  out->qhigh.leftCols(rmax+1) = qhigh_.leftCols(rmax+1);
}

} // namespace gabi
