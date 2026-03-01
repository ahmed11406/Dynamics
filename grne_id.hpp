#pragma once
#include "se3_spatial.hpp"
#include <Eigen/Dense>
#include <vector>

namespace grne {

struct IdOut {
  // base wrench derivatives: 6 x (rmaxMax+1)
  Eigen::MatrixXd Qbase;
  // joint torque derivatives: N x (rmaxMax+1), row 0 is unused (kept 0)
  Eigen::MatrixXd Qjoint;

  int rmax_alloc = -1;
  int r_used     = -1;

  void allocate(int N, int rmaxMax) {
    rmax_alloc = rmaxMax;
    r_used = -1;
    Qbase.setZero(6, rmaxMax+1);
    Qjoint.setZero(N, rmaxMax+1);
  }
};

// Optimized GRNE-FBS inverse dynamics (floating base, spatial/right-invariant).
// 0-based indexing. No recursion, no allocations in compute().
class GrneIdRunner {
public:
  void setModel(
      int N,
      const std::vector<std::vector<int>>& children,
      const std::vector<int>& parent,
      const Mat4& A0_1,
      const std::vector<Mat4>& Ap_j,
      const std::vector<Vec6>& Y,
      const std::vector<Mat6>& Mb,
      int rmaxMax);

  // Inputs:
  //  C0_1          : base pose
  //  V0_1_derivs   : 6 x (>= r+2)  (V^(0..r+1))
  //  q_derivs      : N x (>= r+3)  (q^(0..r+2))
  //  Wapp_flat     : (6N) x (>= r+1), block rows 6j..6j+5 is Wapp_j^(k)
  //  r             : desired ID order
  //  g             : gravity
  void compute(
      const Mat4& C0_1,
      const Eigen::Ref<const Eigen::MatrixXd>& V0_1_derivs,
      const Eigen::Ref<const Eigen::MatrixXd>& q_derivs,
      const Eigen::Ref<const Eigen::MatrixXd>& Wapp_flat,
      int r,
      double g,
      IdOut* out);

private:
  int N_ = 0;
  int rmaxMax_ = 0;

  Mat4 A0_1_;
  std::vector<std::vector<int>> c_;
  std::vector<int> p_;
  std::vector<Mat4> Ap_;
  std::vector<Vec6> Y_;
  std::vector<Mat6> Mb_;

  std::vector<int> preOrder_, postOrder_;

  // binomials B[n,k] = C(n,k) up to n=rmaxMax+1
  int Bdim_ = 0;
  std::vector<double> B_;
  inline double binom(int n, int k) const { return B_[n*Bdim_ + k]; }

  // workspace sizes
  int K_S_ = 0; // S,V: 0..rmaxMax+1 => rmaxMax+2
  int K_M_ = 0; // M,Pi: 0..rmaxMax+1 => rmaxMax+2
  int K_W_ = 0; // W,Wgrav: 0..rmaxMax   => rmaxMax+1

  // poses
  std::vector<Mat4> A0w_, Fw_, C0w_;

  // kinematics derivatives
  std::vector<Vec6> Sder_, Vder_;
  inline Vec6& S(int j, int k) { return Sder_[j*K_S_ + k]; }
  inline const Vec6& S(int j, int k) const { return Sder_[j*K_S_ + k]; }
  inline Vec6& V(int j, int k) { return Vder_[j*K_S_ + k]; }
  inline const Vec6& V(int j, int k) const { return Vder_[j*K_S_ + k]; }

  // dynamics derivatives (stored by order-major for cache)
  std::vector<Mat6> Mder_;   // (r+2)*N
  std::vector<Vec6> Pider_;  // (r+2)*N
  std::vector<Vec6> Wgrav_;  // (r+1)*N
  std::vector<Vec6> Wder_;   // (r+1)*N

  inline Mat6& M(int j, int k) { return Mder_[k*N_ + j]; }      // k=0..r+1
  inline const Mat6& M(int j, int k) const { return Mder_[k*N_ + j]; }
  inline Vec6& Pi(int j, int k) { return Pider_[k*N_ + j]; }    // k=0..r+1
  inline const Vec6& Pi(int j, int k) const { return Pider_[k*N_ + j]; }
  inline Vec6& Wg(int j, int k) { return Wgrav_[k*N_ + j]; }    // k=0..r
  inline const Vec6& Wg(int j, int k) const { return Wgrav_[k*N_ + j]; }
  inline Vec6& W(int j, int k) { return Wder_[k*N_ + j]; }      // k=0..r
  inline const Vec6& W(int j, int k) const { return Wder_[k*N_ + j]; }

  void buildOrders();
  void buildBinom();
  void resizeWorkspace();
};

} // namespace grne
