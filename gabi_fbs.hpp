#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace gabi {

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

inline Mat3 skew3(const Vec3& x) {
  Mat3 S;
  S <<   0.0, -x.z(),  x.y(),
       x.z(),   0.0,  -x.x(),
      -x.y(),  x.x(),   0.0;
  return S;
}

Mat4 expSE3_fast(const Vec6& Y, double theta);
Vec6 Ad_mul_Twist(const Mat4& T, const Vec6& Y);
Vec6 ad_mul_Twist(const Vec6& V, const Vec6& S);
Vec6 minus_adT_mul(const Vec6& V, const Vec6& x);
Mat6 ad_mat(const Vec6& V);
Mat6 transformInertiaToWorld(const Mat4& C0, const Mat6& Mb);

// Output container (contiguous, prealloc-friendly)
struct GabiOut {
  // (6*N) x (rmaxMax+2): body j is rows 6j..6j+5, col k is V^(k)
  Eigen::MatrixXd Vder_flat;

  // N x (rmaxMax+3): col k is q^(k)
  Eigen::MatrixXd qder;

  // N x (rmaxMax+1): col r is q^(r+2)
  Eigen::MatrixXd qhigh;

  int rmax_alloc = -1;
  int rmax_used  = -1;

  void allocate(int N, int rmaxMax) {
    rmax_alloc = rmaxMax;
    rmax_used  = -1;
    Vder_flat.setZero(6*N, rmaxMax + 2);
    qder.setZero(N, rmaxMax + 3);
    qhigh.setZero(N, rmaxMax + 1);
  }
};

class GabiFBSRunner {
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

  // Wprop: 6 x (>= rmax+1)
  // tau_bar: N x (>= rmax+1)
  // Wapp_flat: (6*N) x (>= rmax+1)
  void compute(
      const Mat4& C0_1,
      const Vec6& V0_1,
      const std::vector<double>& q0,
      const std::vector<double>& qdot,
      const Eigen::Ref<const Eigen::MatrixXd>& Wprop,
      const Eigen::Ref<const Eigen::MatrixXd>& tau_bar,
      const Eigen::Ref<const Eigen::MatrixXd>& Wapp_flat,
      int rmax,
      double g,
      GabiOut* out);

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

  // binomials B[n,k] = C(n,k), n up to rmaxMax+1
  int Bdim_ = 0;
  std::vector<double> B_;
  inline double binom(int n, int k) const { return B_[n * Bdim_ + k]; }

  // workspace
  std::vector<Mat4> A0w_, Fw_, C0w_;
  std::vector<Mat6> M0w_, MAw_;
  std::vector<double> Dsc_;

  int K_S_ = 0; // rmaxMax+2
  int K_W_ = 0; // rmaxMax+1
  int K_M_ = 0; // rmaxMax+2

  std::vector<Vec6> Sder_, Vder_, WA_;
  inline Vec6& S(int j, int k) { return Sder_[j * K_S_ + k]; }
  inline const Vec6& S(int j, int k) const { return Sder_[j * K_S_ + k]; }
  inline Vec6& V(int j, int k) { return Vder_[j * K_S_ + k]; }
  inline const Vec6& V(int j, int k) const { return Vder_[j * K_S_ + k]; }
  inline Vec6& WA(int j, int k) { return WA_[j * K_W_ + k]; }
  inline const Vec6& WA(int j, int k) const { return WA_[j * K_W_ + k]; }

  std::vector<Mat6> Mder_;
  inline Mat6& M(int rr, int j) { return Mder_[rr * N_ + j]; }
  inline const Mat6& M(int rr, int j) const { return Mder_[rr * N_ + j]; }

  std::vector<Vec6> Vbias_, Pi_bias_;
  Eigen::MatrixXd qder_, qhigh_, qtil_;
  Eigen::LLT<Mat6> llt_;

  void buildBinom();
  void buildOrders();
  void resizeWorkspace();
};

} // namespace gabi
