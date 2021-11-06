// ElasticRod.h
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Please note that, at the moment this implementation assumes a rectangular cross-section whose
// dimensions are given by the normal and binormal widths variables
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#pragma once

#include "RodStencil.h"
#include "Spring.h"
#include "util/typedefs.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>

namespace fsim
{

class ElasticRod
{
public:
  ElasticRod(const Eigen::Ref<const Mat3<double>> V,
             const Eigen::Ref<const Eigen::VectorXi> indices,
             const Eigen::Vector3d &N,
             const Eigen::Ref<const Eigen::VectorXd> normal_widths,
             const Eigen::Ref<const Eigen::VectorXd> binormal_widths,
             double young_modulus,
             double incompressibility);

  ElasticRod(const Eigen::Ref<const Mat3<double>> V,
             const Eigen::Ref<const Eigen::VectorXi> indices,
             const Eigen::Vector3d N,
             double w_n,
             double w_b,
             double young_modulus,
             double incompressibility);
  ElasticRod() = default;

  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * hessian of the energy  \nabla^2 f : \R^n -> \R^{n \times n}
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of the energy stored in a sparse matrix representation
   */
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessian_triplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

  void update_properties(const Eigen::Ref<const Eigen::VectorXd> X);

  void get_reference_directors(Mat3<double> &D1, Mat3<double> &D2) const;
  void get_rotated_directors(const Eigen::Ref<const Eigen::VectorXd> theta,
                             Mat3<double> &P1,
                             Mat3<double> &P2) const;

  std::vector<RodStencil> const &rod_stencils() const { return _stencils; };
  std::vector<RodStencil> &rod_stencils() { return _stencils; };

  std::vector<Spring<true>> const &springs() const { return _springs; }
  std::vector<Spring<true>> &springs() { return _springs; }

  Spring<true> const &get_spring(int id) const { return _springs[id - 3 * nV]; };

  /**
   * Returns a LocalFrame, uniquely identified by the index of the rotational degree of freedom corresponding to its
   * twist, oriented from vertex x0 to vertex x1
   * @param X a flat vector stacking all degrees of freedom
   * @param x0 index of the 'from' vertex
   * @param x1 index of the 'to' vertex
   * @param id index of the corresponding rotational degree of freedom
   * @return
   */
  LocalFrame<double> get_frame(const Eigen::Ref<const Eigen::VectorXd> X, int x0, int x1, int id) const;

  int nb_edges() const { return _springs.size(); }

  /**
   * Computes the Bishop frame, which is the geometrically most relaxed frame attached to the centerline of a curve
   * @param V  nV by 3 list of vertex positions, including but not limited to rod vertex positions
   * @param E  nE by 2 list of rod vertex indices into V
   * @param n  normal of the first edge (the subsequent ones are defined by parallel transport)
   * if n isn't orthogonal to the first edge, the direction chosen to be the normal will be as close as possible to n
   * @return P1 and P2  nE by 3 lists of frame directors
   */
  static void bishop_frame(const Eigen::Ref<const Mat3<double>> V,
                           const Eigen::Ref<const Eigen::VectorXi> E,
                           const Eigen::Vector3d &n,
                           Mat3<double> &P1,
                           Mat3<double> &P2);

  /**
   * Returns curvature binormals
   * @param P  nP by 3 list of vertex positions, including but not limited to rod vertex positions
   * @param E  nE + 1 by 1 list of rod vertex indices into P
   * @return KB nE - 1 by 3 list of curvature binormals
   */
  static Mat3<double> curvature_binormals(const Eigen::Ref<const Mat3<double>> P,
                                                const Eigen::Ref<const Eigen::VectorXi> E);

protected:
  mutable std::vector<RodStencil> _stencils;
  std::vector<Spring<true>> _springs;
  std::vector<LocalFrame<double>> _frames;
  int nV; // total number of vertices in the simulation (includes non-rod vertices)
  double _stretch_modulus;
};

} // namespace fsim
