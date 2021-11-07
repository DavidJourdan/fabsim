// MassSpring.h
//
// A model for an elastic membrane. Uses a simple mass-spring system.
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 03/27/18

#pragma once
#include "Spring.h"
#include "util/typedefs.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>

namespace fsim
{

/**
 * A model for an elastic membrane. Uses a simple mass-spring system.
 * @tparam allow_compression  whether or not to allow compression in springs. If false, springs whose lengths is smaller
 * than their rest length will have no contribution to the energy and its derivatives
 */
template <bool allow_compression = true>
class MassSpringModel
{
public:
  /**
   * Constructor for the MassSpringModel class
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param young_modulus  a non-negative weight on the energy, tells how important the effect of this model will be
   * compared to other terms
   */
  MassSpringModel(const Eigen::Ref<const Mat3<double>> V,
             const Eigen::Ref<const Mat3<int>> F,
             double young_modulus);
  MassSpringModel() = default;

  /**
   * energy function of this material model f : \R^n -> \R
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

  void set_young_modulus(double young_modulus) { _young_modulus = young_modulus; }

  int nb_dofs() const { return 3 * nV; }

private:
  double _young_modulus;
  std::vector<Spring<allow_compression>> _springs;
  int nV;
};

using MassSpring = MassSpringModel<true>;

} // namespace fsim
