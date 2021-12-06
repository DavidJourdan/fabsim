// StVKElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#pragma once

#include "ElementBase.h"
#include "util/typedefs.h"

namespace fsim
{

class StVKElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the StVKElement class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &face, double thickness);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const;

  /**
   * Computes the Green strain tensor E = \frac 1 2 (F^T F - I)  where F is the deformation gradient
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Green strain
   */
  Eigen::Matrix2d strain(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu) const;

  static void setParameters(double young_modulus, double poisson_ratio);

  double coeff;
  Eigen::Matrix<double, 2, 2> _R;
};

} // namespace fsim
