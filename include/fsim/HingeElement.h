// HingeElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/13/20

#pragma once

#include "ElementBase.h"
#include "util/typedefs.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>

namespace fsim
{

class TanAngleFormulation;
class SquaredAngleFormulation;

/**
 * Stencil for hinge-based shell energies such as Discrete Shells
 * @tparam HingeFormulation  implementation for the shell energy: either (\theta - \bar\theta)^2
 * or (2 \tan\theta/2 - 2 \tan\bar\theta/2)^2
 * @tparam full_hessian  whether or not ot compute the full hessian or a quadratic approximation
 */
template <class HingeFormulation = TanAngleFormulation, bool full_hessian = true>
class HingeElement : public ElementBase<4>
{
public:
  /**
   * Constructor for the HingeElement class
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param indices  list of 4 indices corresponding to 2 faces sharing an edge, first 2 indices are for the shared edge
   * @param coeff  coefficent which will be multiplied to the energy
   */
  HingeElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector4i &indices, double coeff);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of one hinge stencil
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of one hinge stencil, derivatives are stacked in the order of idx
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of one hinge stencil
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * computes the gradient of the bend angle (complement of the dihedral angle) between 2 neighboring faces
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param hessian  optional return parameter: derivative of the gradient
   */
  LocalVector bend_angle_gradient(const Eigen::Ref<const Mat3<double>> V, LocalMatrix *hessian = nullptr) const;

  double _coeff;
  HingeFormulation _hinge;
};

/**
 * SquaredAngleFormulation:
 * func computes (\theta - \bar\theta)^2 for a given "hinge", \theta being the angle between n_0 and n_1
 * deriv computes its first and second derivatives w.r.t \theta
 */
class SquaredAngleFormulation
{
public:
  SquaredAngleFormulation(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector4i &indices);
  SquaredAngleFormulation() : _rest_angle(0) {}
  double func(const Eigen::Vector3d &n0, const Eigen::Vector3d &n1, const Eigen::Vector3d &axis) const;
  double deriv(const Eigen::Vector3d &n0,
               const Eigen::Vector3d &n1,
               const Eigen::Vector3d &axis,
               double *second_deriv = nullptr) const;

private:
  double _rest_angle;
};

/**
 * TanAngleFormulation:
 * func computes (2 \tan\theta/2 - 2 \tan\bar\theta/2)^2 for a given "hinge", \theta being the angle between n_0 and n_1
 * deriv computes its first and second derivatives w.r.t \theta
 */
class TanAngleFormulation
{
public:
  TanAngleFormulation(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector4i &indices);
  TanAngleFormulation() : _rest_tangent(0) {}
  double func(const Eigen::Vector3d &n0, const Eigen::Vector3d &n1, const Eigen::Vector3d &axis) const;
  double deriv(const Eigen::Vector3d &n0,
               const Eigen::Vector3d &n1,
               const Eigen::Vector3d &axis,
               double *second_deriv = nullptr) const;

private:
  double _rest_tangent;
};

} // namespace fsim
