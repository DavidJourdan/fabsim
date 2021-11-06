// RodStencil.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/21/20

#pragma once

#include "ElasticRod.h"
#include "ElementBase.h"
#include "util/typedefs.h"

namespace fsim
{

class RodStencil : public ElementBase<3, 2>
{
public:
  RodStencil(const Eigen::Ref<const Mat3<double>> V,
             const ElasticRod::LocalFrame &f1,
             const ElasticRod::LocalFrame &f2,
             const Eigen::Matrix<int, 5, 1> &dofs,
             const Eigen::Vector2d &widths,
             double young_modulus);
  double energy(const Eigen::Ref<const Eigen::VectorXd> X,
                const ElasticRod::LocalFrame &f1,
                const ElasticRod::LocalFrame &f2) const;
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                       const ElasticRod::LocalFrame &f1,
                       const ElasticRod::LocalFrame &f2) const;
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X,
                      const ElasticRod::LocalFrame &f1,
                      const ElasticRod::LocalFrame &f2) const;

  double get_reference_twist() const { return _ref_twist; }
  void set_reference_twist(double ref_twist) { _ref_twist = ref_twist; }
  void update_reference_twist(const ElasticRod::LocalFrame &f1, const ElasticRod::LocalFrame &f2);

  void set_curvatures(const Eigen::Vector2d &restK) { _restK = restK; }
  Eigen::Vector2d get_curvatures() { return _restK; }

  void set_stiffness(const Eigen::Vector2d &stiff) { _stiffnesses = stiff; }
  Eigen::Vector2d &stiffness() { return _stiffnesses; }

protected:
  Eigen::DiagonalMatrix<double, 2> bend_matrix() const { return Eigen::DiagonalMatrix<double, 2>(_stiffnesses); }
  double twist_coeff() const { return _stiffnesses.sum(); }

  Eigen::Vector2d material_curvature(const Eigen::Ref<const Eigen::VectorXd> X,
                                     const ElasticRod::LocalFrame &f1,
                                     const ElasticRod::LocalFrame &f2) const;
  Eigen::Matrix<double, 11, 2> material_curvature_derivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                             const ElasticRod::LocalFrame &f1,
                                                             const ElasticRod::LocalFrame &f2,
                                                             Eigen::Matrix<double, 11, 11> *dderiv = nullptr) const;

  double twist_angle(const Eigen::Ref<const Eigen::VectorXd> X) const;
  Eigen::Matrix<double, 11, 1> twist_angle_derivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                      const ElasticRod::LocalFrame &f1,
                                                      const ElasticRod::LocalFrame &f2,
                                                      Eigen::Matrix<double, 11, 11> *dderiv = nullptr) const;

private:
  double _vertex_length;
  Eigen::Vector2d _stiffnesses;
  Eigen::Vector2d _restK; // rest material curvature (product of curvature binormal and mean material director)
  double _ref_twist;
};

// matrix of material directors
Eigen::Matrix<double, 2, 3> material_matrix(const ElasticRod::LocalFrame &f, double theta);

} // namespace fsim
