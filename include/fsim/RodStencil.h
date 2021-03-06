// RodStencil.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/21/20

#pragma once

#include "ElementBase.h"
#include "LocalFrame.h"
#include "util/geometry.h"
#include "util/typedefs.h"

namespace fsim
{

class RodStencil : public ElementBase<3, 2>
{
public:
  RodStencil(const Eigen::Ref<const Mat3<double>> V,
             const LocalFrame &f1,
             const LocalFrame &f2,
             const Eigen::Matrix<int, 5, 1> &dofs);
  double energy(const Eigen::Ref<const Eigen::VectorXd> X,
                const LocalFrame &f1,
                const LocalFrame &f2,
                const Eigen::Vector2d &stiffnesses,
                double stretchCoeff,
                double mass = 0) const;
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                       const LocalFrame &f1,
                       const LocalFrame &f2,
                       const Eigen::Vector2d &stiffnesses,
                       double stretchCoeff,
                       double mass = 0) const;
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X,
                      const LocalFrame &f1,
                      const LocalFrame &f2,
                      const Eigen::Vector2d &stiffnesses,
                      double stretchCoeff,
                      double mass = 0) const;

  LocalMatrix hessianApprox(const Eigen::Ref<const Eigen::VectorXd> X,
                            const LocalFrame &f1,
                            const LocalFrame &f2,
                            const Eigen::Vector2d &stiffnesses,
                            double stretchCoeff,
                            double mass = 0) const;

  double getReferenceTwist() const { return _ref_twist; }
  void setReferenceTwist(double ref_twist) { _ref_twist = ref_twist; }
  void updateReferenceTwist(const LocalFrame &f1, const LocalFrame &f2);

  void setCurvature(const Eigen::Vector2d &restK) { _restK = restK; }
  Eigen::Vector2d getCurvature() { return _restK; }

protected:
  Eigen::Vector2d
  materialCurvature(const Eigen::Ref<const Eigen::VectorXd> X, const LocalFrame &f1, const LocalFrame &f2) const;
  Eigen::Matrix<double, 11, 2> materialCurvatureDerivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                           const LocalFrame &f1,
                                                           const LocalFrame &f2,
                                                           const Eigen::Vector2d stiffnesses = Eigen::Vector2d::Zero(),
                                                           Eigen::Matrix<double, 11, 11> *dderiv = nullptr) const;

  double twistAngle(const Eigen::Ref<const Eigen::VectorXd> X) const;
  Eigen::Matrix<double, 11, 1> twistAngleDerivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                    const LocalFrame &f1,
                                                    const LocalFrame &f2,
                                                    Eigen::Matrix<double, 11, 11> *dderiv = nullptr) const;

  Vec<double, 9> stretchGradient(const Eigen::Ref<const Eigen::VectorXd> X) const;
  Mat<double, 9, 9> stretchHessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

private:
  Eigen::Vector2d _lengths;
  Eigen::Vector2d _restK; // rest material curvature (product of curvature binormal and mean material director)
  double _ref_twist;
};

} // namespace fsim
