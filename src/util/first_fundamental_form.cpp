// first_fundamental_form.cpp
//
// Author: Etienne Vouga, David Jourdan (david.jourdan@inria.fr)
// Created: 09/10/19
// Code adapted with permission from Etienne Vouga's implementation of the Discrete shell energy
// For original implementation see https://github.com/evouga/libshell

#include "fsim/util/first_fundamental_form.h"

Eigen::Matrix2d first_fundamental_form(const Eigen::Ref<const fsim::Mat3<double>> V,
                                       const Eigen::Vector3i &face,
                                       Eigen::Matrix<double, 4, 9> *derivative,
                                       Eigen::Matrix<double, 36, 9> *hessian)
{
  using namespace Eigen;

  Vector3d q0 = V.row(face(0));
  Vector3d q1 = V.row(face(1));
  Vector3d q2 = V.row(face(2));
  Matrix2d result;
  // clang-format off
  result << (q1 - q0).dot(q1 - q0), (q1 - q0).dot(q2 - q0),
            (q2 - q0).dot(q1 - q0), (q2 - q0).dot(q2 - q0);
  // clang-format on

  if(derivative)
  {
    derivative->block<1, 3>(0, 3) = 2.0 * (q1 - q0);
    derivative->block<1, 3>(0, 6).setZero();
    derivative->block<1, 3>(1, 3) = q2 - q0;
    derivative->block<1, 3>(1, 6) = q1 - q0;
    derivative->block<1, 3>(3, 3).setZero();
    derivative->block<1, 3>(3, 6) = 2.0 * (q2 - q0);

    derivative->block<4, 3>(0, 0) = -derivative->block<4, 3>(0, 3) - derivative->block<4, 3>(0, 6);
    derivative->row(2) = derivative->row(1); // symmetric rank 3 tensor
  }

  if(hessian)
  {
    hessian->setZero();
    Matrix3d I = Matrix3d::Identity();
    hessian->block<3, 3>(0, 0) += 2 * I;
    hessian->block<3, 3>(3, 3) += 2 * I;
    hessian->block<3, 3>(0, 3) -= 2 * I;
    hessian->block<3, 3>(3, 0) -= 2 * I;

    hessian->block<3, 3>(12, 6) += I;
    hessian->block<3, 3>(15, 3) += I;
    hessian->block<3, 3>(9, 3) -= I;
    hessian->block<3, 3>(9, 6) -= I;
    hessian->block<3, 3>(12, 0) -= I;
    hessian->block<3, 3>(15, 0) -= I;
    hessian->block<3, 3>(9, 0) += 2 * I;

    hessian->block<9, 9>(18, 0) = hessian->block<9, 9>(9, 0); // symmetric rank 4 tensor

    hessian->block<3, 3>(27, 0) += 2 * I;
    hessian->block<3, 3>(33, 6) += 2 * I;
    hessian->block<3, 3>(27, 6) -= 2 * I;
    hessian->block<3, 3>(33, 0) -= 2 * I;
  }

  return result;
}
