// first_fundamental_form.h
// Computation of the first fundamental form and its derivatives
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 09/10/19
// Code adapted with permission from Etienne Vouga's implementation of the Discrete shell energy
// For original implementation see https://github.com/evouga/libshell

#pragma once

#include "typedefs.h"

#include <Eigen/Dense>

/**
 * Computes the (discrete) first fundamental form I of a triangle whose indices are stored in face
 * Optionally computes derivative and hessian (if pointers are not null)
 * @param V  nV by 3 list of vertex positions
 * @param face  list of 3 indices, one per vertex of the triangle
 * @param derivative optional return parameter, each row corresponds to the gradient of 1 coefficient in the matrix
 * @param hessian optional return parameter, each 9x9 block corresponds to the hessian of 1 coefficient in the matrix
 */
Eigen::Matrix2d first_fundamental_form(const Eigen::Ref<const fsim::Mat3<double>> V,
                                       const Eigen::Vector3i &face,
                                       Eigen::Matrix<double, 4, 9> *derivative = nullptr,
                                       Eigen::Matrix<double, 36, 9> *hessian = nullptr);
