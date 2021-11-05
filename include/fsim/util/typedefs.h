// typedefs.h
//
// Author: David Jourdan
// Created: 03/02/2020

#pragma once

#include <Eigen/Dense>

namespace fsim
{
template <typename Scalar, int i = -1, int j = -1>
using Matrix = Eigen::Matrix<Scalar, i, j, Eigen::RowMajor>;

template <typename Scalar, int i = -1>
using Vector = Eigen::Matrix<Scalar, i, 1>;

template <typename Scalar>
using Mat4 = Eigen::Matrix<Scalar, -1, 4, Eigen::RowMajor>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, -1, 3, Eigen::RowMajor>;

template <typename Scalar>
using Mat2 = Eigen::Matrix<Scalar, -1, 2, Eigen::RowMajor>;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
} // namespace fsim
