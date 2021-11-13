// geometry.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 8/20/19

#pragma once

#include "fsim/util/typedefs.h"

#include <Eigen/Dense>

namespace fsim
{

/**
 * Computes the signed angle from one vector to another given an orientation vector.
 * @param u The "from" vector.
 * @param v The "to" vector.
 * @param n The vector giving the orientation of the rotation plane.
 * @return The signed angle between the two vectors
 */
double signed_angle(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n);

/**
 * Computes the sinus of the signed angle
 * @param u The normalized "from" vector.
 * @param v The normalized "to" vector.
 * @param n The vector giving the orientation of the rotation plane.
 * @return The signed angle between the two vectors
 */
double sin_angle(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n);

/**
 * Computes the tangent of half the signed angle from one vector to another
 * i.e. if theta = signed_angle(u,v,n), then tangle_angle(u,v,n) returns tan(theta/2)
 * Note that u and v have to be normalized
 *
 * @param u The normalized "from" vector.
 * @param v The normalized "to" vector.
 * @param n The vector giving the orientation of the rotation plane.
 * @return The signed angle between the two vectors
 */
double tan_angle_2(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n);

/** Returns the matrix representation of the cross product operator
    associated to a vector. Given a vector $a$ it returns a matrix
    $A$ defined so that $Ab = a\times b$ for all vectors
    $b$. */

/**
 * The cross matrix is the matrix representation of the cross product operator associated to a vector.
 * Given a vector $a$ it returns a matrix $A$ defined so that $Ab = a\times b$ for all vectors $b$
 * @param a  vector in R^3
 * @return  the cross matrix of a
 */
Eigen::Matrix3d cross_matrix(const Eigen::Vector3d &a);

/**
 * Computes the parallel transport (twist-free rotation) of u from t1 to t2
 * @param u  The 3-vector to be parallel-transported
 * @param t1  The normalized "from" vector
 * @param t2  The normalized "to" vector
 * @return the parallel-transported vector
 */
Eigen::Vector3d parallel_transport(const Eigen::Vector3d &u, const Eigen::Vector3d &t1, const Eigen::Vector3d &t2);

/**
 * Check whether point x is in the segment bound by points a & b
 * @param x  input point in 3d space
 * @param a  "left bound" point in 3d space
 * @param b  "right bound" point in 3d space
 * @return  whether x is aligned and in between a and b
 */
bool point_in_segment(const Eigen::Vector3d &x, const Eigen::Vector3d &a, const Eigen::Vector3d &b);

/**
 * Both normalizes x in place and returns its norm
 * @param x  vector to be normalized
 * @return  norm of x
 */
template <int n = 3>
double normalize(Vec<double, n> &x)
{
  double norm = x.norm();
  x /= norm;
  return norm;
}

/**
 * Returns the outer product (tensor product of vectors)
 * @param x
 * @param y
 * @return  $$M = x \cdot y^T$$
 */
template <int i = 3, int j = 3>
auto outer_prod(const Vec<double, i> &x, const Vec<double, j> &y)
{
  return x * y.transpose();
}

/**
 * Returns the symmetric part of a square matrix
 * @param M
 * @return  $$\frac 1 2 (M + M^T)$$
 */
template <class DerivedM>
auto sym(const Eigen::MatrixBase<DerivedM> &M)
{
  return (M + M.transpose()) / 2;
}

} // namespace fsim
