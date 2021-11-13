// StVKElement.h
//
// Authors: Etienne Vouga and David Jourdan (david.jourdan@inria.fr)
// Created: 01/13/20
// Code adapted with permission from Etienne Vouga's implementation of the Discrete shell energy
// For original implementation see https://github.com/evouga/libshell

#pragma once

#include "ElementBase.h"
#include "util/first_fundamental_form.h"
#include "util/geometry.h"
#include "util/typedefs.h"

namespace fsim
{

/**
 * Stencil for the StVK energy
 * @tparam mat  enum representing which material model to use when computing the energy and its derivatives
 * @tparam id  unique identifier meant to disambiguate between different TriangleElement instances so
 * that they don't have the same Lamé parameters and thicknesses (which are stored as static variables)
 */
template <int id = 0>
class StVKElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the TriangleElement class
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &face, double thickness);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Green strain tensor E = \frac 1 2 (\bar a^{-1} a - I)  where a is the first fundamental form
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Green strain
   */
  Eigen::Matrix2d strain(const Eigen::Ref<const Mat3<double>> V) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Ref<const Mat3<double>> V) const;

  /**
   * Computes the principal strain directions and their corresponding eigenvalues
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param max_dir  maximum strain direction
   * @param min_dir  minimum strain direction
   * @param eigs  eigenvalues (in ascending order)
   */
  void principalStrains(const Eigen::Ref<const Mat3<double>> V,
                        Eigen::Vector3d &max_dir,
                        Eigen::Vector3d &min_dir,
                        Eigen::Vector2d &eigs) const;

  /**
   * Computes the principal stress directions and their corresponding eigenvalues
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param max_dir  maximum stress direction
   * @param min_dir  minimum stress direction
   * @param eigs  eigenvalues (in ascending order)
   */
  void principalStresses(const Eigen::Ref<const Mat3<double>> V,
                         Eigen::Vector3d &max_dir,
                         Eigen::Vector3d &min_dir,
                         Eigen::Vector2d &eigs) const;

  static double nu;   // Poisson's ratio
  static double mass; // mass per unit volume
  static double E;    // Young's modulus
  double coeff;

protected:
  void principalDirectionsAndEigenvalues(const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> &solver,
                                         const Eigen::Ref<const Mat3<double>> V,
                                         Eigen::Vector3d &max_dir,
                                         Eigen::Vector3d &min_dir,
                                         Eigen::Vector2d &eigs) const;

  Eigen::Matrix2d abar_inv;
};

// the ids are there to disambiguate between TriangleElements pertaining to different Membrane instances
// so that they don't have the same static fields
template <int id>
double StVKElement<id>::nu = 0;

template <int id>
double StVKElement<id>::mass = 0;

template <int id>
double StVKElement<id>::E = 0;

template <int id>
StVKElement<id>::StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &E, double thickness)
{
  using namespace Eigen;

  idx = E;
  Matrix2d abar = first_fundamental_form(V, idx);
  abar_inv = abar.inverse();
  coeff = thickness / 2 * sqrt(abar.determinant());
}

template <int id>
double StVKElement<id>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);
  double res = 9.8 * coeff / 3 * mass * (V(idx(0), 2) + V(idx(1), 2) + V(idx(2), 2));

  Matrix2d a = first_fundamental_form(V, idx);
  // Green strain tensor (abar_inv * a  can be identified as the right Cauchy-Green deformation tensor)
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2;
  return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * pow(M.trace(), 2) + (M * M).trace()) +
         9.8 * coeff / 3 * mass * (V(idx(0), 2) + V(idx(1), 2) + V(idx(2), 2));
}

template <int id>
typename StVKElement<id>::LocalVector StVKElement<id>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Matrix<double, 4, 9> aderiv;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv);
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2; // Green strain tensor

  Matrix2d temp = M * abar_inv + nu / (1 - nu) * M.trace() * abar_inv;
  Map<Vector4d> flat(temp.data());

  Vec<double, 9> res = coeff * E / (2 * (1 + nu)) * aderiv.transpose() * flat;

  res(2) += 9.8 * coeff / 3 * mass;
  res(5) += 9.8 * coeff / 3 * mass;
  res(8) += 9.8 * coeff / 3 * mass;

  return res;
}

template <int id>
typename StVKElement<id>::LocalMatrix StVKElement<id>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Matrix<double, 4, 9> aderiv;
  Matrix<double, 36, 9> ahess;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv, &ahess);
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2; // Green strain tensor

  Matrix<double, 9, 1> inner = aderiv.transpose() * Map<Vector4d>{const_cast<double *>(abar_inv.data())};
  Matrix<double, 9, 9> hess = nu / (2 * (1 - nu)) * outer_prod(inner, inner);

  Matrix2d Mabarinv = M * abar_inv;
  for(int i = 0; i < 4; ++i) // iterate over Mabarinv and abar_inv as if they were vectors
    hess += (Mabarinv(i) + nu / (1 - nu) * M.trace() * abar_inv(i)) * ahess.block<9, 9>(9 * i, 0);

  Matrix<double, 9, 1> inner00 = abar_inv(0, 0) * aderiv.row(0) + abar_inv(0, 1) * aderiv.row(2);
  Matrix<double, 9, 1> inner01 = abar_inv(0, 0) * aderiv.row(1) + abar_inv(0, 1) * aderiv.row(3);
  Matrix<double, 9, 1> inner10 = abar_inv(1, 0) * aderiv.row(0) + abar_inv(1, 1) * aderiv.row(2);
  Matrix<double, 9, 1> inner11 = abar_inv(1, 0) * aderiv.row(1) + abar_inv(1, 1) * aderiv.row(3);
  hess += 0.5 * outer_prod(inner00, inner00);
  hess += sym(outer_prod(inner01, inner10));
  hess += 0.5 * outer_prod(inner11, inner11);

  hess *= coeff * E / (2 * (1 + nu));
  return hess;
}

template <int id>
Eigen::Matrix2d StVKElement<id>::strain(const Eigen::Ref<const Mat3<double>> V) const
{
  Eigen::Matrix2d a = first_fundamental_form(V, idx);
  return 0.5 * (abar_inv * a - Eigen::Matrix2d::Identity());
}

template <int id>
Eigen::Matrix2d StVKElement<id>::stress(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2;
  return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * M.trace() * Matrix2d::Identity() + M);
}

template <int id>
void StVKElement<id>::principalDirectionsAndEigenvalues(
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> &eigensolver,
    const Eigen::Ref<const Mat3<double>> V,
    Eigen::Vector3d &max_dir,
    Eigen::Vector3d &min_dir,
    Eigen::Vector2d &eigs) const
{
  // triangle frame, edge-aligned
  Eigen::Matrix<double, 3, 2> X;
  X.col(0) = V.row(idx(1)) - V.row(idx(0));
  X.col(1) = V.row(idx(2)) - V.row(idx(0));

  Eigen::Matrix<double, 3, 2> dirs = X * eigensolver.eigenvectors();
  Eigen::Vector2d lambda = eigensolver.eigenvalues();

  // order eigenvalues and eigenvectors
  if(lambda(0) < lambda(1))
  {
    min_dir = dirs.col(0);
    max_dir = dirs.col(1);
    eigs = lambda;
  }
  else
  {
    min_dir = dirs.col(1);
    max_dir = dirs.col(0);
    eigs(0) = lambda(1);
    eigs(1) = lambda(0);
  }
}

template <int id>
void StVKElement<id>::principalStrains(const Eigen::Ref<const Mat3<double>> V,
                                       Eigen::Vector3d &max_dir,
                                       Eigen::Vector3d &min_dir,
                                       Eigen::Vector2d &eigs) const
{
  using namespace Eigen;
  Matrix2d abar = abar_inv.inverse();
  return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(abar * strain(V), abar), V,
                                           max_dir, min_dir, eigs);
}

template <int id>
void StVKElement<id>::principalStresses(const Eigen::Ref<const Mat3<double>> V,
                                        Eigen::Vector3d &max_dir,
                                        Eigen::Vector3d &min_dir,
                                        Eigen::Vector2d &eigs) const
{
  using namespace Eigen;

  Matrix2d abar = abar_inv.inverse();
  Matrix2d a = first_fundamental_form(V, idx);

  return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(abar * stress(V), abar), V,
                                           max_dir, min_dir, eigs);
}

} // namespace fsim
