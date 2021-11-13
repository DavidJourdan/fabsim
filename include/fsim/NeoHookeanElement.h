// NeoHookeanElement.h
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
 * Stencil for the neo-hookean energy
 * @tparam mat  enum representing which material model to use when computing the energy and its derivatives
 * @tparam id  unique identifier meant to disambiguate between different TriangleElement instances so
 * that they don't have the same Lamé parameters and thicknesses (which are stored as static variables)
 */
template <int id = 0>
class NeoHookeanElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the TriangleElement class
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  NeoHookeanElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &face, double thickness);

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
double NeoHookeanElement<id>::nu = 0;

template <int id>
double NeoHookeanElement<id>::mass = 0;

template <int id>
double NeoHookeanElement<id>::E = 0;

template <int id>
NeoHookeanElement<id>::NeoHookeanElement(const Eigen::Ref<const Mat3<double>> V,
                                         const Eigen::Vector3i &E,
                                         double thickness)
{
  using namespace Eigen;

  idx = E;
  Matrix2d abar = first_fundamental_form(V, idx);
  abar_inv = abar.inverse();
  coeff = thickness / 2 * sqrt(abar.determinant());
}

template <int id>
double NeoHookeanElement<id>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Matrix2d a = first_fundamental_form(V, idx);
  // = \ln J with J^2 = \det C (C = abar_inv * a is the right Cauchy-Green deformation tensor)
  double lnJ = log((a * abar_inv).determinant()) / 2;
  return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * pow(lnJ, 2) + (abar_inv * a).trace() / 2 - 1 - lnJ) +
         9.8 * coeff / 3 * mass * (V(idx(0), 2) + V(idx(1), 2) + V(idx(2), 2));
}

template <int id>
typename NeoHookeanElement<id>::LocalVector
NeoHookeanElement<id>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Matrix<double, 4, 9> aderiv;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv);
  double lnJ = log((a * abar_inv).determinant()) / 2;

  Matrix2d temp = abar_inv + (2 * nu / (1 - nu) * lnJ - 1) * a.inverse();
  temp *= coeff * E / (4 * (1 + nu));

  Vec<double, 9> res = aderiv.transpose() * Map<Vector4d>(temp.data());

  res(2) += 9.8 * coeff / 3 * mass;
  res(5) += 9.8 * coeff / 3 * mass;
  res(8) += 9.8 * coeff / 3 * mass;

  return res;
}

template <int id>
typename NeoHookeanElement<id>::LocalMatrix
NeoHookeanElement<id>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Matrix<double, 4, 9> aderiv;
  Matrix<double, 36, 9> ahess;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv, &ahess);

  Matrix2d a_inv = a.inverse();
  double k = -1 + nu / (1 - nu) * log((a * abar_inv).determinant());

  Matrix<double, 9, 1> ainvda = aderiv.transpose() * Map<Vector4d>(a_inv.data());
  Matrix<double, 9, 9> hess = (nu / (1 - nu) - k) * ainvda * ainvda.transpose();

  Matrix<double, 4, 9> a_deriv_adj;
  a_deriv_adj << aderiv.row(3), -aderiv.row(1), -aderiv.row(2), aderiv.row(0);

  hess += k * a_inv.determinant() * a_deriv_adj.transpose() * aderiv;

  for(int j = 0; j < 4; ++j)
    hess += (a_inv(j) * k + abar_inv(j) * 1) * ahess.block<9, 9>(9 * j, 0);

  hess *= coeff * E / (4 * (1 + nu));
  return hess;
}

template <int id>
Eigen::Matrix2d NeoHookeanElement<id>::strain(const Eigen::Ref<const Mat3<double>> V) const
{
  Eigen::Matrix2d a = first_fundamental_form(V, idx);
  return 0.5 * (abar_inv * a - Eigen::Matrix2d::Identity());
}

template <int id>
Eigen::Matrix2d NeoHookeanElement<id>::stress(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);

  double lnJ = log((a * abar_inv).determinant()) / 2;
  return coeff * E / (2 * (1 + nu)) * (Matrix2d::Identity() + (abar_inv * a).inverse() * (2 * nu / (1 - nu) * lnJ - 1));
}

template <int id>
void NeoHookeanElement<id>::principalDirectionsAndEigenvalues(
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
void NeoHookeanElement<id>::principalStrains(const Eigen::Ref<const Mat3<double>> V,
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
void NeoHookeanElement<id>::principalStresses(const Eigen::Ref<const Mat3<double>> V,
                                              Eigen::Vector3d &max_dir,
                                              Eigen::Vector3d &min_dir,
                                              Eigen::Vector2d &eigs) const
{
  using namespace Eigen;

  Matrix2d abar = abar_inv.inverse();
  Matrix2d a = first_fundamental_form(V, idx);

  return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(a * stress(V), a), V, max_dir,
                                           min_dir, eigs);
}

} // namespace fsim
