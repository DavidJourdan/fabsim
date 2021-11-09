// TriangleElement.h
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

enum class MaterialType { StVK, NeoHookean, NeoHookeanIncompressible };

/**
 * Stencil for triangle-based elastic energies such as StVK
 * @tparam mat  enum representing which material model to use when computing the energy and its derivatives
 * @tparam id  unique identifier meant to disambiguate between different TriangleElement instances so
 * that they don't have the same Lamé parameters and thicknesses (which are stored as static variables)
 */
template <MaterialType mat, int id = 0>
class TriangleElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the TriangleElement class
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  TriangleElement(const Eigen::Ref<const Mat3<double>> V,
                  const Eigen::Vector3i &face,
                  double thickness);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    using namespace Eigen;
    Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);
    double res = 9.8 * coeff * mass * (V(idx(0), 2) + V(idx(1), 2) + V(idx(2), 2));
    if(mat == MaterialType::StVK)
      res += energyStVK(V);
    if(mat == MaterialType::NeoHookean)
      res += energyNeoHookean(V);
    if(mat == MaterialType::NeoHookeanIncompressible)
      res += energyIncompressibleNeoHookean(V);
    return res;
  }

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    using namespace Eigen;
    Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);
    LocalVector res;
    if(mat == MaterialType::StVK)
      res = gradientStVK(V);
    if(mat == MaterialType::NeoHookean)
      res = gradientNeoHookean(V);
    if(mat == MaterialType::NeoHookeanIncompressible)
      res = gradientIncompressibleNeoHookean(V);
    res(2) += 9.8 * coeff * mass;
    res(5) += 9.8 * coeff * mass;
    res(8) += 9.8 * coeff * mass;
    return res;
  }

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    using namespace Eigen;
    Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);
    if(mat == MaterialType::StVK)
      return hessianStVK(V);
    if(mat == MaterialType::NeoHookean)
      return hessianNeoHookean(V);
    if(mat == MaterialType::NeoHookeanIncompressible)
      return hessianIncompressibleNeoHookean(V);
    else
      return LocalMatrix{};
  }

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

  static double nu; // Poisson's ratio
  static double mass; // mass per unit volume
  static double E; // Young's modulus
  double coeff;

protected:
  void principalDirectionsAndEigenvalues(const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> &solver,
                                         const Eigen::Ref<const Mat3<double>> V,
                                         Eigen::Vector3d &max_dir,
                                         Eigen::Vector3d &min_dir,
                                         Eigen::Vector2d &eigs) const;

  double energyStVK(const Eigen::Ref<const Mat3<double>> V) const;
  double energyNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;
  double energyIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;

  LocalVector gradientStVK(const Eigen::Ref<const Mat3<double>> V) const;
  LocalVector gradientNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;
  LocalVector gradientIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;

  LocalMatrix hessianStVK(const Eigen::Ref<const Mat3<double>> V) const;
  LocalMatrix hessianNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;
  LocalMatrix hessianIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const;

  Eigen::Matrix2d abar_inv;
};

// the ids are there to disambiguate between TriangleElements pertaining to different Membrane instances
// so that they don't have the same static fields
template <int id = 0>
using StVKElement = TriangleElement<MaterialType::StVK, id>;

template <int id = 0>
using NeoHookeanElement = TriangleElement<MaterialType::NeoHookean, id>;

template <int id = 0>
using NHIncompressibleElement = TriangleElement<MaterialType::NeoHookeanIncompressible, id>;

// https://stackoverflow.com/questions/3229883/static-member-initialization-in-a-class-template
template <MaterialType mat, int id>
double TriangleElement<mat, id>::nu = 0;

template <MaterialType mat, int id>
double TriangleElement<mat, id>::mass = 0;

template <MaterialType mat, int id>
double TriangleElement<mat, id>::E = 0;

template <MaterialType mat, int id>
TriangleElement<mat, id>::TriangleElement(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Vector3i &E,
                                          double thickness)
{
  using namespace Eigen;

  idx = E;
  Matrix2d abar = first_fundamental_form(V, idx);
  abar_inv = abar.inverse();
  coeff = thickness / 2 * sqrt(abar.determinant());
}

template <MaterialType mat, int id>
double TriangleElement<mat, id>::energyStVK(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);
  // Green strain tensor (abar_inv * a  can be identified as the right Cauchy-Green deformation tensor)
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2;
  return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * pow(M.trace(), 2) + (M * M).trace());
}

template <MaterialType mat, int id>
double TriangleElement<mat, id>::energyNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);
  // = \ln J with J^2 = \det C (C = abar_inv * a is the right Cauchy-Green deformation tensor)
  double lnJ = log((a * abar_inv).determinant()) / 2;
  return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * pow(lnJ, 2) + (abar_inv * a).trace() / 2 - 1 - lnJ);
}

template <MaterialType mat, int id>
double TriangleElement<mat, id>::energyIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);
  double J = sqrt((a * abar_inv).determinant());
  return coeff * E / (4 * (1 + nu)) * ((abar_inv * a).trace() / J - 2 + 1e4 * pow(J - 1, 2));
}

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalVector
TriangleElement<mat, id>::gradientStVK(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix<double, 4, 9> aderiv;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv);
  Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2; // Green strain tensor

  Matrix2d temp = M * abar_inv + nu / (1 - nu) * M.trace() * abar_inv;
  Map<Vector4d> flat(temp.data());

  return coeff * E / (2 * (1 + nu)) * aderiv.transpose() * flat;
}

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalVector
TriangleElement<mat, id>::gradientNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix<double, 4, 9> aderiv;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv);
  double lnJ = log((a * abar_inv).determinant()) / 2;

  Matrix2d temp = abar_inv + (2 * nu / (1 - nu) * lnJ - 1) * a.inverse();
  temp *= coeff * E / (4 * (1 + nu));

  return aderiv.transpose() * Map<Vector4d>(temp.data());
}

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalVector
TriangleElement<mat, id>::gradientIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix<double, 4, 9> aderiv;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv);
  double J = sqrt((a * abar_inv).determinant());
  double trC = (abar_inv * a).trace() / J;

  Matrix2d temp = abar_inv / J + (1e4 * J * (J - 1) - trC / 2) * a.inverse();
  temp *= coeff * E / (4 * (1 + nu));

  return aderiv.transpose() * Map<Vector4d>(temp.data());
}

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalMatrix
TriangleElement<mat, id>::hessianStVK(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

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

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalMatrix
TriangleElement<mat, id>::hessianNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

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

template <MaterialType mat, int id>
typename TriangleElement<mat, id>::LocalMatrix
TriangleElement<mat, id>::hessianIncompressibleNeoHookean(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix<double, 4, 9> aderiv;
  Matrix<double, 36, 9> ahess;
  Matrix2d a = first_fundamental_form(V, idx, &aderiv, &ahess);

  double J = sqrt((a * abar_inv).determinant());
  double trC = (a * abar_inv).trace() / J;
  double k = 1e4;

  Matrix2d a_inv = a.inverse();
  double a_det = a.determinant();

  Matrix<double, 9, 9> hess = (3 * trC / 4 + k * J / 2) * aderiv.transpose() * Map<Vector4d>(a_inv.data()) *
                                     Map<RowVector4d>(a_inv.data()) * aderiv;

  Matrix<double, 4, 9> a_deriv_adj;
  a_deriv_adj << aderiv.row(3), -aderiv.row(1), -aderiv.row(2), aderiv.row(0);

  hess += (k * J * (J - 1) - trC / 2) / a_det * aderiv.transpose() * a_deriv_adj;

  for(int j = 0; j < 4; ++j)
  {
    hess += (abar_inv(j) / J + (k * J * (J - 1) - trC / 2) * a_inv(j)) * ahess.block<9, 9>(9 * j, 0);
  }

  Vector4d flat_abarinv = Map<Vector4d>(const_cast<double *>(abar_inv.data()));
  hess += -1 / (2 * J) * (aderiv.transpose() * Map<Vector4d>(a_inv.data())) * (flat_abarinv.transpose() * aderiv);
  hess += -1 / (2 * J) * (aderiv.transpose() * flat_abarinv) * (Map<RowVector4d>(a_inv.data()) * aderiv);

  hess *= 0.5 * coeff * E / (2 * (1 + nu));
  return hess;
}

template <MaterialType mat, int id>
Eigen::Matrix2d TriangleElement<mat, id>::strain(const Eigen::Ref<const Mat3<double>> V) const
{
  Eigen::Matrix2d a = first_fundamental_form(V, idx);
  return 0.5 * (abar_inv * a - Eigen::Matrix2d::Identity());
}

template <MaterialType mat, int id>
Eigen::Matrix2d TriangleElement<mat, id>::stress(const Eigen::Ref<const Mat3<double>> V) const
{
  using namespace Eigen;

  Matrix2d a = first_fundamental_form(V, idx);

  if(mat == MaterialType::StVK)
  {
    Matrix2d M = (abar_inv * a - Matrix2d::Identity()) / 2;
    return coeff * E / (2 * (1 + nu)) * (nu / (1 - nu) * M.trace() * Matrix2d::Identity() + M);
  }
  else if(mat == MaterialType::NeoHookean)
  {
    double lnJ = log((a * abar_inv).determinant()) / 2;
    return coeff * E / (2 * (1 + nu)) * (Matrix2d::Identity() + (abar_inv * a).inverse() * (2 * nu / (1 - nu) * lnJ - 1));
  }
  else if(mat == MaterialType::NeoHookeanIncompressible)
  {
    double J = sqrt((a * abar_inv).determinant());
    double trC = (abar_inv * a).trace() / J;
    return coeff * E / (2 * (1 + nu)) *
           (Matrix2d::Identity() / J + (1e4 * J * (J - 1) - trC / 2) * (abar_inv * a).inverse());
  }
  else
    return Matrix2d{};
}

template <MaterialType mat, int id>
void TriangleElement<mat, id>::principalDirectionsAndEigenvalues(
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

template <MaterialType mat, int id>
void TriangleElement<mat, id>::principalStrains(const Eigen::Ref<const Mat3<double>> V,
                                                Eigen::Vector3d &max_dir,
                                                Eigen::Vector3d &min_dir,
                                                Eigen::Vector2d &eigs) const
{
  using namespace Eigen;
  Matrix2d abar = abar_inv.inverse();
  return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(abar * strain(V), abar), V,
                                           max_dir, min_dir, eigs);
}

template <MaterialType mat, int id>
void TriangleElement<mat, id>::principalStresses(const Eigen::Ref<const Mat3<double>> V,
                                                 Eigen::Vector3d &max_dir,
                                                 Eigen::Vector3d &min_dir,
                                                 Eigen::Vector2d &eigs) const
{
  using namespace Eigen;

  Matrix2d abar = abar_inv.inverse();
  Matrix2d a = first_fundamental_form(V, idx);

  if(mat == MaterialType::StVK)
    return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(abar * stress(V), abar), V,
                                             max_dir, min_dir, eigs);
  else if(mat == MaterialType::NeoHookean)
    return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(a * stress(V), a), V, max_dir,
                                             min_dir, eigs);
  else if(mat == MaterialType::NeoHookeanIncompressible)
    return principalDirectionsAndEigenvalues(GeneralizedSelfAdjointEigenSolver<Matrix2d>(a * stress(V), a), V, max_dir,
                                             min_dir, eigs);
}

} // namespace fsim
