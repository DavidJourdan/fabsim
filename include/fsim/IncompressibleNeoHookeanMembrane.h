// IncompressibleNeoHookeanMembrane.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/19/20

#pragma once

#include "IncompressibleNeoHookeanElement.h"
#include "ModelBase.h"
#include "util/typedefs.h"

#include <exception>
#include <iostream>

namespace fsim
{

/**
 * template class for isotropic membrane models (e.g. StVK, neohookean...)
 * @tparam IncompressibleNeoHookeanElement  triangle stencil class such as StVKElement, NeoHookeanElement, etc.
 */
template <int id = 0>
class IncompressibleNeoHookeanMembrane : public ModelBase<IncompressibleNeoHookeanElement<id>>
{
public:
  /**
   * constructor for IncompressibleNeoHookeanMembrane
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param thickness  membrane's thickness
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  IncompressibleNeoHookeanMembrane(const Eigen::Ref<const Mat3<double>> V,
                                   const Eigen::Ref<const Mat3<int>> F,
                                   double thickness,
                                   double young_modulus,
                                   double poisson_ratio,
                                   double mass = 0);

  IncompressibleNeoHookeanMembrane(const Eigen::Ref<const Mat3<double>> V,
                                   const Eigen::Ref<const Mat3<int>> F,
                                   const std::vector<double> &thicknesses,
                                   double young_modulus,
                                   double poisson_ratio,
                                   double mass = 0);

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio);
  double getPoissonRatio() const { return IncompressibleNeoHookeanElement<id>::nu; }

  // set Young's modulus (positive coefficient)
  void setYoungModulus(double E);
  double getYoungModulus() const { return IncompressibleNeoHookeanElement<id>::E; }

  // set thickness of the membrane (controls the amount of stretching and the total weight)
  // negative values are not allowed
  void setThickness(double t);
  double getThickness() const { return _thickness; }

  void setMass(double mass);
  double getMass() const { return IncompressibleNeoHookeanElement<id>::mass; }

private:
  int nV, nF;
  double _thickness = -1;
};

// the ids are there to disambiguate between different instances so that they don't have the same Lamé coefficients
// and thicknesses (which are stored as static variables in each TriangleElement)
// If you only want to declare one Membrane instance (or if you're using several with the same Lamé coefficents),
// you can safely leave the angle brackets empty (e.g. StVKMembrane<>). However if you declared several instances with
// different Lamé coefficents, please declare them as e.g. StVKMembrane<0>, StVKMembrane<1>, etc.

template <int id>
IncompressibleNeoHookeanMembrane<id>::IncompressibleNeoHookeanMembrane(const Eigen::Ref<const Mat3<double>> V,
                                                                       const Eigen::Ref<const Mat3<int>> F,
                                                                       double thickness,
                                                                       double young_modulus,
                                                                       double poisson_ratio,
                                                                       double mass)
    : IncompressibleNeoHookeanMembrane(V,
                                       F,
                                       std::vector<double>(F.rows(), thickness),
                                       young_modulus,
                                       poisson_ratio,
                                       mass)
{
  _thickness = thickness;
}

template <int id>
IncompressibleNeoHookeanMembrane<id>::IncompressibleNeoHookeanMembrane(const Eigen::Ref<const Mat3<double>> V,
                                                                       const Eigen::Ref<const Mat3<int>> F,
                                                                       const std::vector<double> &thicknesses,
                                                                       double young_modulus,
                                                                       double poisson_ratio,
                                                                       double mass)
{
  nV = V.rows();
  nF = F.rows();

  if(IncompressibleNeoHookeanElement<id>::E != 0 && IncompressibleNeoHookeanElement<id>::E != young_modulus ||
     IncompressibleNeoHookeanElement<id>::nu != 0 && IncompressibleNeoHookeanElement<id>::nu != poisson_ratio ||
     IncompressibleNeoHookeanElement<id>::mass != 0 && IncompressibleNeoHookeanElement<id>::mass != mass)
    std::cerr << "Warning: overwriting properties. Please declare your different instances as "
                 "IncompressibleNeoHookeanMembrane<0>, IncompressibleNeoHookeanMembrane<1>, etc.\n";
  IncompressibleNeoHookeanElement<id>::E = young_modulus;
  IncompressibleNeoHookeanElement<id>::nu = poisson_ratio;
  IncompressibleNeoHookeanElement<id>::mass = mass;

  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), thicknesses[i]);
}

template <int id>
void IncompressibleNeoHookeanMembrane<id>::setPoissonRatio(double poisson_ratio)
{
  IncompressibleNeoHookeanElement<id>::nu = poisson_ratio;
}

template <int id>
void IncompressibleNeoHookeanMembrane<id>::setYoungModulus(double young_modulus)
{
  IncompressibleNeoHookeanElement<id>::E = young_modulus;
}

template <int id>
void IncompressibleNeoHookeanMembrane<id>::setMass(double mass)
{
  IncompressibleNeoHookeanElement<id>::mass = mass;
}

template <int id>
void IncompressibleNeoHookeanMembrane<id>::setThickness(double t)
{
  if(_thickness <= 0)
    throw std::runtime_error(
        "Warning: membrane may have a locally varying thickness\nCan't set it to a constant value\n");
  for(auto &elem: this->_elements)
  {
    elem.coeff *= t / _thickness;
  }
  _thickness = t;
}
} // namespace fsim
