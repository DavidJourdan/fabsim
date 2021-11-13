// ElasticMembrane.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/19/20

#pragma once

#include "ModelBase.h"
#include "TriangleElement.h"
#include "util/typedefs.h"

#include <exception>
#include <iostream>

namespace fsim
{

/**
 * template class for isotropic membrane models (e.g. StVK, neohookean...)
 * @tparam Element  triangle stencil class such as StVKElement, NeoHookeanElement, etc.
 */
template <class Element>
class ElasticMembrane : public ModelBase<Element>
{
public:
  /**
   * constructor for ElasticMembrane
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param thickness  membrane's thickness
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                  const Eigen::Ref<const Mat3<int>> F,
                  double thickness,
                  double young_modulus,
                  double poisson_ratio,
                  double mass = 0);

  ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                  const Eigen::Ref<const Mat3<int>> F,
                  const std::vector<double> &thicknesses,
                  double young_modulus,
                  double poisson_ratio,
                  double mass = 0);

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio);
  double getPoissonRatio() const { return Element::nu; }

  // set Young's modulus (positive coefficient)
  void setYoungModulus(double E);
  double getYoungModulus() const { return Element::E; }

  // set thickness of the membrane (controls the amount of stretching and the total weight) negative values are not allowed
  void setThickness(double t);
  double getThickness() const { return _thickness; }

  void setMass(double mass);
  double getMass() const { return Element::mass; }

private:
  int nV, nF;
  double _thickness = -1;
};

// the ids are there to disambiguate between different instances so that they don't have the same Lamé coefficients
// and thicknesses (which are stored as static variables in each TriangleElement)
// If you only want to declare one Membrane instance (or if you're using several with the same Lamé coefficents),
// you can safely leave the angle brackets empty (e.g. StVKMembrane<>). However if you declared several instances with
// different Lamé coefficents, please declare them as e.g. StVKMembrane<0>, StVKMembrane<1>, etc.

template <int id = 0>
using StVKMembrane = ElasticMembrane<StVKElement<id>>;
template <int id = 0>
using NeoHookeanMembrane = ElasticMembrane<NeoHookeanElement<id>>;
template <int id = 0>
using NeoHookeanIncompressibleMembrane = ElasticMembrane<NeoHookeanIncompressibleElement<id>>;

template <class Element>
ElasticMembrane<Element>::ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          double thickness,
                                          double young_modulus,
                                          double poisson_ratio,
                                          double mass)
    : ElasticMembrane(V, F, std::vector<double>(F.rows(), thickness), young_modulus, poisson_ratio, mass)
{
  _thickness = thickness;
}

template <class Element>
ElasticMembrane<Element>::ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          const std::vector<double> &thicknesses,
                                          double young_modulus,
                                          double poisson_ratio,
                                          double mass)
{
  nV = V.rows();
  nF = F.rows();

  if(Element::E != 0 && Element::E != young_modulus || Element::nu != 0 && Element::nu != poisson_ratio ||
     Element::mass != 0 && Element::mass != mass)
    std::cerr << "Warning: overwriting properties. Please declare your different instances as e.g. "
                 "StVKMembrane<0>, StVKMembrane<1>, etc.\n";
  Element::E = young_modulus;
  Element::nu = poisson_ratio;
  Element::mass = mass;

  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), thicknesses[i]);
}

template <class Element>
void ElasticMembrane<Element>::setPoissonRatio(double poisson_ratio)
{
  Element::nu = poisson_ratio;
}

template <class Element>
void ElasticMembrane<Element>::setYoungModulus(double young_modulus)
{
  Element::E = young_modulus;
}

template <class Element>
void ElasticMembrane<Element>::setMass(double mass)
{
  Element::mass = mass;
}

template <class Element>
void ElasticMembrane<Element>::setThickness(double t)
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
