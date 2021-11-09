// ElasticMembraneModel.h
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
 * template class for any FEM-based membrane material model (e.g. StVK, neohookean...)
 * @tparam Element  triangle stencil class such as StVKElement, NeoHookeanElement, etc.
 */
template <class Element>
class ElasticMembraneModel : public ModelBase<Element>
{
public:
  /**
   * constructor for ElasticMembraneModel
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param thickness  membrane's thickness
   */
  ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                       const Eigen::Ref<const Mat3<int>> F,
                       double thickness,
                       double young_modulus,
                       double poisson_ratio);

  ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                       const Eigen::Ref<const Mat3<int>> F,
                       const std::vector<double> &thicknesses,
                       double young_modulus,
                       double poisson_ratio);

  ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                       const Eigen::Ref<const Mat3<int>> F,
                       const std::vector<double> &thicknesses,
                       const std::vector<double> &young_moduli,
                       double poisson_ratio);

  int nbDOFs() const { return 3 * nV; }
  void setPoissonRatio(double poisson_ratio);
  void setYoungModulus(double E);
  void setThickness(double t);

  double getPoissonRatio() { return Element::nu; }
  double getYoungModulus() { return _young_modulus; }
  double getThickness() { return _thickness; }

private:
  int nV, nF;
  double _thickness = -1;
  double _young_modulus = -1;
};

// the ids are there to disambiguate between different instances so that they don't have the same Lamé coefficients
// and thicknesses (which are stored as static variables in each TriangleElement)
// If you only want to declare one Membrane instance (or if you're using several with the same Lamé coefficents),
// you can safely use the provided typedefs (e.g. StVKMembrane). However if you declared several instances with
// different Lamé coefficents, please declare them as e.g. StVKMembrane<0>, StVKMembrane<1>, etc.

using StVKMembrane = ElasticMembraneModel<StVKElement<0>>;
using NeoHookeanMembrane = ElasticMembraneModel<NeoHookeanElement<0>>;
using NHIncompressibleMembrane = ElasticMembraneModel<NHIncompressibleElement<0>>;

template <int id = 0>
using StVKMembraneModel = ElasticMembraneModel<StVKElement<id>>;
template <int id = 0>
using NeoHookeanMembraneModel = ElasticMembraneModel<NeoHookeanElement<id>>;
template <int id = 0>
using NHIncompressibleMembraneModel = ElasticMembraneModel<NHIncompressibleElement<id>>;

template <class Element>
ElasticMembraneModel<Element>::ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                                                    const Eigen::Ref<const Mat3<int>> F,
                                                    double thickness,
                                                    double young_modulus,
                                                    double poisson_ratio)
    : ElasticMembraneModel(V,
                           F,
                           std::vector<double>(F.rows(), thickness),
                           std::vector<double>(F.rows(), young_modulus),
                           poisson_ratio)
{
  _young_modulus = young_modulus;
  _thickness = thickness;
}

template <class Element>
ElasticMembraneModel<Element>::ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                                                    const Eigen::Ref<const Mat3<int>> F,
                                                    const std::vector<double> &thicknesses,
                                                    double young_modulus,
                                                    double poisson_ratio)
    : ElasticMembraneModel(V, F, thicknesses, std::vector<double>(F.rows(), young_modulus), poisson_ratio)
{
  _young_modulus = young_modulus;
}

template <class Element>
ElasticMembraneModel<Element>::ElasticMembraneModel(const Eigen::Ref<const Mat3<double>> V,
                                                    const Eigen::Ref<const Mat3<int>> F,
                                                    const std::vector<double> &thicknesses,
                                                    const std::vector<double> &young_moduli,
                                                    double poisson_ratio)
{
  using namespace Eigen;

  nV = V.rows();
  nF = F.rows();

  if(Element::nu != 0)
    std::cerr << "Warning: overwriting Poisson's ratio. Please declare your different instances as e.g. "
                 "StVKMembraneModel<0>, StVKMembraneModel<1>, etc.\n";
  Element::nu = poisson_ratio;

  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
  {
    this->_elements.emplace_back(V, F.row(i), thicknesses[i], young_moduli[i]);
  }
}

template <class Element>
void ElasticMembraneModel<Element>::setPoissonRatio(double poisson_ratio)
{
  Element::nu = poisson_ratio;
}

template <class Element>
void ElasticMembraneModel<Element>::setYoungModulus(double young_modulus)
{
  if(_young_modulus <= 0)
    throw std::runtime_error(
        "Warning: membrane may have a locally varying Young's modulus\nCan't set it to a constant value\n");
  for(auto &e: this->_elements)
  {
    e.coeff *= young_modulus / _young_modulus;
  }
  _young_modulus = young_modulus;
}

template <class Element>
void ElasticMembraneModel<Element>::setThickness(double t)
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
