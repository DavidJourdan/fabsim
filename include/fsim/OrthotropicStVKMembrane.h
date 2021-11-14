// OrthotropicStVKMembrane.h
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#include "ModelBase.h"
#include "StVKElement.h"

#include <array>
#include <iostream>

namespace fsim
{

template <int id = 0>
class OrthotropicStVKMembrane : public ModelBase<StVKElement<id>>
{
public:
  /**
   * constructor for OrthotropicStVKMembrane
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  membrane's thickness (per-triangle value)
   * @param E1  0 degree Young's modulus
   * @param E2  90 degrees Young's modulus
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          double thickness,
                          double E1,
                          double E2,
                          double poisson_ratio,
                          double mass = 0);

  OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          const std::vector<double> &thicknesses,
                          double E1,
                          double E2,
                          double poisson_ratio,
                          double mass = 0);

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio);
  double getPoissonRatio() const { return _poisson_ratio; }

  // set Young's moduli (E1 and E2 respectively control the horizontal and vertical stiffness)
  void setYoungModuli(double E1, double E2);
  std::array<double, 2> getYoungModuli() const { return {_E1, _E2}; }

  // set thickness of the membrane (controls the amount of stretching and the total weight)
  // negative values are not allowed
  void setThickness(double t);
  double getThickness() const { return _thickness; }

  void setMass(double mass);
  double getMass() const { return StVKElement<id>::mass; }

private:
  int nV;
  double _thickness = -1;
  double _poisson_ratio;
  double _E1;
  double _E2;
};

template <int id>
OrthotropicStVKMembrane<id>::OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     double thickness,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     double mass)
    : OrthotropicStVKMembrane(V, F, std::vector<double>(F.rows(), thickness), E1, E2, poisson_ratio, mass)
{
  _thickness = thickness;
}

template <int id>
OrthotropicStVKMembrane<id>::OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     const std::vector<double> &thicknesses,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     double mass)
    : _poisson_ratio{poisson_ratio}, _E1{E1}, _E2{E2}
{
  using namespace Eigen;

  nV = V.rows();

  if(StVKElement<id>::_C.norm() != 0)
    std::cerr << "Warning: overwriting elasticity tensor. Please declare your different instances as "
                 "OrthotropicStVKMembrane<0>, OrthotropicStVKMembrane<1>, etc.\n";

  StVKElement<id>::_C << 
    E1, poisson_ratio * sqrt(E1 * E2), 0, 
    poisson_ratio * sqrt(E1 * E2), E2, 0, 
    0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
  StVKElement<id>::_C /= (1 - std::pow(poisson_ratio, 2));

  StVKElement<id>::mass = mass;
  int nF = F.rows();
  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), thicknesses[i]);
}

template <int id>
void OrthotropicStVKMembrane<id>::setPoissonRatio(double poisson_ratio)
{
  _poisson_ratio = poisson_ratio;

  StVKElement<id>::_C << 
    _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
    _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 0, 
    0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);

  StVKElement<id>::_C /= (1 - std::pow(_poisson_ratio, 2));
}

template <int id>
void OrthotropicStVKMembrane<id>::setYoungModuli(double E1, double E2)
{
  _E1 = E1;
  _E2 = E2;

  StVKElement<id>::_C << 
    _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
    _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 0, 
    0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);

  StVKElement<id>::_C /= (1 - std::pow(_poisson_ratio, 2));
}

template <int id>
void OrthotropicStVKMembrane<id>::setThickness(double t)
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

template <int id>
void OrthotropicStVKMembrane<id>::setMass(double mass)
{
  StVKElement<id>::mass = mass;
}

} // namespace fsim
