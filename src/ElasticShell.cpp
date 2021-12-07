// ElasticShell.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 09/11/19

#include "fsim/ElasticShell.h"

#include "fsim/util/geometry.h"

namespace fsim
{

template <class Formulation>
DiscreteShell<Formulation>::DiscreteShell(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          double thickness,
                                          double young_modulus,
                                          double poisson_ratio)
    : _young_modulus(young_modulus), _thickness(thickness), _poisson_ratio(poisson_ratio)
{
  using namespace Eigen;

  nV = V.rows();
  nF = F.rows();

  MatrixXi E = hingeIndices(V, F);
  nE = E.rows();

  double flexural_energy = young_modulus * pow(thickness, 3) / 24 / (1 - pow(poisson_ratio, 2));

  this->_elements.reserve(nE);
  for(int i = 0; i < nE; ++i)
  {
    int v0 = E(i, 0);
    int v1 = E(i, 1);
    int v2 = E(i, 2);
    int v3 = E(i, 3);

    Vector3d n0 = (V.row(v0) - V.row(v2)).cross(V.row(v1) - V.row(v2));
    Vector3d n1 = (V.row(v1) - V.row(v3)).cross(V.row(v0) - V.row(v3));
    Vector3d axis = V.row(v1) - V.row(v0);

    // formula from "Discrete bending forces and their Jacobians" (p.5)
    double coeff = flexural_energy * 3 * axis.squaredNorm() / (n0.norm() / 2 + n1.norm() / 2);

    this->_elements.emplace_back(V, E.row(i), coeff);
  }
}

template <class Formulation>
Mat4<int> DiscreteShell<Formulation>::hingeIndices(const Eigen::Ref<const Mat3<double>> V,
                                                   const Eigen::Ref<const Mat3<int>> F)
{
  using namespace Eigen;

  std::vector<std::tuple<int, int, int>> indices; // oriented face indices, reordered to merge them into hinge indices
  indices.reserve(3 * F.rows());
  for(int i = 0; i < F.rows(); ++i)
  {
    // clang-format off
    auto fill = [&](int a, int b, int c)
    {
      // clang-format on
      if(a < b)
        indices.emplace_back(a, b, c);
      else
        indices.emplace_back(b, a, c);
    };

    fill(F(i, 0), F(i, 1), F(i, 2));
    fill(F(i, 1), F(i, 2), F(i, 0));
    fill(F(i, 2), F(i, 0), F(i, 1));
  }
  std::sort(indices.begin(), indices.end());
  Mat4<int> E(indices.size(), 4);

  // "merge" neighboring indices to form a hinge index list
  int i = 0, k = 0;
  while(i < indices.size() - 1)
  {
    int x0, x1, x2, y0, y1, y2;
    std::tie(x0, x1, x2) = indices[i];
    std::tie(y0, y1, y2) = indices[i + 1];
    if(x0 == y0 && x1 == y1)
    {
      assert(x2 != y2);
      if((V.row(x1) - V.row(x0)).cross(V.row(x2) - V.row(x0)).dot(RowVector3d::UnitZ()) < 0)
        E.row(k++) << x1, x0, x2, y2;
      else
        E.row(k++) << x0, x1, x2, y2;
      i += 2;
    }
    else
    {
      i += 1;
    }
  }
  E.conservativeResize(k, 4);

  return E;
}

template <class Formulation>
void DiscreteShell<Formulation>::setYoungModulus(double young_modulus)
{
  assert(young_modulus > 0);
  for(auto &element: this->_elements)
  {
    element._coeff *= young_modulus / _young_modulus;
  }
  _young_modulus = young_modulus;
}

template <class Formulation>
void DiscreteShell<Formulation>::setPoissonRatio(double poisson_ratio)
{
  assert(poisson_ratio > 0 && poisson_ratio <= 0.5);
  for(auto &element: this->_elements)
  {
    element._coeff *= (1 - pow(_poisson_ratio, 2)) / (1 - pow(poisson_ratio, 2));
  }
  _poisson_ratio = poisson_ratio;
}

template <class Formulation>
void DiscreteShell<Formulation>::setThickness(double thickness)
{
  assert(thickness > 0);
  for(auto &element: this->_elements)
  {
    element._coeff *= pow(thickness, 3) / pow(_thickness, 3);
  }
  _thickness = thickness;
}

template class DiscreteShell<TanAngleFormulation>;
template class DiscreteShell<SquaredAngleFormulation>;

} // namespace fsim
