// io.cpp
// helper functions for loading and saving mesh and rod data
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/11/19

#include "fsim/util/io.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace fsim
{

/**
 * Read OFF file and store vertex and face information into V and F
 * @param file  path to file where V and F information is stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
void readOFF(const std::string &file, Mat3<double> &V, Mat<int> &F)
{
  std::ifstream mesh_stream(file);
  if(!mesh_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
    return;
  }

  // read OFF header
  std::string line;
  std::getline(mesh_stream, line);
  if(line != "OFF")
  {
    std::cerr << "Header " << line << "not equal to OFF" << std::endl;
    return;
  }
  int nV, nF, _;
  mesh_stream >> nV >> nF >> _;

  // read vertices
  V.resize(nV, 3);
  for(int i = 0; i < nV; ++i)
  {
    double a, b, c;
    mesh_stream >> a >> b >> c;
    V.row(i) << a, b, c;
  }

  // read elements
  int elementSize;
  mesh_stream >> elementSize;
  F.resize(nF, elementSize);

  for(int j = 0; j < elementSize; ++j)
  {
    int k;
    mesh_stream >> k;
    F(0, j) = k;
  }

  for(int i = 1; i < nF; ++i)
  {
    int n;
    mesh_stream >> n;

    if(n != elementSize)
    {
      std::cerr << "Inconsistent element size: " << n << " != " << elementSize << std::endl;
      return;
    }

    for(int j = 0; j < n; ++j)
    {
      int k;
      mesh_stream >> k;
      F(i, j) = k;
    }
  }
}

void readOFF(const std::string &file, Mat3<double> &V, Mat3<int> &F)
{
  std::ifstream mesh_stream(file);
  if(!mesh_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
    return;
  }

  // read OFF header
  std::string line;
  std::getline(mesh_stream, line);
  if(line != "OFF")
  {
    std::cerr << "Header " << line << "not equal to OFF" << std::endl;
    return;
  }
  int nV, nF, _;
  mesh_stream >> nV >> nF >> _;

  // read vertices
  V.resize(nV, 3);
  for(int i = 0; i < nV; ++i)
  {
    double a, b, c;
    mesh_stream >> a >> b >> c;
    V.row(i) << a, b, c;
  }

  // read elements
  F.resize(nF, 3);
  for(int i = 0; i < nF; ++i)
  {
    int n;
    mesh_stream >> n;

    if(n != 3)
    {
      std::cerr << "Inconsistent element size: " << n << " != 3\n";
      return;
    }

    int a, b, c;
    mesh_stream >> a >> b >> c;
    F.row(i) << a, b, c;
  }
}

void readOFF(const std::string &file, Mat3<double> &V, Mat4<int> &F)
{
  std::ifstream mesh_stream(file);
  if(!mesh_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
    return;
  }

  // read OFF header
  std::string line;
  std::getline(mesh_stream, line);
  if(line != "OFF")
  {
    std::cerr << "Header " << line << "not equal to OFF" << std::endl;
    return;
  }
  int nV, nF, _;
  mesh_stream >> nV >> nF >> _;

  // read vertices
  V.resize(nV, 3);
  for(int i = 0; i < nV; ++i)
  {
    double a, b, c;
    mesh_stream >> a >> b >> c;
    V.row(i) << a, b, c;
  }

  // read elements
  F.resize(nF, 4);
  for(int i = 0; i < nF; ++i)
  {
    int n;
    mesh_stream >> n;

    if(n != 4)
    {
      std::cerr << "Inconsistent element size: " << n << " != 4\n";
      return;
    }

    int a, b, c, d;
    mesh_stream >> a >> b >> c >> d;
    F.row(i) << a, b, c, d;
  }
}

/**
 * Read .rod file and store indices into rod_indices
 * @param file  path to .rod file where rod indices information is stored
 * @return rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void readROD(const std::string &file, std::vector<std::vector<int>> &rod_indices)
{
  std::ifstream rod_stream(file);
  if(!rod_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
    return;
  }
  std::string line;
  std::getline(rod_stream, line);
  if(line != "ROD")
  {
    std::cerr << "Header " << line << "not equal to ROD" << std::endl;
    return;
  }
  int nR;
  rod_stream >> nR;
  rod_indices = std::vector<std::vector<int>>(nR);

  for(int i = 0; i < nR; ++i)
  {
    int n;
    rod_stream >> n;
    rod_indices[i].reserve(n);
    for(int j = 0; j < n; ++j)
    {
      int k;
      rod_stream >> k;
      rod_indices[i].push_back(k);
    }
  }
}

/**
 * save vertex and face information in OFF format
 * @param file  path to .off file where V and F information will be stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
void saveOFF(const std::string &file, const Eigen::Ref<const Mat3<double>> V, const Eigen::Ref<const Mat<int>> F)
{
  using namespace Eigen;
  std::ofstream mesh_stream(file);
  if(!mesh_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
  }
  else
  {
    std::string elementSize;
    if(F.cols() == 3)
      elementSize = "3 ";
    else if(F.cols() == 4)
      elementSize = "4 ";
    else
      return;
    
    mesh_stream << "OFF\n"
                << V.rows() << " " << F.rows() << " 0\n"
                << V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "", "", "", "\n"))
                << (F.array()).format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", elementSize, "", "", "\n"));
  }
}

/**
 * save rod indices information in .rod file
 * @param file  path to .rod file where rod indices information will be stored stored
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void saveROD(const std::string &file, const std::vector<std::vector<int>> &rod_indices)
{
  std::ofstream rod_stream(file);
  if(!rod_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
  }
  else
  {
    rod_stream << "ROD\n" << rod_indices.size() << "\n\n";
    for(auto &indices: rod_indices)
    {
      rod_stream << indices.size() << "\n";
      for(int i: indices)
      {
        rod_stream << i << "\n";
      }
      rod_stream << std::endl;
    }
  }
}

/**
 * save both mesh and line geometry in Wavefront OBJ format
 * @param file_name  basename of .off file for V and F, and .rod file for rod_indices
 * @param V  nV by 3 matrix of vertices
 * @param F  nF by 3 matrix of face indices
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void saveOBJ(const std::string &file,
             const Eigen::Ref<const Mat3<double>> V,
             const Eigen::Ref<const Mat<int>> F,
             const std::vector<std::vector<int>> &rod_indices)
{
  using namespace Eigen;
  assert(V.cols() == 3 && "V should have 3 columns");

  std::ofstream mesh_stream(file + ".obj");
  std::ofstream rod_stream(file + "_rod.obj");
  if(!mesh_stream.is_open() || !rod_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
  }
  else
  {
    for(int i = 0; i < V.rows(); ++i)
    {
      mesh_stream << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
      rod_stream << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }

    for(int i = 0; i < F.rows(); ++i)
      mesh_stream << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";

    for(auto &indices: rod_indices)
    {
      rod_stream << "l ";
      for(int i: indices)
      {
        rod_stream << i + 1 << " ";
      }
      rod_stream << std::endl;
    }
  }
}

} // namespace fsim
