// io.h
// helper functions for loading and saving mesh and rod data
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/11/19

#pragma once

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <vector>

namespace fsim
{

/**
 * Read OFF file and store vertex and face information into V and F
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @tparam DerivedV  Eigen::Matrix derived type
 * @tparam DerivedF  Eigen::Matrix derived type
 * @param file  path to file where V and F information is stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
template <class File, class DerivedV, class DerivedF>
void read_from_file(const File &file, Eigen::PlainObjectBase<DerivedV> &V, Eigen::PlainObjectBase<DerivedF> &F)
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

  // read faces
  F.resize(nF, 3);
  for(int i = 0; i < nF; ++i)
  {
    int a, b, c;
    mesh_stream >> _ >> a >> b >> c;
    F.row(i) << a, b, c;
  }
}

/**
 * Read .rod file and store indices into rod_indices
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @param file  path to .rod file where rod indices information is stored
 * @return rod_indices  nested vector: each vector represents a rod and contains its indices
 */
template <class File>
void read_from_file(const File &file, std::vector<std::vector<int>> &rod_indices)
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
 * Read off and rod files and store both rod indices and mesh information
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @tparam DerivedV  Eigen::Matrix derived type
 * @tparam DerivedF  Eigen::Matrix derived type
 * @param file_name  basename of .off file where V and F information is stored, and .rod file where rod_indices
 * information is stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 * @return rod_indices  nested vector: each vector represents a rod and contains its indices
 */
template <class File, class DerivedV, class DerivedF>
void read_from_file(const File &file_name,
                    Eigen::PlainObjectBase<DerivedV> &V,
                    Eigen::PlainObjectBase<DerivedF> &F,
                    std::vector<std::vector<int>> &rod_indices)
{
  // read OFF header
  std::string mesh_file_name = std::string(file_name) + ".off";
  read_from_file(mesh_file_name, V, F);

  // read rods
  std::string rod_file_name = std::string(file_name) + ".rod";
  read_from_file(rod_file_name, rod_indices);
}

/**
 * save vertex and face information in OFF format
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @tparam DerivedV  Eigen::Matrix derived type
 * @tparam DerivedF  Eigen::Matrix derived type
 * @param file  path to .off file where V and F information will be stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
template <class File, class DerivedV, class DerivedF>
void save_to_file(const File &file, const Eigen::MatrixBase<DerivedV> &V, const Eigen::MatrixBase<DerivedF> &F)
{
  using namespace Eigen;
  assert(V.cols() == 3 && "V should have 3 columns");

  std::ofstream mesh_stream(file);
  if(!mesh_stream.is_open())
  {
    std::cerr << "IOError: could not open " << file << std::endl;
  }
  else
  {
    mesh_stream << "OFF\n"
                << V.rows() << " " << F.rows() << " 0\n"
                << V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "", "", "", "\n"))
                << (F.array()).format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "3 ", "", "", "\n"));
  }
}

/**
 * save rod indices information in .rod file
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @param file  path to .rod file where rod indices information will be stored stored
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
template <class File>
void save_to_file(const File &file, const std::vector<std::vector<int>> &rod_indices)
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
 * save both mesh information and rod indices in 2 separate files
 * @tparam File  can be a string representing the file name or a std::filesystem object (c++17)
 * @tparam DerivedV  Eigen::Matrix derived type
 * @tparam DerivedF  Eigen::Matrix derived type
 * @param file_name  basename of .off file for V and F, and .rod file for rod_indices
 * @param V  nV by 3 matrix of vertices
 * @param F  nF by 3 matrix of face indices
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
template <class File, class DerivedV, class DerivedF>
void save_to_file(const File &file_name,
                  const Eigen::MatrixBase<DerivedV> &V,
                  const Eigen::MatrixBase<DerivedF> &F,
                  const std::vector<std::vector<int>> &rod_indices)
{
  // save mesh to an OFF file
  save_to_file(std::string(file_name) + ".off", V, F);

  // save rod indices
  save_to_file(std::string(file_name) + ".rod", rod_indices);
}

/**
 * save both mesh and line geometry in Wavefront OBJ format
 * @tparam DerivedV  Eigen::Matrix derived type
 * @tparam DerivedF  Eigen::Matrix derived type
 * @param file_name  basename of .off file for V and F, and .rod file for rod_indices
 * @param V  nV by 3 matrix of vertices
 * @param F  nF by 3 matrix of face indices
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
template <class DerivedV, class DerivedF>
void save_to_obj(const std::string &file,
                 const Eigen::MatrixBase<DerivedV> &V,
                 const Eigen::MatrixBase<DerivedF> &F,
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
