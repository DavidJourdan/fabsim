// vector_utils.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

namespace fsim
{

/**
 * Removes elements in container whose indices are in the indices list
 * @param container
 * @param indices
 * @return the container with its elements removed
 */
template <typename T, typename I>
std::vector<T> remove_from_index_list(const std::vector<T> &container, const std::vector<I> &indices)
{
  std::vector<T> vec = container;
  std::vector<I> index_list = indices;
  // sort index list by descending order
  std::sort(index_list.begin(), index_list.end(), std::greater<I>());
  // erase element from last to first (so that the k-th element in vec is not shifted)
  for(auto k: index_list)
    vec.erase(vec.begin() + k);

  return vec;
}

// Creates a list whose elements are constant vectors of a[i] and sizes b[i].size()
template <typename T, class C>
std::vector<std::vector<T>> constant(const std::vector<T> &a, const std::vector<C> &b)
{
  std::vector<std::vector<T>> _constant;
  for(int i = 0; i < a.size(); ++i)
  {
    _constant.emplace_back(b[i].size() - 1, a[i]);
  }
  return _constant;
}
} // namespace fsim
