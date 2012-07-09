/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef SAC_MODEL_REGISTRATION_GRAPH_H_
#define SAC_MODEL_REGISTRATION_GRAPH_H_

#include <opencv2/core/core.hpp>

#include "sac_model.h"

#include "maximum_clique.h"

namespace
{
  /** Compute the squared distance between two vectors
   * @param vec1
   * @param vec2
   * @return
   */
  inline
  float
  distSq(const cv::Vec3f &vec1, const cv::Vec3f & vec2)
  {
    cv::Vec3f tmp = vec1 - vec2;
    return tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2];
  }
}

namespace tod
{
  /**
   * Class that computes the registration between two point clouds in the specific case where we have an adjacency graph
   * (and some points cannot be connected together)
   */
  class SampleConsensusModelRegistrationGraph: public pcl::SampleConsensusModel
  {
    using pcl::SampleConsensusModel::indices_;
    using pcl::SampleConsensusModel::shuffled_indices_;

  public:
    using pcl::SampleConsensusModel::drawIndexSample;
    typedef unsigned int Index;
    typedef std::vector<Index> IndexVector;

    /** \brief Constructor for base SampleConsensusModelRegistration.
     * \param cloud the input point cloud dataset
     * \param indices a vector of point indices to be used from \a cloud
     */
    SampleConsensusModelRegistrationGraph(const std::vector<cv::Vec3f> &query_points,
                                          const std::vector<cv::Vec3f> &target, const IndexVector &indices,
                                          float threshold, const maximum_clique::AdjacencyMatrix & physical_adjacency,
                                          const maximum_clique::AdjacencyMatrix &sample_adjacency)
        :
          physical_adjacency_(physical_adjacency),
          sample_adjacency_(sample_adjacency),
          best_inlier_number_(8),
          threshold_(threshold)
    {
      indices_ = indices;
      shuffled_indices_ = indices;
      query_points_ = query_points;
      training_points_ = target;
    }

    bool
    drawIndexSampleHelper(IndexVector & valid_samples, unsigned int n_samples) const
    {
      if (n_samples == 0)
        return true;
      if (valid_samples.empty())
        return false;
      while (true)
      {
        int sample = valid_samples[rand() % valid_samples.size()];
        IndexVector new_valid_samples(valid_samples.size());
        IndexVector::iterator end = std::set_intersection(valid_samples.begin(), valid_samples.end(),
                                                               sample_adjacency_.neighbors(sample).begin(),
                                                               sample_adjacency_.neighbors(sample).end(),
                                                               new_valid_samples.begin());
        new_valid_samples.resize(end - new_valid_samples.begin());
        if (drawIndexSampleHelper(new_valid_samples, n_samples - 1))
        {
          samples_.push_back(sample);
          return true;
        }
        else
        {
          IndexVector::iterator end = std::remove(valid_samples.begin(), valid_samples.end(), sample);
          valid_samples.resize(end - valid_samples.begin());
          if (valid_samples.empty())
            return false;
        }
      }
      return false;
    }

    bool
    isSampleGood(const IndexVector &samples) const
    {
      IndexVector valid_samples = indices_;
      size_t sample_size = samples.size();
      const_cast<IndexVector &>(samples_).clear();
      bool is_good = drawIndexSampleHelper(valid_samples, sample_size);

      if (is_good)
        const_cast<IndexVector &>(samples) = samples_;

      return is_good;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    selectWithinDistance(const cv::Matx33f &R, const cv::Vec3f&T, double threshold, IndexVector &inliers)
    {
      if (samples_.empty())
        return;

      // First, figure out the common neighbors of all the samples
      IndexVector possible_inliers = physical_adjacency_.neighbors(samples_[0]);
      for (unsigned int i = 1; i < samples_.size(); ++i)
        possible_inliers.resize(
            std::set_intersection(possible_inliers.begin(), possible_inliers.end(),
                                  physical_adjacency_.neighbors(samples_[i]).begin(),
                                  physical_adjacency_.neighbors(samples_[i]).end(), possible_inliers.begin())
            - possible_inliers.begin());
      for (unsigned int i = 0; i < samples_.size(); ++i)
        possible_inliers.push_back(samples_[i]);

      // Then, check which ones of those verify the geometric constraint
      inliers.resize(possible_inliers.size());

      int nr_p = 0;
      for (size_t i = 0; i < possible_inliers.size(); ++i)
      {
        const cv::Vec3f & pt_src = query_points_[possible_inliers[i]];
        const cv::Vec3f & pt_tgt = training_points_[possible_inliers[i]];
        // Calculate the distance from the transformed point to its correspondence
        if (distSq(R * pt_src + T, pt_tgt) < threshold * threshold)
          inliers[nr_p++] = possible_inliers[i];
      }
      inliers.resize(nr_p);

      // If that set is not bigger than the best so far, no need to refine it
      unsigned int minimal_size = 8;
      if ((inliers.size() < best_inlier_number_) && (inliers.size() < minimal_size))
        return;
      std::sort(inliers.begin(), inliers.end());

      // We are now going to check that we can come up with a big enough sample adjacency clique
      // As this rarely happens, first make sure that some inliers have enough neighbors
      size_t max_possible_clique = 0;
      std::vector<unsigned int> neighbors(inliers.size());
      for (unsigned int j = 0; j < inliers.size() - 1; ++j)
      {
        max_possible_clique = std::max(
            size_t(
                std::set_intersection(sample_adjacency_.neighbors(j).begin(), sample_adjacency_.neighbors(j).end(),
                                      inliers.begin() + j + 1, inliers.end(), neighbors.begin())
                - neighbors.begin()),
            max_possible_clique);
      }
      if ((max_possible_clique < minimal_size) || (max_possible_clique < best_inlier_number_))
      {
        inliers.clear();
        return;
      }

      // look for big enough cliques
      std::map<unsigned int, unsigned int> map_index_to_graph_index;
      for (unsigned int j = 0; j < inliers.size(); ++j)
        map_index_to_graph_index[inliers[j]] = j;

      maximum_clique::Graph graph(inliers.size());
      for (unsigned int j = 0; j < inliers.size() - 1; ++j)
      {
        neighbors.resize(inliers.size());
        std::vector<unsigned int>::iterator end = std::set_intersection(sample_adjacency_.neighbors(j).begin(),
                                                                        sample_adjacency_.neighbors(j).end(),
                                                                        inliers.begin() + j, inliers.end(),
                                                                        neighbors.begin());
        neighbors.resize(end - neighbors.begin());

        BOOST_FOREACH(unsigned int neighbor, neighbors){
        if (j== map_index_to_graph_index[neighbor])
        continue;
        graph.AddEdgeSorted(j, map_index_to_graph_index[neighbor]);
      }
    }

    // If we cannot even find enough points well distributed in the sample, stop here
      std::vector<unsigned int> vertices;
      graph.FindClique(vertices, minimal_size);
      if (vertices.size() < minimal_size)
      {
        inliers.clear();
        return;
      }

      best_inlier_number_ = std::max(inliers.size(), best_inlier_number_);
    }

    bool
    computeModelCoefficients(const IndexVector &samples, cv::Matx33f &R, cv::Vec3f&T)
    {
      // Need 3 samples
      if (samples.size() != 3)
        return (false);

      if (!estimateRigidTransformationSVD(samples, R, T))
        return false;

      // Make sure the sample do verify the transform
      /*BOOST_FOREACH(Index sample, samples){
      if (distSq(R*query_points_[sample] + T, training_points_[sample])>threshold_*threshold_)
      return false;
    }*/

      return true;
    }

    void
    optimizeModelCoefficients(const IndexVector &inliers, cv::Matx33f&R, cv::Vec3f&T)
    {
      estimateRigidTransformationSVD(inliers, R, T);
    }

    mutable IndexVector samples_;

    /** \brief Estimate a rigid transformation between a source and a target point cloud using an SVD closed-form
     * solution of absolute orientation using unit quaternions
     * \param[in] indices_src the vector of indices describing the points of interest in cloud_src
     * \param[in] R the rotation part of the transform
     * \param[out] T the translation part of the transform
     *
     * This method is an implementation of: Horn, B. “Closed-Form Solution of Absolute Orientation Using Unit Quaternions,” JOSA A, Vol. 4, No. 4, 1987
     */
    bool
    estimateRigidTransformationSVD(const IndexVector &indices_src, cv::Matx33f &R_in, cv::Vec3f&T)
    {
      if (indices_src.size() < 3)
        return false;

      cv::Vec3f centroid_training(0, 0, 0), centroid_query(0, 0, 0);

      // Estimate the centroids of source, target
      BOOST_FOREACH(Index index, indices_src)
      {
        centroid_training += training_points_[index];
        centroid_query += query_points_[index];
      }
      centroid_training /= float(indices_src.size());
      centroid_query /= float(indices_src.size());

      // Subtract the centroids from source, target
      cv::Mat_<cv::Vec3f> sub_training(indices_src.size(),1), sub_query(indices_src.size(),1);
      unsigned int i = 0;
      BOOST_FOREACH(Index index, indices_src)
      {
        sub_training(i) = training_points_[index] - centroid_training;
        sub_query(i) = query_points_[index] - centroid_query;
        ++i;
      }

      // Assemble the correlation matrix
      cv::Mat H = sub_training.reshape(1, indices_src.size()).t() * sub_query.reshape(1, indices_src.size());

      // Compute the Singular Value Decomposition
      cv::SVD svd(H);

      // Compute R = U * V'
      cv::Mat_<float> vt = cv::Mat(svd.vt);
      if (cv::determinant(svd.u) * cv::determinant(vt) < 0)
      {
        for (int x = 0; x < 3; ++x)
        vt(2, x) *= -1;
      }

      R_in = cv::Mat(svd.u * vt);
      T = centroid_training - R_in * centroid_query;

      return true;
    }

  private:

    const maximum_clique::AdjacencyMatrix &physical_adjacency_;
    const maximum_clique::AdjacencyMatrix &sample_adjacency_;
  size_t best_inlier_number_;
  float threshold_;

    std::vector<cv::Vec3f> query_points_;
    std::vector<cv::Vec3f> training_points_;
  };
}

#endif
