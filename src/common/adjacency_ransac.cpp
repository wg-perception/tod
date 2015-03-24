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

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "adjacency_ransac.h"
#include "ransac.h"
#include "sac_model_registration_graph.h"

namespace tod
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  AdjacencyRansac::AddPoints(const cv::Vec3f &training_point, const cv::Vec3f & query_point, unsigned int query_index)
  {
    valid_indices_.push_back(query_indices_.size());

    training_points_.push_back(training_point);
    query_points_.push_back(query_point);
    query_indices_.push_back(query_index);
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  AdjacencyRansac::InvalidateIndices(const IndexVector &indices)
  {
    IndexVector indices_to_invalidate = indices;

    while (!indices_to_invalidate.empty())
    {
      // Remove the indices from the valid ones
      std::sort(indices_to_invalidate.begin(), indices_to_invalidate.end());
      indices_to_invalidate.resize(
          std::unique(indices_to_invalidate.begin(), indices_to_invalidate.end()) - indices_to_invalidate.begin());

      IndexVector::iterator end = std::set_difference(valid_indices_.begin(), valid_indices_.end(),
                                                      indices_to_invalidate.begin(), indices_to_invalidate.end(),
                                                      valid_indices_.begin());
      valid_indices_.resize(end - valid_indices_.begin());

      // Reset the matrices
      physical_adjacency_.InvalidateCluster(indices_to_invalidate);
      sample_adjacency_.InvalidateCluster(indices_to_invalidate);

      // Go over the valid indices and remove the ones that do not have enough neighbors
      indices_to_invalidate.clear();
      BOOST_FOREACH(Index index, valid_indices_)if (sample_adjacency_.neighbors(index).size() < min_sample_size_)
      indices_to_invalidate.push_back(index);
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  AdjacencyRansac::InvalidateQueryIndices(IndexVector &query_indices)
  {
    if (query_indices.empty())
      return;
    // Figure out the points with those query indices
    std::sort(query_indices.begin(), query_indices.end());
    IndexVector::iterator end = std::unique(query_indices.begin(), query_indices.end());
    query_indices.resize(end - query_indices.begin());

    IndexVector indices_to_remove;
    indices_to_remove.reserve(query_indices_.size());
    IndexVector::const_iterator iter = query_indices.begin();
    BOOST_FOREACH(unsigned int index, valid_indices_){
    unsigned int query_index = query_indices_[index];
    if (query_index < *iter)
    continue;
    // If the match has a keypoint in the inliers, remove the match
    while ((iter != end) && (query_index > *iter))
    ++iter;
    if (query_index == *iter)
    {
      indices_to_remove.push_back(index);
      continue;
    }

    if (iter == end)
    break;
  }
    InvalidateIndices(indices_to_remove);
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  AdjacencyRansac::FillAdjacency(const std::vector<cv::KeyPoint> & keypoints, float object_span, float sensor_error)
  {
    // The error the 3d sensor makes, distance wise
    unsigned int n_matches = training_points_.size();
    physical_adjacency_ = maximum_clique::AdjacencyMatrix(n_matches);
    sample_adjacency_ = maximum_clique::AdjacencyMatrix(n_matches);
    std::vector<cv::Vec3f>::const_iterator query_point_1 = query_points_.begin(), training_point_1 =
        training_points_.begin(), query_point_2;
    for (unsigned int i = 0; i < n_matches; ++i, ++query_point_1, ++training_point_1)
    {
      // For every other match that might end up in the same cluster
      query_point_2 = query_point_1 + 1;
      for (unsigned int j = i + 1; j < n_matches; ++j, ++query_point_2)
      {
        // Two training points can be connected if they are within the span of an object
        float dist_query = distSq(*query_point_1, *query_point_2);
        if (dist_query > (object_span + 2 * sensor_error) * (object_span + 2 * sensor_error))
          continue;
        dist_query = std::sqrt(dist_query);

        const cv::Vec3f & training_point_2 = training_points_[j];
        float dist_training = cv::norm(*training_point_1 - training_point_2);
        // Make sure the distance between two points is somewhat conserved
        if (std::abs(dist_training - dist_query) > 4 * sensor_error)
          continue;

        // If all those conditions are respected, those two matches are potentially part of the same cluster
        physical_adjacency_.set_sorted(i, j);

        const cv::KeyPoint & keypoint1 = keypoints[query_indices_[i]], &keypoint2 = keypoints[query_indices_[j]];
        if ((((keypoint1.pt.x - keypoint2.pt.x) * (keypoint1.pt.x - keypoint2.pt.x)
            + (keypoint1.pt.y - keypoint2.pt.y) * (keypoint1.pt.y - keypoint2.pt.y))
             > 20 * 20)
            && (std::abs(dist_training - dist_query) < 2 * sensor_error))
        //((dist_query >= 5 * sensor_error) && (dist_training >= 5 * sensor_error))
        {
          sample_adjacency_.set_sorted(i, j);
        }
      }
    }

    // Clean the valid indices
    IndexVector indices;
    InvalidateIndices(indices);
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  ClusterPerObject(const std::vector<cv::KeyPoint> & keypoints, const cv::Mat &point_cloud,
                   const std::vector<std::vector<cv::DMatch> > & matches, const std::vector<cv::Mat> & matches_3d,
                   OpenCVIdToObjectPoints &object_points)
  {
    for (unsigned int query_index = 0; query_index < matches.size(); ++query_index)
    {
      // Figure out the 3d query point
      const cv::KeyPoint & keypoint = keypoints[query_index];
      const cv::Vec3f &query_point = point_cloud.at<cv::Vec3f>(keypoint.pt.y, keypoint.pt.x);

      // Make sure it does not contain any NaN's
      // We could have a solver that would consider Nan's as missing entries
      if (cvIsNaN(query_point[0]))
        continue;

      const std::vector<cv::DMatch> &local_matches = matches[query_index];
      const cv::Mat &local_matches_3d = matches_3d[query_index];

      // Get the matches for that point
      for (unsigned int match_index = 0; match_index < local_matches.size(); ++match_index)
      {
        const cv::Vec3f & training_point = local_matches_3d.at<cv::Vec3f>(0, match_index);

        // Fill in the clouds
        size_t opencv_object_id = local_matches[match_index].imgIdx;
        object_points[opencv_object_id].AddPoints(training_point, query_point, query_index);
      }
    }
  }

  void
  DrawClustersPerObject(const std::vector<cv::KeyPoint> & keypoints, const std::vector<cv::Scalar> & colors,
                        const cv::Mat & initial_image, const OpenCVIdToObjectPoints &object_points)
  {
    cv::Mat out_img = initial_image.clone();
    unsigned int i = 0;
    // Draw the keypoints with a different color per object
    for (OpenCVIdToObjectPoints::const_iterator query_iterator = object_points.begin();
        query_iterator != object_points.end(); ++query_iterator)
    {
      AdjacencyRansac::IndexVector query_indices = query_iterator->second.query_indices();
      AdjacencyRansac::IndexVector::iterator end = std::unique(query_indices.begin(), query_indices.end());
      query_indices.resize(end - query_indices.begin());
      std::vector<cv::KeyPoint> local_keypoints(query_indices.size());
      for (unsigned int j = 0; j < query_indices.size(); ++j)
        local_keypoints[j] = keypoints[query_indices[j]];
      cv::drawKeypoints(out_img, local_keypoints, out_img, colors[i]);
      ++i;
      if (i >= colors.size())
        break;
    }
    cv::namedWindow("keypoints from objects", 0);
    cv::imshow("keypoints from objects", out_img);
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void
  AdjacencyRansac::Ransac(float sensor_error, unsigned int n_ransac_iterations, IndexVector& inliers_in, cv::Matx33f &R,
                          cv::Vec3f &T)
  {
    if (valid_indices_.size() < 3) {
      inliers_in.clear();
      return;
    }

    // Perform RANSAC on the input clouds, making sure to include adjacent pairs in the samples
    SampleConsensusModelRegistrationGraph::Ptr model(
        new SampleConsensusModelRegistrationGraph(query_points_, training_points_, valid_indices_, sensor_error,
                                                  physical_adjacency_, sample_adjacency_));
    pcl::RandomSampleConsensus sample_consensus(model);

    sample_consensus.setMaxIterations(n_ransac_iterations);

    inliers_in.clear();
    if (!sample_consensus.computeModel())
      return;

    std::vector<unsigned int> inliers;
    BOOST_FOREACH(unsigned int inl, inliers_in)inliers.push_back(inl);
    sample_consensus.getInliers(inliers);
    std::sort(inliers.begin(), inliers.end());
    sample_consensus.getModelCoefficients(R, T);
    std::vector<unsigned int> valid_indices_vect = valid_indices_;
    std::vector<unsigned int>::iterator valid_end = std::set_difference(valid_indices_vect.begin(),
                                                                        valid_indices_vect.end(), inliers.begin(),
                                                                        inliers.end(), valid_indices_vect.begin());
    valid_indices_vect.resize(valid_end - valid_indices_vect.begin());

    bool do_final = false;
    double thresh = sensor_error * sensor_error;
    // Try to bring more points to the model, without removing points, which could bias the model
    // Also, for the last iteration (when we cannot add points anymore), we get a looser threshold
    while (true)
    {
      model->optimizeModelCoefficients(inliers, R, T);

      // Check if the model is valid given the user constraints
      std::vector<int> extra_inliers;
      BOOST_FOREACH(int index, valid_indices_vect){
      const cv::Vec3f &pt_src = query_points_[index];
      const cv::Vec3f &pt_tgt = training_points_[index];
      cv::Vec3f p_tr = R * pt_src + T;
      // Calculate the distance from the transformed point to its correspondence
      if (cv::norm(p_tr - pt_tgt)*cv::norm(p_tr - pt_tgt) < thresh)
      extra_inliers.push_back(index);
    }

    // Add those extra inliers to the inliers and remove them from the valid indices
      {
        std::vector<unsigned int> tmp_inliers = inliers;
        inliers.resize(inliers.size() + extra_inliers.size());
        std::merge(tmp_inliers.begin(), tmp_inliers.end(), extra_inliers.begin(), extra_inliers.end(), inliers.begin());
      }
      valid_end = std::set_difference(valid_indices_vect.begin(), valid_indices_vect.end(), extra_inliers.begin(),
                                      extra_inliers.end(), valid_indices_vect.begin());
      valid_indices_vect.resize(valid_end - valid_indices_vect.begin());

      if (do_final)
        break;
      if (extra_inliers.empty())
      {
        do_final = true;
        thresh *= 4;
      }

    }
    R = R.t();
    T = -R * T;
    BOOST_FOREACH(unsigned int inl, inliers)inliers_in.push_back(query_indices_[inl]);
    std::sort(inliers_in.begin(), inliers_in.end());
    inliers_in.resize(std::unique(inliers_in.begin(), inliers_in.end()) - inliers_in.begin());
  }
}
