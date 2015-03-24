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

#ifndef ADJACENCY_RANSAC_H_
#define ADJACENCY_RANSAC_H_

#include <map>
#include <vector>

#include <opencv2/core/core.hpp>

#include "maximum_clique.h"

namespace tod
{
  class AdjacencyRansac
  {
  public:
    typedef unsigned int Index;
    typedef std::vector<Index> IndexVector;
    AdjacencyRansac()
        :
          object_index_(0),
          min_sample_size_(3)
    {
    }

    void
    FillAdjacency(const std::vector<cv::KeyPoint> & keypoints, float object_span, float sensor_error);

    void
    AddPoints(const cv::Vec3f &training_point, const cv::Vec3f & query_point, unsigned int query_index);

    void
    InvalidateQueryIndices(IndexVector &query_indices);

    inline const IndexVector &
    query_indices() const
    {
      return query_indices_;
    }

    inline unsigned int
    query_indices(unsigned int index) const
    {
      return query_indices_[index];
    }

    void
    Ransac(float sensor_error, unsigned int n_ransac_iterations, IndexVector& inliers, cv::Matx33f &R, cv::Vec3f &T);

    std::string object_id_;
    size_t object_index_;
    tod::maximum_clique::Graph graph_;
    /** matrix indicating whether two points are close enough physically */
    maximum_clique::AdjacencyMatrix physical_adjacency_;
    /** matrix indicating whether two points can be drawn in a RANSAC sample (belong to physical_adjacency but are not
     * too close) */
    maximum_clique::AdjacencyMatrix sample_adjacency_;

  private:
    bool
    DrawSample(IndexVector & valid_samples, unsigned int n_samples, IndexVector & samples) const;

    /** Remove a set of indices from the valid indices and clean the remaining valid indices
     * @param indices the indices to invalidate
     */
    void
    InvalidateIndices(const IndexVector &indices);

    void
    SelectWithinDistance(const cv::Matx33f & R, const cv::Vec3f &T, const IndexVector &samples, float threshold,
                         std::map<Index, Index> &query_inliers);

    /** Estimate the rigid transformation between the training and query points of the sample
     * @param samples the indices of the points forming the samples
     * @param threshold the threshold within which the transformation is valid
     * @param R the output pose rotation
     * @param T the output pose translation
     * @return true if a rigid transformation was found
     */
    bool
    EstimateRigidTransformationSVD(const IndexVector samples, float threshold, cv::Matx33f & R, cv::Vec3f &T) const;

    std::vector<cv::Vec3f> query_points_;
    std::vector<cv::Vec3f> training_points_;
    IndexVector query_indices_;
    /** The list of indices that are actually valid in the current data structures */
    IndexVector valid_indices_;
    /** The minimum sample size when performing RANSAC: 3 is good enough for a rigid pose */
    size_t min_sample_size_;
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  typedef std::map<size_t, AdjacencyRansac> OpenCVIdToObjectPoints;

  void
  ClusterPerObject(const std::vector<cv::KeyPoint> & keypoints, const cv::Mat &point_cloud,
                   const std::vector<std::vector<cv::DMatch> > & matches, const std::vector<cv::Mat> & matches_3d,
                   OpenCVIdToObjectPoints &object_points);

  void
  DrawClustersPerObject(const std::vector<cv::KeyPoint> & keypoints, const std::vector<cv::Scalar> & colors,
                        const cv::Mat & initial_image, const OpenCVIdToObjectPoints &object_points);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

#endif // ADJACENCY_RANSAC_H_ 
