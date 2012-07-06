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
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"
#include "sac_model.h"

#include "maximum_clique.h"

namespace tod
{
  /**
   * Class that computes the registration between two point clouds in the specific case where we have an adjacency graph
   * (and some points cannot be connected together)
   */
  template<typename PointT>
  class SampleConsensusModelRegistrationGraph: public pcl::SampleConsensusModel<PointT>
  {
    using pcl::SampleConsensusModel<PointT>::input_;
    using pcl::SampleConsensusModel<PointT>::indices_;

  public:
    typedef typename pcl::SampleConsensusModel<PointT>::PointCloud PointCloud;
    typedef typename pcl::SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
    typedef typename pcl::SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
    typedef boost::shared_ptr<SampleConsensusModelRegistrationGraph> Ptr;
    typedef const Eigen::Map<const Eigen::Vector4f, Eigen::Aligned> Vector4fMapConst;

    using pcl::SampleConsensusModel<PointT>::drawIndexSample;

    /** \brief Constructor for base SampleConsensusModelRegistration.
     * \param cloud the input point cloud dataset
     * \param indices a vector of point indices to be used from \a cloud
     */
    SampleConsensusModelRegistrationGraph(
        const PointCloudConstPtr &cloud, const std::vector<int> &indices, float threshold,
        const maximum_clique::AdjacencyMatrix & physical_adjacency,
        const maximum_clique::AdjacencyMatrix &sample_adjacency)
        :
          pcl::SampleConsensusModel<PointT>(cloud, indices),
          physical_adjacency_(physical_adjacency),
          sample_adjacency_(sample_adjacency),
          best_inlier_number_(0),
          threshold_(threshold)
    {
      input_ = cloud;
      computeSampleDistanceThreshold (cloud);

      BuildNeighbors();
    }


    /** \brief Computes an "optimal" sample distance threshold based on the
      * principal directions of the input cloud.
      * \param cloud the const boost shared pointer to a PointCloud message
      */
    inline void
    computeSampleDistanceThreshold (const PointCloudConstPtr &cloud)
    {
      // Compute the principal directions via PCA
      Eigen::Vector4f xyz_centroid;
      compute3DCentroid (*cloud, xyz_centroid);
      EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
      computeCovarianceMatrixNormalized (*cloud, xyz_centroid, covariance_matrix);
      EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
      EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
      pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);

      // Compute the distance threshold for sample selection
      sample_dist_thresh_ = eigen_values.array ().sqrt ().sum () / 3.0;
      sample_dist_thresh_ *= sample_dist_thresh_;
      PCL_DEBUG ("[pcl::SampleConsensusModelRegistration::setInputCloud] Estimated a sample selection distance threshold of: %f\n", sample_dist_thresh_);
    }


    /** \brief Set the input point cloud target.
      * \param target the input point cloud target
      * \param indices_tgt a vector of point indices to be used from \a target
      */
    inline void
    setInputTarget (const PointCloudConstPtr &target, const std::vector<int> &indices_tgt)
    {
      target_ = target;
      indices_tgt_.reset (new std::vector<int> (indices_tgt));
    }

    bool
    drawIndexSampleHelper(std::vector<int> & valid_samples, unsigned int n_samples, std::vector<int> & samples) const
    {
      if (n_samples == 0)
        return true;
      if (valid_samples.empty())
        return false;
      while (true)
      {
        int sample = valid_samples[rand() % valid_samples.size()];
        std::vector<int> new_valid_samples(valid_samples.size());
        std::vector<int>::iterator end = std::set_intersection(valid_samples.begin(), valid_samples.end(),
                                                               sample_adjacency_.neighbors(sample).begin(),
                                                               sample_adjacency_.neighbors(sample).end(),
                                                               new_valid_samples.begin());
        new_valid_samples.resize(end - new_valid_samples.begin());
        std::vector<int> new_samples;
        if (drawIndexSampleHelper(new_valid_samples, n_samples - 1, new_samples))
        {
          samples = new_samples;
          valid_samples = new_valid_samples;
          samples.push_back(sample);
          return true;
        }
        else
        {
          std::vector<int>::iterator end = std::remove(valid_samples.begin(), valid_samples.end(), sample);
          valid_samples.resize(end - valid_samples.begin());
          if (valid_samples.empty())
            return false;
        }
      }
      return false;
    }

    bool
    isSampleGood(const std::vector<int> &samples) const
    {
      std::vector<int> valid_samples = sample_pool_;
      std::vector<int> &new_samples = const_cast<std::vector<int> &>(samples);
      size_t sample_size = new_samples.size();
      bool is_good = drawIndexSampleHelper(valid_samples, sample_size, new_samples);

      if (is_good)
        samples_ = new_samples;

      return is_good;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  selectWithinDistance(const cv::Matx33f &R, const cv::Vec3f&T, double threshold, std::vector<int> &in_inliers)
  {
    std::vector<int> inliers;
    if (indices_->size () != indices_tgt_->size ())
    {
      PCL_ERROR ("[pcl::SampleConsensusModelRegistration::selectWithinDistance] Number of source indices (%lu) differs than number of target indices (%lu)!\n", (unsigned long)indices_->size (), (unsigned long)indices_tgt_->size ());
      inliers.clear ();
      return;
    }

    double thresh = threshold * threshold;

    inliers.resize (indices_->size ());

    int nr_p = 0;
    for (size_t i = 0; i < indices_->size (); ++i)
    {
      Vector4fMapConst pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
      Vector4fMapConst pt_tgt = target_->points[(*indices_tgt_)[i]].getVector4fMap ();
      cv::Vec3f p_tr  = R * cv::Vec3f(pt_src(0), pt_src(1), pt_src(2)) + T;
      // Calculate the distance from the transformed point to its correspondence
      if (cv::norm(p_tr - cv::Vec3f(pt_tgt(0), pt_tgt(1), pt_tgt(2)))*cv::norm(p_tr-cv::Vec3f(pt_tgt(0), pt_tgt(1), pt_tgt(2))) < thresh)
        inliers[nr_p++] = (*indices_)[i];
    }
    inliers.resize (nr_p);






    in_inliers.clear();
    // Make sure the sample belongs to the inliers
    BOOST_FOREACH(int sample, samples_)
    if (std::find(inliers.begin(), inliers.end(), sample) == inliers.end())
    return;

    // Remove all the points that cannot belong to a clique including the samples
    BOOST_FOREACH(int inlier, inliers)
    {
      bool is_good = true;
      BOOST_FOREACH(int sample, samples_)
      {
        if (sample == inlier)
        break;
        if (!physical_adjacency_.test(inlier, sample))
        {
          is_good = false;
          break;
        }
      }
      if (is_good)
      in_inliers.push_back(inlier);
    }

    // If that set is not bigger than the best so far, no need to refine it
    if (in_inliers.size() < best_inlier_number_)
    return;

    maximum_clique::Graph graph(in_inliers.size());
    for (unsigned int j = 0; j < in_inliers.size(); ++j)
    for (unsigned int i = j + 1; i < in_inliers.size(); ++i)
    if (sample_adjacency_.test(in_inliers[j], in_inliers[i]))
    graph.AddEdgeSorted(j, i);

    // If we cannot even find enough points well distributed in the sample, stop here
    unsigned int minimal_size = 8;
    std::vector<unsigned int> vertices;
    graph.FindClique(vertices, minimal_size);
    if (vertices.size() < minimal_size)
    {
      in_inliers.clear();
      return;
    }

    best_inlier_number_ = std::max(in_inliers.size(), best_inlier_number_);
  }

  bool
  computeModelCoefficients (const std::vector<int> &samples, cv::Matx33f &R, cv::Vec3f&T)
  {
    // Need 3 samples
    if (samples.size () != 3)
      return (false);

    estimateRigidTransformationSVD(*input_, samples, *target_, samples, R, T);

    return (true);
  }

  void
  optimizeModelCoefficients(const PointCloudConstPtr &target, const std::vector<int> &inliers,
      cv::Matx33f&R, cv::Vec3f&T)
  {
    estimateRigidTransformationSVD(*input_, inliers, *target, inliers, R, T);
  }

  mutable std::vector<int> samples_;

  /** \brief Estimate a rigid transformation between a source and a target point cloud using an SVD closed-form
   * solution of absolute orientation using unit quaternions
   * \param[in] cloud_src the source point cloud dataset
   * \param[in] indices_src the vector of indices describing the points of interest in cloud_src
   * \param[in] cloud_tgt the target point cloud dataset
   * \param[in] indices_tgt the vector of indices describing the correspondences of the interest points from
   * indices_src
   * \param[out] transform the resultant transformation matrix (as model coefficients)
   *
   * This method is an implementation of: Horn, B. “Closed-Form Solution of Absolute Orientation Using Unit Quaternions,” JOSA A, Vol. 4, No. 4, 1987
   * THIS IS COPIED STRAIGHT UP FROM PCL AS THEY CHANGED THE API ANDMADE IT PRIVATE
   */
  void
  estimateRigidTransformationSVD(const typename pcl::PointCloud<PointT> &cloud_src,
      const std::vector<int> &indices_src,
      const typename pcl::PointCloud<PointT> &cloud_tgt,
      const std::vector<int> &indices_tgt, cv::Matx33f &R_in, cv::Vec3f&T)
  {
    Eigen::Vector4f centroid_src, centroid_tgt;
    // Estimate the centroids of source, target
    compute3DCentroid(cloud_src, indices_src, centroid_src);
    compute3DCentroid(cloud_tgt, indices_tgt, centroid_tgt);

    // Subtract the centroids from source, target
    Eigen::MatrixXf cloud_src_demean;
    demeanPointCloud(cloud_src, indices_src, centroid_src, cloud_src_demean);

    Eigen::MatrixXf cloud_tgt_demean;
    demeanPointCloud(cloud_tgt, indices_tgt, centroid_tgt, cloud_tgt_demean);

    // Assemble the correlation matrix H = source * target'
    Eigen::Matrix3f H = (cloud_src_demean * cloud_tgt_demean.transpose()).topLeftCorner<3, 3>();

    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Matrix3f v = svd.matrixV();

    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0)
    {
      for (int x = 0; x < 3; ++x)
      v(x, 2) *= -1;
    }

    Eigen::Matrix3f R = v * u.transpose();

    // Return the correct transformation
    Eigen::Vector3f t = centroid_tgt.head<3>() - R * centroid_src.head<3>();
    for(unsigned char j=0;j<3;++j) {
      for(unsigned char i=0;i<3;++i)
        R_in(j,i) = R(j,i);
      T[j] = t[j];
    }
  }
  private:
  void
  BuildNeighbors()
  {
    size_t max_neighbors_size = 10;
    for (unsigned int j = 0; j < sample_adjacency_.size(); ++j)
    {
      size_t size = sample_adjacency_.neighbors(j).size();
      max_neighbors_size = std::max(max_neighbors_size, size);
      if (size >= 3)
      sample_pool_.push_back(j);
    }
    if (!indices_->empty())
    {
      std::vector<int>::iterator end = std::set_intersection(sample_pool_.begin(), sample_pool_.end(),
          indices_->begin(), indices_->end(),
          sample_pool_.begin());
      sample_pool_.resize(end - sample_pool_.begin());
    }
  }

  const maximum_clique::AdjacencyMatrix physical_adjacency_;
  const maximum_clique::AdjacencyMatrix sample_adjacency_;
  std::vector<int> sample_pool_;
  size_t best_inlier_number_;
  float threshold_;


  /** \brief A boost shared pointer to the target point cloud data array. */
  PointCloudConstPtr target_;

  /** \brief A pointer to the vector of target point indices to use. */
  boost::shared_ptr <std::vector<int> > indices_tgt_;

  /** \brief Internal distance threshold used for the sample selection step. */
  double sample_dist_thresh_;

  };}

#endif
