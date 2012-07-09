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
 * $Id: ransac.h 1632 2011-07-08 16:36:36Z nizar $
 *
 */

#ifndef PCL_SAMPLE_CONSENSUS_RANSAC_H_
#define PCL_SAMPLE_CONSENSUS_RANSAC_H_

#include "sac.h"
#include "sac_model_registration_graph.h"

namespace pcl
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief @b RandomSampleConsensus represents an implementation of the RANSAC (RAndom SAmple Consensus) algorithm, as 
    * described in: "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and 
    * Automated Cartography", Martin A. Fischler and Robert C. Bolles, Comm. Of the ACM 24: 381–395, June 1981.
    * \author Radu Bogdan Rusu
    * \ingroup sample_consensus
    */
  class RandomSampleConsensus : public SampleConsensus
  {
    using SampleConsensus::max_iterations_;
    using SampleConsensus::threshold_;
    using SampleConsensus::iterations_;
    using SampleConsensus::sac_model_;
    using SampleConsensus::R_;
    using SampleConsensus::T_;
    using SampleConsensus::inliers_;
    using SampleConsensus::probability_;
    using SampleConsensus::SampleConsensusModelPtr;

    public:
      /** \brief RANSAC (RAndom SAmple Consensus) main constructor
        * \param model a Sample Consensus model
     */
    RandomSampleConsensus(const SampleConsensusModelPtr &model)
        :
          SampleConsensus(model)
    {
        // Maximum number of trials before we give up.
        max_iterations_ = 10000;
      }

    /** \brief Compute the actual model and find the inliers
     * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
     */
    bool
    computeModel()
    {
      iterations_ = 0;
      int n_best_inliers_count = -INT_MAX;
      double k = 1.0;

      std::vector<unsigned int> inliers;
      std::vector<unsigned int> selection;
      cv::Matx33f R;
      cv::Vec3f T;

      int n_inliers_count = 0;

      // Iterate
      while (iterations_ < k)
      {
        // Get X samples which satisfy the model criteria
        sac_model_->getSamples(iterations_, selection);

        if (selection.empty())
          break;

        // Search for inliers in the point cloud for the current plane model M
        if (!sac_model_->computeModelCoefficients(selection, R, T))
          continue;

        // Select the inliers that are within threshold_ from the model
        sac_model_->selectWithinDistance(R, T, threshold_, inliers);
        //if (inliers.empty () && k > 1.0)
        //  continue;

        n_inliers_count = inliers.size();

        // Better match ?
        if (n_inliers_count > n_best_inliers_count)
        {
          n_best_inliers_count = n_inliers_count;

          // Save the current model/inlier/coefficients selection as being the best so far
          inliers_ = inliers;
          R_ = R;
          T_ = T;

          // Compute the k parameter (k=log(z)/log(1-w^n))
          double w = (double) ((double) n_best_inliers_count / (double) sac_model_->getIndices().size());
          double p_no_outliers = 1.0 - pow(w, (double) selection.size());
          p_no_outliers = (std::max)(std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by -Inf
          p_no_outliers = (std::min)(1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by 0.
          k = log(1.0 - probability_) / log(p_no_outliers);
        }

        ++iterations_;
        if (iterations_ > max_iterations_)
          break;
      }

      if (inliers_.empty())
        return (false);

      // Get the set of inliers that correspond to the best model found so far
      //sac_model_->selectWithinDistance (model_coefficients_, threshold_, inliers_);
      return (true);
    }
  };
}

#endif  //#ifndef PCL_SAMPLE_CONSENSUS_RANSAC_H_
