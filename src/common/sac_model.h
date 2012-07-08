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
 * $Id: sac_model.h 1896 2011-07-26 19:04:32Z rusu $
 *
 */

#ifndef PCL_SAMPLE_CONSENSUS_MODEL_H_
#define PCL_SAMPLE_CONSENSUS_MODEL_H_

#include <cfloat>
#include <limits.h>
#include <set>

namespace pcl
{
  /** \brief @b SampleConsensusModel represents the base model class. All sample consensus models must inherit from 
    * this class.
    * \author Radu Bogdan Rusu
    * \ingroup sample_consensus
    */
  class SampleConsensusModel
  {
  public:
    typedef boost::shared_ptr<SampleConsensusModel> Ptr;
    typedef boost::shared_ptr<const SampleConsensusModel> ConstPtr;
    typedef std::vector<unsigned int> IndexVector;

      /** \brief Destructor for base SampleConsensusModel. */
      virtual ~SampleConsensusModel () {};

      /** \brief Get a set of random data samples and return them as point
        * indices. Pure virtual.  
        * \param iterations the internal number of iterations used by SAC methods
        * \param samples the resultant model samples
        */
      void 
      getSamples (int &iterations, std::vector<unsigned int> &samples)
    {
      if (indices_.size() < 3)
      {
        samples.clear();
        iterations = INT_MAX - 1;

        return;
      }

        // Get a second point which is different than the first
        samples.resize (3);
        for (unsigned int iter = 0; iter < max_sample_checks_; ++iter)
        {
          // Choose the random indices
          SampleConsensusModel::drawIndexSample (samples);

          // If it's a good sample, stop here
          if (isSampleGood (samples))
            return;
        }
        samples.clear ();
      }

      /** \brief Check whether the given index samples can form a valid model,
        * compute the model coefficients from these samples and store them
        * in model_coefficients. Pure virtual.
        * \param samples the point indices found as possible good candidates
        * for creating a valid model 
        * \param model_coefficients the computed model coefficients
        */
      virtual bool
      computeModelCoefficients (const IndexVector &samples, cv::Matx33f &R, cv::Vec3f&T) = 0;

      virtual
      void
      optimizeModelCoefficients(const IndexVector &inliers, cv::Matx33f&R, cv::Vec3f&T) =0;

      /** \brief Select all the points which respect the given model
        * coefficients as inliers. Pure virtual.
        *
        * \param model_coefficients the coefficients of a model that we need to
        * compute distances to
        * \param threshold a maximum admissible distance threshold for
        * determining the inliers from the outliers
        * \param inliers the resultant model inliers
        */
    virtual void
    selectWithinDistance(const cv::Matx33f &R, const cv::Vec3f&T, double threshold, IndexVector &inliers) = 0;

      /** \brief Provide the vector of indices that represents the input data.
        * \param indices the vector of indices that represents the input data.
        */
      inline void 
      setIndices (IndexVector &indices)
      { 
        indices_ = indices;
        shuffled_indices_ = indices;
      }

      /** \brief Get a pointer to the vector of indices used. */
      inline const IndexVector &
      getIndices () const { return (indices_); }

    protected:
      /** \brief Fills a sample array with random samples from the indices_ vector
        * Sure, there are some swaps in there but it is linear in the size of the sample, no stupid while loop to
        * compare the elements between them
        * \param sample the set of indices of target_ to analyze
        */
      inline void
      drawIndexSample (IndexVector & sample)
      {
        size_t sample_size = sample.size ();
        size_t index_size = shuffled_indices_.size ();
        for (unsigned int i = 0; i < sample_size; ++i)
          // The 1/(RAND_MAX+1.0) trick is when the random numbers are not uniformly distributed and for small modulo
          // elements, that does not matter (and nowadays, random number generators are good)
          std::swap (shuffled_indices_[i], shuffled_indices_[i + (rand () % (index_size - i))]);
        std::copy (shuffled_indices_.begin (), shuffled_indices_.begin () + sample_size, sample.begin ());
      }

      /** \brief Check if a sample of indices results in a good sample of points
        * indices. Pure virtual.
        * \param samples the resultant index samples
        */
      virtual bool
      isSampleGood (const IndexVector &samples) const = 0;

      /** \brief A pointer to the vector of point indices to use. */
      IndexVector indices_;

      /** The maximum number of samples to try until we get a good one */
      static const unsigned int max_sample_checks_ = 1000;

      /** Data containing a shuffled version of the indices. This is used and modified when drawing samples. */
      IndexVector shuffled_indices_;
  };
}

#endif  //#ifndef PCL_SAMPLE_CONSENSUS_MODEL_H_
