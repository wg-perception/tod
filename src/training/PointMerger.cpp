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

#include <vector>

#include <boost/foreach.hpp>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>

#include <object_recognition_core/common/types_eigen.h>

namespace tod
{
  /** cell storing the 3d points and descriptors while a model is being computed
   */
  struct PointMerger
  {
  public:
    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare < std::vector<cv::Mat> > ("descriptors", "The descriptors per image.").required(true);
      inputs.declare < std::vector<cv::Mat> > ("points", "The 3d points.").required(true);

      outputs.declare < cv::Mat > ("descriptors", "The stacked descriptors.");
      outputs.declare < cv::Mat > ("points", "The 3d position of the points.");
    }

    void
    configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      in_points_ = inputs["points"];
      descriptors_ = inputs["descriptors"];

      out_points_ = outputs["points"];
      out_descriptors_ = outputs["descriptors"];
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      // Figure out the number of points
      unsigned int n_points = 0, n_images = descriptors_->size();
      for (size_t image_id = 0; image_id < n_images; ++image_id)
        n_points += (*descriptors_)[image_id].rows;

      // Fill the descriptors and 3d points
      *out_descriptors_ = cv::Mat(n_points, (*descriptors_)[0].cols, (*descriptors_)[0].depth());
      *out_points_ = cv::Mat(1, n_points, CV_32FC3);
      size_t row_index = 0;
      for (size_t image_id = 0; image_id < n_images; ++image_id)
      {
        // Copy the descriptors
        const cv::Mat & descriptors = (*descriptors_)[image_id];
        cv::Mat sub_descriptors = out_descriptors_->rowRange(row_index, row_index + descriptors.rows);
        descriptors.copyTo(sub_descriptors);
        // Copy the 3d points
BOOST_FOREACH      ( const cv::Mat & vec, (*in_points_))
      {
        out_points_->at<cv::Vec3f>(0, row_index) = cv::Vec3f(vec.at<float>(0), vec.at<float>(1), vec.at<float>(2));
        ++row_index;
      }
    }

      return ecto::OK;
    }
  private:
    ecto::spore<std::vector<cv::Mat> > in_points_;
    ecto::spore<std::vector<cv::Mat> > descriptors_;
    ecto::spore<cv::Mat> out_points_;
    ecto::spore<cv::Mat> out_descriptors_;
  };
}

ECTO_CELL(ecto_training, tod::PointMerger, "PointMerger", "Merge the points and descriptor")
