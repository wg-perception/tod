
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

#include <limits.h>
#include <vector>

#include <boost/foreach.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/rgbd/rgbd.hpp>

#include "training.h"

using ecto::tendrils;

namespace
{
  inline unsigned int
  roundWithinBounds(float xy, int xy_min, int xy_max)
  {
    return std::min(std::max(cvRound(xy), xy_min), xy_max);
  }
}
namespace
{
  /** Ecto module that makes sure keypoints are part of a mask
   */
  struct KeypointsValidator
  {
    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare<std::vector<cv::KeyPoint> >("keypoints", "The keypoints").required(true);
      inputs.declare<cv::Mat>("descriptors", "The descriptors").required(true);
      inputs.declare<cv::Mat>("mask", "The mask keypoint have to belong to").required(true);
      inputs.declare<cv::Mat>("K", "The calibration matrix").required(true);
      inputs.declare<cv::Mat>("depth", "The depth image (with a size similar to the mask one).").required(true);

      outputs.declare(&KeypointsValidator::points_out_, "points", "The valid keypoints: 1 x n_points x 2 channels (x in pixels, y in pixels)");
      outputs.declare(&KeypointsValidator::descriptors_out_, "descriptors", "The matching descriptors, n_points x feature_length");
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      const std::vector<cv::KeyPoint> & in_keypoints = inputs.get<std::vector<cv::KeyPoint> >("keypoints");
      cv::Mat in_mask, depth, descriptors, in_K;
      inputs["mask"] >> in_mask;
      inputs["depth"] >> depth;
      inputs["K"] >> in_K;
      inputs["descriptors"] >> descriptors;

      validateKeyPoints(in_keypoints, in_mask, depth, in_K, descriptors, *points_out_, *descriptors_out_);

      return ecto::OK;
    }
  private:
    /** The valid 2d points */
    ecto::spore<cv::Mat> points_out_;
    /** The valid descriptors */
    ecto::spore<cv::Mat> descriptors_out_;
  };
}

ECTO_CELL(ecto_training, KeypointsValidator, "KeypointsValidator",
          "Given keypoints and a mask, make sure they belong to the mask by rounding their coordinates.")
