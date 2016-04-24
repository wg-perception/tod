/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Willow Garage, Inc.
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

#include <opencv2/imgproc/imgproc.hpp>
#if CV_VERSION_MAJOR == 3
#include <opencv2/rgbd.hpp>
namespace cv {
using namespace cv::rgbd;
}
#else
#include <opencv2/rgbd/rgbd.hpp>
#endif

#include "training.h"

inline unsigned int roundWithinBounds(float xy, int xy_min, int xy_max) {
  return std::min(std::max(cvRound(xy), xy_min), xy_max);
}

void validateKeyPoints(const std::vector<cv::KeyPoint> & in_keypoints, const cv::Mat &in_mask, const cv::Mat & depth,
                       const cv::Mat &in_K, const cv::Mat & descriptors, std::vector<cv::KeyPoint> & final_keypoints,
                       cv::Mat &final_points, cv::Mat & final_descriptors) {
  cv::Mat K;
  in_K.convertTo(K, CV_32FC1);
  size_t n_points = descriptors.rows;
  cv::Mat clean_descriptors = cv::Mat(descriptors.size(), descriptors.type());
  cv::Mat_<cv::Vec2f> clean_points(1, n_points);
  final_keypoints.clear();
  final_keypoints.reserve(n_points);

  cv::Mat_<uchar> mask;
  in_mask.convertTo(mask, CV_8U);
  // Erode just because of the possible rescaling
  cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 4);

  int width = mask.cols, height = mask.rows;
  size_t clean_row_index = 0;
  for (size_t keypoint_index = 0; keypoint_index < n_points; ++keypoint_index) {
    // First, make sure that the keypoint belongs to the mask
    const cv::KeyPoint & in_keypoint = in_keypoints[keypoint_index];
    unsigned int x = roundWithinBounds(in_keypoint.pt.x, 0, width), y = roundWithinBounds(in_keypoint.pt.y, 0, height);
    float z;
    bool is_good = false;
    if (mask(y, x))
      is_good = true;
    else {
      // If the keypoint does not belong to the mask, look in a slightly bigger neighborhood
      int window_size = 2;
      float min_dist_sq = std::numeric_limits<float>::max();
      // Look into neighborhoods of different sizes to see if we have a point in the mask
      for (unsigned int i = roundWithinBounds(x - window_size, 0, width);
          i <= roundWithinBounds(x + window_size, 0, width); ++i)
        for (unsigned int j = roundWithinBounds(y - window_size, 0, height);
            j <= roundWithinBounds(y + window_size, 0, height); ++j)
          if (mask(j, i)) {
            float dist_sq = (i - in_keypoint.pt.x) * (i - in_keypoint.pt.x)
                + (j - in_keypoint.pt.y) * (j - in_keypoint.pt.y);
            if (dist_sq < min_dist_sq) {
              // If the point is in the mask and the closest from the keypoint
              x = i;
              y = j;
              min_dist_sq = dist_sq;
              is_good = true;
            }
          }
    }
    if (!is_good)
      continue;

    // Now, check that the depth of the keypoint is valid
    switch (depth.depth()) {
      case CV_16U:
        z = depth.at<uint16_t>(y, x);
        if (!cv::isValidDepth(uint16_t(z)))
          is_good = false;
        z /= 1000;
        break;
      case CV_16S:
        z = depth.at<int16_t>(y, x);
        if (!cv::isValidDepth(int16_t(z)))
          is_good = false;
        z /= 1000;
        break;
      case CV_32F:
        z = depth.at<float>(y, x);
        if (!cv::isValidDepth(float(z)))
          is_good = false;
        break;
      default:
        continue;
        break;
    }

    if (!is_good)
      continue;

    // Store the keypoint and descriptor
    clean_points.at<cv::Vec2f>(0, clean_row_index) = cv::Vec2f(x, y);
    cv::Mat clean_descriptor_row = clean_descriptors.row(clean_row_index++);
    descriptors.row(keypoint_index).copyTo(clean_descriptor_row);
    final_keypoints.push_back(in_keypoint);
  }

  if (clean_row_index > 0) {
    clean_points.colRange(0, clean_row_index).copyTo(final_points);
    clean_descriptors.rowRange(0, clean_row_index).copyTo(final_descriptors);
  }
}

void mergePoints(const std::vector<cv::Mat> &in_descriptors, const std::vector<cv::Mat> &in_points,
                 cv::Mat &out_descriptors, cv::Mat &out_points) {
  // Figure out the number of points
  size_t n_points = 0, n_images = in_descriptors.size();
  for (size_t image_id = 0; image_id < n_images; ++image_id)
    n_points += in_descriptors[image_id].rows;
  if (n_points == 0)
    return;

  // Fill the descriptors and 3d points
  out_descriptors = cv::Mat(n_points, in_descriptors[0].cols, in_descriptors[0].depth());
  out_points = cv::Mat(1, n_points, CV_32FC3);
  size_t row_index = 0;
  for (size_t image_id = 0; image_id < n_images; ++image_id) {
    // Copy the descriptors
    const cv::Mat & descriptors = in_descriptors[image_id];
    int n_points = descriptors.rows;
    cv::Mat sub_descriptors = out_descriptors.rowRange(row_index, row_index + n_points);
    descriptors.copyTo(sub_descriptors);
    // Copy the 3d points
    const cv::Mat & points = in_points[image_id];
    cv::Mat sub_points = out_points.colRange(row_index, row_index + n_points);
    points.copyTo(sub_points);

    row_index += n_points;
  }
}

void cameraToWorld(const cv::Mat &R_in, const cv::Mat &T_in, const cv::Mat & in_points_ori,
                   cv::Mat &points_out) {
  cv::Mat_<float> R, T, in_points;
  R_in.convertTo(R, CV_32F);
  T_in.reshape(1, 1).convertTo(T, CV_32F);
  if (in_points_ori.empty() == false)
  {
    in_points_ori.reshape(1, in_points_ori.size().area()).convertTo(in_points, CV_32F);
    cv::Mat_<float> T_repeat;
    cv::repeat(T, in_points.rows, 1, T_repeat);

    // Apply the inverse translation/rotation
    cv::Mat points = (in_points - T_repeat) * R;
    // Reshape to the original size
    points_out = points.reshape(3, in_points_ori.rows);
  }
  else
  {
    points_out = cv::Mat();
  }
}
