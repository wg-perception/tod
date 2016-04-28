/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
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

#include "training.h"

#include <string>

#include <ecto/ecto.hpp>

#include <boost/foreach.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if CV_VERSION_MAJOR == 3
#include <opencv2/rgbd.hpp>
using cv::rgbd::depthTo3dSparse;
using cv::rgbd::rescaleDepth;
#else
#include <opencv2/rgbd/rgbd.hpp>
using cv::depthTo3dSparse;
using cv::rescaleDepth;
#endif

#include <object_recognition_core/common/json_spirit/json_spirit.h>
#include <object_recognition_core/common/types_eigen.h>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/prototypes/observations.hpp>
#include <object_recognition_core/db/view.h>

void
rescale_depth(const cv::Mat depth_in, const cv::Size & isize, cv::Mat &depth_out)
{
  cv::Size dsize = depth_in.size();
  cv::Mat depth;
  rescaleDepth(depth_in, CV_32F, depth);

  if (dsize == isize) {
    depth_out = depth;
    return;
  }

  float factor = static_cast<float>(isize.width) / dsize.width; //scaling factor.
  cv::Mat output(isize, depth.type(), std::numeric_limits<float>::quiet_NaN()); //output is same size as image.

  //resize into the subregion of the correct aspect ratio
  cv::Mat subregion(output.rowRange(0, dsize.height * factor));

  //use nearest neighbor to prevent discontinuities causing bogus depth.
  cv::resize(depth, subregion, subregion.size(), cv::INTER_NEAREST);
  depth_out = output;
}

/** cell storing the 3d points and descriptors while a model is being computed
 */
struct Trainer
{
public:
  static void
  declare_params(ecto::tendrils& params)
  {
    params.declare(&Trainer::json_feature_params_, "json_feature_params",
        std::string("Parameters for the feature as a JSON string. ")
            + std::string(
                "It should have the format: \"{\"type\":\"ORB/SIFT whatever\", ")
            + std::string(
                "\"module\":\"where_it_is\", \"param_1\":val1, ....}"),
        "{\"type\": \"ORB\", \"module\": \"ecto_opencv.features2d\"}").required(
        true);
    params.declare(&Trainer::json_descriptor_params_, "json_descriptor_params",
        std::string("Parameters for the descriptor as a JSON string. ")
            + std::string(
                "It should have the format: \"{\"type\":\"ORB/SIFT whatever\", ")
            + std::string(
                "\"module\":\"where_it_is\", \"param_1\":val1, ....}" ""),
        "{\"type\": \"ORB\", \"module\": \"ecto_opencv.features2d\"}").required(
        true);

    params.declare(&Trainer::visualize_, "visualize",
        "If true, debug data is visualized.", false);
  }

  static void
  declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
  {
    // Inputs
    inputs.declare(&Trainer::json_db_, "json_db",
        "The parameters of the DB as a JSON string.").required(true);
    inputs.declare(&Trainer::object_id_, "object_id",
        "The id of the object in the DB.").required(true);

    // Outputs
    outputs.declare(&Trainer::descriptors_out_, "descriptors",
        "The stacked descriptors.");
    outputs.declare(&Trainer::points_out_, "points",
        "The 3d position of the points.");
  }

  void
  configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
  {
    or_json::mObject feature_param_tree;
    std::stringstream ssparams;
    ssparams << params.get < std::string > ("json_feature_params");

    {
      or_json::mValue value;
      or_json::read(ssparams, value);
      feature_param_tree = value.get_obj();
    }

    // Create the pointer depending on the type of descriptors
    std::string feature_type = feature_param_tree["type"].get_str();

#if CV_VERSION_MAJOR == 3
    if (feature_type == "ORB")
    {
      int n_features = feature_param_tree["n_features"].get_uint64();
      int n_levels = feature_param_tree["n_levels"].get_uint64();
      float scale_factor = feature_param_tree["scale_factor"].get_real();

      feature2d_ = cv::ORB::create(n_features, scale_factor, n_levels);
    }
    else if(feature_type == "AKAZE")
    {
      feature2d_ = cv::AKAZE::create();
    }
#else
    if (feature_type == "ORB")
    {
      feature2d_ = new cv::ORB(n_features, scale_factor, n_levels);
    }
#endif
    else
    {
      std::cerr << "Features not implemented for that type: " << feature_type << std::endl;
      throw;
    }

  }

  int
  process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
  {
    // Get the DB
    object_recognition_core::db::ObjectDbPtr db =
        object_recognition_core::db::ObjectDbParameters(*json_db_).generateDb();

    // Get observations from the DB
    object_recognition_core::db::View view(
        object_recognition_core::db::View::VIEW_OBSERVATION_WHERE_OBJECT_ID);
    view.set_key(*object_id_);

    object_recognition_core::db::ViewIterator view_iterator(view, db);

    std::vector<cv::Mat> descriptors_all, points_all;
    object_recognition_core::db::ViewIterator iter = view_iterator.begin(),
        end = view_iterator.end();

    int counter = 1;

    for(; iter != end; ++iter) {

      // Convert the observation to a usable type
      object_recognition_core::prototypes::Observation obs;
      object_recognition_core::db::Document view_element = (*iter);
      obs << &view_element;

      // Compute the features/descriptors on the image
      cv::Mat /*points,*/ descriptors;
      std::vector<cv::KeyPoint> keypoints;

      // TODO actually use the params and do not force ORB
#if CV_VERSION_MAJOR == 3
      cv::Ptr<cv::DescriptorExtractor> orb = cv::ORB::create();
      orb->detectAndCompute(obs.image, obs.mask, keypoints, descriptors);
#else
      cv::ORB orb;
      orb(obs.image, obs.mask, keypoints, descriptors);
#endif

      std::cout << "Frame " << counter << " --> found " << descriptors.rows << " descriptors" << std::endl;

      // Rescale the depth
      cv::Mat depth;
      rescale_depth(obs.depth, obs.image.size(), depth);

      // Validate the keypoints
      cv::Mat points_clean, descriptors_final;
      std::vector<cv::KeyPoint> keypoints_final;
      validateKeyPoints(keypoints, obs.mask, depth, obs.K, descriptors,
          keypoints_final, points_clean, descriptors_final);

      if (points_clean.empty())
        continue;
      descriptors_all.push_back(descriptors_final);

      // Convert the points to world coordinates
      cv::Mat points_clean_3d, points_final;
      cv::depthTo3dSparse(depth, obs.K, points_clean, points_clean_3d);
      cameraToWorld(obs.R, obs.T, points_clean_3d, points_final);
      points_all.push_back(points_final);

      // visualize data if asked for
      if (*visualize_)
      {
        // draw keypoints on the masked object
        cv::namedWindow("keypoints");
        cv::Mat img;
        cv::drawKeypoints(obs.image, keypoints, img, cv::Scalar(255,0,0));
        cv::imshow("keypoints", img);
        cv::waitKey(3);
      }

      counter++;
    }

    // merge the points into unique cv::Mat
    mergePoints(descriptors_all, points_all, *descriptors_out_, *points_out_);

    std::cout << "Found total " << (*descriptors_out_).rows << " descriptors" << std::endl;

    return ecto::OK;
  }
private:
  ecto::spore<std::string> json_feature_params_;
  ecto::spore<std::string> json_descriptor_params_;
  ecto::spore<std::string> object_id_;
  ecto::spore<std::string> json_db_;
  ecto::spore<std::vector<cv::Mat> > descriptors_in_;
  ecto::spore<std::vector<cv::Mat> > points_in_;
  ecto::spore<cv::Mat> points_out_;
  ecto::spore<cv::Mat> descriptors_out_;
  ecto::spore<bool> visualize_;
  cv::Ptr<cv::Feature2D> feature2d_;
};

ECTO_CELL(ecto_training, Trainer, "Trainer",
    "Compute TOD models for a given object")
