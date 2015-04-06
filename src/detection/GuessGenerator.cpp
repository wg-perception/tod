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

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <object_recognition_core/common/types.h>
#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/db.h>
#include "adjacency_ransac.h"

//#define DO_VALGRIND
#ifdef DO_VALGRIND
#include <valgrind/callgrind.h>
#endif

using ecto::tendrils;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;

namespace tod
{
  /** Ecto implementation of a module that takes
   *
   */
  struct GuessGenerator
  {
    static void
    declare_params(ecto::tendrils& params)
    {
      params.declare(&GuessGenerator::min_inliers_, "min_inliers", "Minimum number of inliers", 15);
      params.declare(&GuessGenerator::n_ransac_iterations_, "n_ransac_iterations", "Number of RANSAC iterations.",
                     1000);
      params.declare(&GuessGenerator::sensor_error_, "sensor_error", "The error (in meters) from the Kinect", 0.01);
      params.declare(&GuessGenerator::visualize_, "visualize", "If true, display temporary info through highgui",
                     false);
      params.declare(&GuessGenerator::json_db_, "db", "The DB to get data from, as a JSON string").required(true);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare<cv::Mat>("image", "The height by width 3 channel point cloud");
      inputs.declare<cv::Mat>("points3d", "The height by width 3 channel point cloud");
      inputs.declare<std::vector<cv::KeyPoint> >("keypoints", "The interesting keypoints");
      inputs.declare<std::vector<std::vector<cv::DMatch> > >("matches", "The list of OpenCV DMatch");
      inputs.declare<std::vector<cv::Mat> >(
          "matches_3d",
          "The corresponding 3d position of those matches. For each point, a 1 by n 3 channel matrix (for x,y and z)");
      inputs.declare<std::map<ObjectId, float> >("spans", "For each found object, its span based on known features.");
      inputs.declare<std::vector<ObjectId> >("object_ids", "The ids used in the matches");

      outputs.declare(&GuessGenerator::pose_results_, "pose_results", "The results of object recognition");
      outputs.declare(&GuessGenerator::Rs_, "Rs", "The rotations of the poses (useful for visualization)");
      outputs.declare(&GuessGenerator::Ts_, "Ts", "The translations of the poses (useful for visualization)");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      if (*visualize_)
      {
        colors_.push_back(cv::Scalar(255, 255, 0));
        colors_.push_back(cv::Scalar(0, 255, 255));
        colors_.push_back(cv::Scalar(255, 0, 255));
        colors_.push_back(cv::Scalar(255, 0, 0));
        colors_.push_back(cv::Scalar(0, 255, 0));
        colors_.push_back(cv::Scalar(0, 0, 255));
        colors_.push_back(cv::Scalar(0, 0, 0));
        colors_.push_back(cv::Scalar(85, 85, 85));
        colors_.push_back(cv::Scalar(170, 170, 170));
        colors_.push_back(cv::Scalar(255, 255, 255));
      }

      // Set the DB
      db_ = object_recognition_core::db::ObjectDbParameters(*json_db_).generateDb();
    }

    /** Get the 2d keypoints and figure out their 3D position from the depth map
     * @param inputs
     * @param outputs
     * @return
     */
    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      // Get the different matches
      const std::vector<std::vector<cv::DMatch> > & matches = inputs.get<std::vector<std::vector<cv::DMatch> > >(
          "matches");
      const std::vector<cv::Mat> & matches_3d = inputs.get<std::vector<cv::Mat> >("matches_3d");

      // Get the original keypoints and point cloud
      const std::vector<cv::KeyPoint> & keypoints = inputs.get<std::vector<cv::KeyPoint> >("keypoints");
      const cv::Mat point_cloud = inputs.get<cv::Mat>("points3d");
      const std::vector<ObjectId> & object_ids_in = inputs.get<std::vector<ObjectId> >("object_ids");
      const std::map<ObjectId, float> & spans = inputs.get<std::map<ObjectId, float> >("spans");

      const cv::Mat & initial_image = inputs.get<cv::Mat>("image");

      // Get the outputs
      pose_results_->clear();
      Rs_->clear();
      Ts_->clear();
      if (point_cloud.empty())
      {
        // Only use 2d to 3d matching
        // TODO
        //const std::vector<cv::KeyPoint> &keypoints = inputs.get<std::vector<cv::KeyPoint> >("keypoints");
      }
      else
      {
#ifdef DO_VALGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
        // Cluster the matches per object ID
        OpenCVIdToObjectPoints all_object_points;
        ClusterPerObject(keypoints, point_cloud, matches, matches_3d, all_object_points);
        cv::Mat visualize_img;
        size_t color_index = 0;
        if (*visualize_)
        {
          DrawClustersPerObject(keypoints, colors_, initial_image, all_object_points);
          initial_image.copyTo(visualize_img);
        }

        // For each object, build the connectivity graph between the matches
        while (!all_object_points.empty())
        {
          // Create a graph for that object
          AdjacencyRansac & adjacency_ransac = all_object_points.begin()->second;
          size_t opencv_object_id = all_object_points.begin()->first;
          ObjectId object_id = object_ids_in[opencv_object_id];

          std::cout << "***Starting object: " << opencv_object_id << std::endl;

          {
            std::vector<unsigned int> query_indices = adjacency_ransac.query_indices();
            std::sort(query_indices.begin(), query_indices.end());
            std::vector<unsigned int>::iterator end = std::unique(query_indices.begin(), query_indices.end());
            query_indices.resize(end - query_indices.begin());

            std::cout << query_indices.size() << " keypoints in " << adjacency_ransac.query_indices().size()
                      << " matches" << std::endl;
          }

          adjacency_ransac.FillAdjacency(keypoints, spans.find(object_id)->second, *sensor_error_);
          std::cout << "done filling" << std::endl;
          // Keep processing the graph until there is no maximum clique of the right size
          while (true)
          {
            // Compute the maximum of clique of that graph
            std::vector<unsigned int> query_inliers;
            cv::Matx33f R_mat;
            cv::Vec3f tvec;
            adjacency_ransac.Ransac(*sensor_error_, *n_ransac_iterations_, query_inliers, R_mat, tvec);


            // Figure out the matches to remove
            std::cout << "RANSAC done with " << query_inliers.size() << " inliers" << std::endl;

            // If no pose was found, forget about all the connections in that clique
            if (query_inliers.size() < *min_inliers_)
              break;

            adjacency_ransac.InvalidateQueryIndices(query_inliers);

            // Store the matches for debug purpose
            if (*visualize_)
            {
              std::vector<cv::KeyPoint> draw_keypoints;
              BOOST_FOREACH(unsigned int index, query_inliers)draw_keypoints.push_back(
                  keypoints[index]);
              if (color_index < colors_.size())
              {
                cv::drawKeypoints(visualize_img, draw_keypoints, visualize_img, colors_[color_index]);
                ++color_index;
              }
            }

            // Save the result;
            PoseResult pose_result;
            pose_result.set_R(cv::Mat(R_mat));
            pose_result.set_T(cv::Mat(tvec));
            pose_result.set_object_id(db_, object_id);
            pose_results_->push_back(pose_result);
            Rs_->push_back(cv::Mat(R_mat));
            Ts_->push_back(cv::Mat(tvec));
          }

          // Remove that object so that some data gets freed
          all_object_points.erase(opencv_object_id);
        }

        if (*visualize_)
        {
          cv::namedWindow("inliers", 0);
          cv::imshow("inliers", visualize_img);
        }

        std::cout << "********************* found " << pose_results_->size() << " poses" << std::endl;
      }
#ifdef DO_VALGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
#endif

      return ecto::OK;
    }
  private:
    /** List of very different colors, for debugging purposes */
    std::vector<cv::Scalar> colors_;
    /** Rotations of the poses */
    ecto::spore<std::vector<cv::Mat> > Rs_;
    /** Translations of the poses */
    ecto::spore<std::vector<cv::Mat> > Ts_;
    /** flag indicating whether we run in debug mode */
    ecto::spore<bool> visualize_;
    /** The minimum number of inliers in order to do pose matching */
    ecto::spore<unsigned int> min_inliers_;
    /** The number of RANSAC iterations to perform */
    ecto::spore<unsigned int> n_ransac_iterations_;
    /** How much can the sensor be wrong at most */
    ecto::spore<float> sensor_error_;
    /** The object recognition results */
    ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
    /** The DB */
    ecto::spore<std::string> json_db_;
    object_recognition_core::db::ObjectDbPtr db_;
  }
  ;
}

ECTO_CELL(ecto_detection, tod::GuessGenerator, "GuessGenerator",
          "Given descriptors and 3D positions, compute object guesses.");
