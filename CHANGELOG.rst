^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package object_recognition_tod
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.5.5 (2015-04-19)
------------------
* Fix seg fault while detecting objects
  In many cases, matches[i] contains less than 5 elements (varies seemingly random between 0 and 4)
  Long history available [here](https://github.com/plasmodic/ecto_opencv/commit/2f2fe7fb75d09337c1d594cee416bd948f337b30#commitcomment-10687068)
* Contributors: Jorge Santos Simón

0.5.4 (2015-04-15)
------------------
* fix radius search not support with LSH
* remove useless code
* Contributors: Vincent Rabaud

0.5.3 (2015-03-31)
------------------
* use OpenCV and not opencv_candidate for LSH
* Contributors: Vincent Rabaud

0.5.2 (2015-03-30)
------------------
* add opencv_candidate as a dependency
  It is included but just transitively so it's better to have it
  directly
* Revert "remove dependency on opencv_candidate"
  This reverts commit 3a515e48c815b9e0c7f6df20b730b98d0dccd5df.
  The RGBD module from opencv_candidate is indeed needed
* Contributors: Vincent Rabaud

0.5.1 (2015-03-29)
------------------
* remove dependency on opencv_candidate
* clean extensions
* remove useless build dependency
* Contributors: Vincent Rabaud

0.5.0 (2014-04-13)
------------------
* compile under Indigo
* Merge pull request `#6 <https://github.com/wg-perception/tod/issues/6>`_ from awesomebytes/patch-2
  For some reason this is needed using openni
* For some reason this is needed using openni
  This is just a workaround, the real problem is that the default values expect:
  ```[ INFO] [1395248076.002277778]: Subscribed to topic:/camera/rgb/camera_info with queue size of 1
  [ INFO] [1395248076.003073369]: Subscribed to topic:/camera/rgb/image_raw with queue size of 1
  [ INFO] [1395248076.003968179]: Subscribed to topic:/camera/depth_registered/camera_info with queue size of 1
  [ INFO] [1395248076.004994459]: Subscribed to topic:/camera/depth_registered/image_raw with queue size of 1
  ```
  Using openni2 this is not a problem as it's published by default but using openni those are not published, so this workarounds that.
* Merge pull request `#5 <https://github.com/wg-perception/tod/issues/5>`_ from awesomebytes/patch-1
  Configuration that gives detections
* Configuration that gives detections
  Tested with an Asus Xtion and 6 objects trained.
  The sink2 + adding to pipeline1 the output sink2 makes that /recognized_object_array has publications!
* fix tests
* add openni runtime dependency
* fix the ROS config file to use ROS
* drop Fuerte support
* update url
* udpate the docs
* Merge pull request `#3 <https://github.com/wg-perception/tod/issues/3>`_ from destogl/ros_cell_config_correction
  Corrected cell cofing to work out-of-the-box with ROS.
* Corrected cell cofing to work out of the box with ros.
  There was problem with wrong type and parameters cell is added.
* fixes `wg-perception/object_recognition_core#13 <https://github.com/wg-perception/object_recognition_core/issues/13>`_
* Contributors: Denis Štogl, Sammy Pfeiffer, Vincent Rabaud

0.4.16 (2013-05-29)
-------------------
* use proper K_image/K_depth
* update email address
* Contributors: Vincent Rabaud

0.4.15 (2013-04-14)
-------------------
* fix display
* fix the 3d point non-sense
* comply to the new core API
* finally convert the Trainer fully to C++
* remove old split modules in favor of one cell
* comply to the new DB API
* implement moreof the pipeline
* comply to the new headers in core
* add a Trainer cell to replace the Python mess
* make the cell contents be in a common library-like file
* fix the pipeline
* fix the invalid plasm
* Contributors: Vincent Rabaud

0.4.14 (2013-03-12)
-------------------
* comply to the new DB API
* fix a Sphinx warning
* Contributors: Vincent Rabaud

0.4.13 (2013-02-25)
-------------------
* comply to the tighter BlackBox API
* comply to the API
* fix a member that can be static
* remove rst warning
* build its own docs
* get the training test to pass
* rename the config files
* clean the CMake
* Contributors: Vincent Rabaud

0.4.12 (2013-01-17)
-------------------
* comply to the new ORK API
* Contributors: Vincent Rabaud

0.4.11 (2013-01-13)
-------------------
* use the new DB API
* Contributors: Vincent Rabaud

0.4.10 (2013-01-04)
-------------------
* use the new BlackBox API
* comply to the new core API
* clean CMake
* fix the catkin buildtool_depend
* Contributors: Vincent Rabaud

0.4.9 (2012-11-18 17:47)
------------------------
* add the Eigen dependency for Fuerte
* Contributors: Vincent Rabaud

0.4.8 (2012-11-18 17:26)
------------------------
* make the setup.py work under Fuerte
* Contributors: Vincent Rabaud

0.4.7 (2012-11-03)
------------------
* Merge branch 'master' of github.com:wg-perception/tod
* use catkin_pkg
* fixed typos, package name changes, tendril connection issues and spore types
* Contributors: Tommaso Cavallari, Vincent Rabaud

0.4.6 (2012-11-01)
------------------
* remove the copyright
* use the new ecto_catkin interface
* get the information from the package.xml
* comply to the new API
* remove electric support
* add the missing Eigen dependency
* Contributors: Vincent Rabaud

0.4.5 (2012-10-10)
------------------
* fix some warnings
* comply to the new API
* comply to the new catkin API
* depends are just messy
* include EIgen properly
* comply to the new API
* fix the Groovy install
* Contributors: Vincent Rabaud

0.4.4 (2012-09-08)
------------------
* have code work with Electric/Fuerte/Groovy
* add depth in case feature_descriptor needs it
* use the FeatureDescriptor from ecto_opencv
* use the new ectomodule API
* remove G2O stuff as that should be done in capture
* changed doc index heading
* Contributors: David Gossow, Vincent Rabaud

0.4.3 (2012-08-23)
------------------
* fixed default tod configs + rst documentation
* no more include folder to share
* make the test be gtest
* try a different for the gtest on Oneiric
* Contributors: David Gossow, Vincent Rabaud

0.4.2 (2012-07-31)
------------------
* fix typo
* add a linker instruction for Oneiric
* use the new isValidDepth API
* Contributors: Vincent Rabaud

0.4.1 (2012-07-17)
------------------
* fix a bug in the sub-graph building to improve accuracy. Also add speedups
* create the 3d points in the pipeline (new API)
* small optimizations
* merge sac_model and sac_model_registration_graph for speed
* now that RANSAC is fat enough, use valgrind on the whole GuessGenerator
* Contributors: Vincent Rabaud

0.4.0 (2012-07-09)
------------------
* big optimization
* use faster norm function
* no need for the sample pool anymore as the indices_ are filtered before-hand in InvalidateIndices
* add a check for ths size of the indices
* add a check when no sample can be chosen
* remove more useless members
* merge files
* free from PCL and API breakages
* no need for templates anymore
* use unsigned int for indices
* make the clique test compile again
* remove the useless conversion to a PointCloud
* remove more useless members
* remove more useless member functions and switch the transform computation to OpenCV
* remove more useless members/headers
* start using R and T for the model
* remove a lot of useless members
* get rid of the sac_model_registration
* tweak parameters for ORB2 temporarily
* bring back some PCL 1.1 headers as 1.5 has too many internal breakages ....
* corrected an include guard
* Contributors: Mac Mason, Vincent Rabaud

0.3.1 (2012-06-07)
------------------
* fix some install issues
* Contributors: Vincent Rabaud

0.3.0 (2012-06-06)
------------------
* use a stack.xml
* output Rs and Ts for pose drawing
* reenable the scheduler options to not crash
* split the disparities out of the points
* Merge branch 'master' of github.com:wg-perception/tod
* comply to the new API
* remove PCL from the public API
* add a label for the kitchen doc
* Contributors: Vincent Rabaud

0.2.7 (2012-05-18)
------------------
* fix a glitch
* fix the new DB APi
* add Python linkage for Lucid
* Contributors: Vincent Rabaud

0.2.6 (2012-05-11 14:07)
------------------------
* remove pcl_io_ros
* Contributors: Vincent Rabaud

0.2.5 (2012-05-11 13:46)
------------------------
* fix pcl_ros_io maybe ...
* Contributors: Vincent Rabaud

0.2.4 (2012-05-10)
------------------
* clean pcl_ros_io dependency
* write some docs a bit
* no need to tune the scheduler here
* Contributors: Vincent Rabaud

0.2.3 (2012-05-01)
------------------
* make sure all the tests pass
* rename the stack to object_recognition_tod
* remove useless import
* work with the new stack name
* rename the stack and fix the dependencies
* start some docs
* remove useless load_pybinding
* use the new g2o
* cleaner CMake
* use catkin for python
* Merge branch 'master' of github.com:wg-perception/tod
* catkinize TOD
* make sure the tests pass
* use the new ecto_image_pipeline
* clean the dependencies
* rename ecto modules to be tod/ecto_*
* improve the include folder
* no need for the install script anymore
* use the electric compatible way of finding PCL
* simply the linkage
* have the code be compliant with electric and fuerte, yay ...
* use the db instead of the parameters
* minor cleanup
* comply to the new API
* let catkin handle the version
* simplify the PCL bug solution
* rename object_recognition to object_recognition_core
* Merge branch 'master' of github.com:wg-perception/tod
* fix bad linkage with PCL
* fix some bad numeric_limit understanding
* disable the max clique test
* comply to the new API
* use the new Python hierarchy
* link against the proper library
* proper way of requesting for ROS components
* make sure it works with catkin on fuerte
* TOD now only compiles on fuerte and PCL 1.4
* use the --help macro
* comply to the new API
* no more include in here
* LshMatcher is now in ecto_opencv
* move opencv_candidate to ecto_opencv
* add the feature_viewer from object_recognition
* make the tests much simpler
* use the enw PoseResult API
* add a .gitignore
* little cleanup
* fix bad imports
* fix a bad matrix copy
* fix the absence of apps folder
* move TOD from object_recognition
* first commit
* Contributors: Vincent Rabaud
