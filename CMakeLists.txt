cmake_minimum_required(VERSION 2.8)
project(object_recognition_tod)
find_package(catkin REQUIRED)
if (${catkin_VERSION} VERSION_GREATER "0.5.28")
catkin_package()
else()
catkin_stack()
catkin_project(object_recognition_tod)
endif()

#install targets for all things python
catkin_python_setup()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

add_definitions("-Wno-pragmas -fno-strict-aliasing -Wall -Wl,--no-undefined")

find_package(Boost REQUIRED system filesystem serialization)
find_package(ecto REQUIRED)
find_package(ecto_opencv REQUIRED)
find_package(OpenCV REQUIRED)

rosbuild_lite_init()

include_directories(SYSTEM ${ecto_opencv_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIRS}
)

add_subdirectory(src)
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

if (ROS_ELECTRIC_FOUND)
# install the applications
install(PROGRAMS ${PROJECT_SOURCE_DIR}/apps/feature_viewer
        DESTINATION bin
)
endif()
if (ROS_FUERTE_FOUND)
# install the applications
install(PROGRAMS ${PROJECT_SOURCE_DIR}/apps/feature_viewer
        DESTINATION share/${PROJECT_NAME}/bin
)
endif()

if (ROS_GROOVY_OR_ABOVE_FOUND)
# install the applications
install(PROGRAMS ${PROJECT_SOURCE_DIR}/apps/feature_viewer
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
endif()
