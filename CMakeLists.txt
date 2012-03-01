cmake_minimum_required(VERSION 2.8)
project(tod)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

add_definitions("-Wno-pragmas -fno-strict-aliasing -Wall -Wl,--no-undefined")

#set the default path for built libraries to the "ecto_opencv" directory
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ecto_tod)

find_package(Boost REQUIRED system filesystem serialization)
find_package(ecto REQUIRED)
find_package(ecto_opencv REQUIRED)
find_package(OpenCV REQUIRED)
find_pcl_package()

rosbuild_lite_init()

if (ROS_ELECTRIC_FOUND)
    find_library(pcl_ros_io_LIBRARIES pcl_ros_io /opt/ros/electric/stacks/perception_pcl/pcl_ros/lib)
else()
    find_library(pcl_ros_io_LIBRARIES pcl_ros_io /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib)
endif()

set_ecto_install_package_name(ecto_tod)

configure_file(python/__init__.py.in
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
  @ONLY)

include_directories(
    ${object_recognition_INCLUDE_DIRS}
)
include_directories(SYSTEM
    ${PCL_INCLUDE_DIRS}
    ${ecto_opencv_INCLUDE_DIRS}
)

add_subdirectory(src)
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

#these setup the lib to be used by others
include(cmake/config.cmake)

#install targets for all things python
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
  DESTINATION ${PYTHON_PACKAGES_PATH}
  COMPONENT main
)

install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
        DESTINATION ${PYTHON_PACKAGES_PATH}/ecto_tod
        )

catkin_add_env_hooks(10tod SHELLS all )
