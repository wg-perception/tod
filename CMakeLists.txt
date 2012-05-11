cmake_minimum_required(VERSION 2.8)
project(object_recognition_tod)
find_package(catkin REQUIRED)
catkin_stack()
catkin_project(object_recognition_tod
               INCLUDE_DIRS include
               DEPENDS Boost Eigen OpenCV object_recognition_core
)

#install targets for all things python
catkin_python_setup()

set(PYTHON_PACKAGES_PATH gen/py)

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
find_pcl_package()

rosbuild_lite_init()

set_ecto_install_package_name(${PROJECT_NAME})

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
