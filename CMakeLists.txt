cmake_minimum_required(VERSION 2.8)
project(object_recognition_tod)

find_package(catkin REQUIRED cmake_modules ecto ecto_opencv object_recognition_core opencv_candidate)

catkin_package()

#install targets for all things python
catkin_python_setup()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

add_definitions("-Wno-pragmas -fno-strict-aliasing -Wall -Wl,--no-undefined")

find_package(Boost REQUIRED system filesystem serialization)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(SYSTEM ${BOOST_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)
if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# add docs
ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/source doc)

# install the applications
install(PROGRAMS ${PROJECT_SOURCE_DIR}/apps/feature_viewer
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY conf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
