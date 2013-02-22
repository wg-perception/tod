cmake_minimum_required(VERSION 2.8)
project(object_recognition_tod)

find_package(catkin REQUIRED ecto ecto_opencv object_recognition_core opencv_candidate)

if (${catkin_VERSION} VERSION_GREATER "0.5.28")
catkin_package()
else()

find_package(ROS REQUIRED ecto opencv_candidate object_recognition_core)
catkin_stack()
catkin_project(object_recognition_tod)

set(catkin_LIBRARIES ${ROS_LIBRARIES})
set(catkin_INCLUDE_DIRS ${ROS_INCLUDE_DIRS})
set(CATKIN_DEVEL_PREFIX ${CMAKE_BINARY_DIR})
set(CATKIN_PACKAGE_BIN_DESTINATION share/${PROJECT_NAME}/bin)
set(CATKIN_PACKAGE_INCLUDE_DESTINATION include/${PROJECT_NAME})
set(CATKIN_PACKAGE_LIB_DESTINATION lib)
set(CATKIN_PACKAGE_PYTHON_BUILD_DESTINATION gen/py/${PROJECT_NAME}/)
set(CATKIN_PACKAGE_PYTHON_DESTINATION ${INSTALLED_PYTHONPATH}/${PROJECT_NAME}/)
set(CATKIN_PACKAGE_SHARE_DESTINATION share/${PROJECT_NAME}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
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
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(SYSTEM ${BOOST_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
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
