find_package(object_recognition_core)
if (object_recognition_core_FOUND)
    # pipeline tests
    object_recognition_core_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.tod)
    object_recognition_core_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.tod)

    object_recognition_core_help_test(${CMAKE_CURRENT_SOURCE_DIR}/../apps/feature_viewer)
endif()

if (NOT GTEST_FOUND)
  return()
endif()

# clique test
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/common)

if (ROS_GROOVY_FOUND)
catkin_add_gtest(test_tod_clique main.cpp test_maximum_clique.cpp
                                   ../src/common/maximum_clique.cpp
)
else()
add_gtest(test_tod_clique main.cpp test_maximum_clique.cpp
                                   ../src/common/maximum_clique.cpp
)
endif()
find_package(PythonLibs REQUIRED)
target_link_libraries(test_tod_clique ${PYTHON_LIBRARIES}
                                      ${OpenCV_LIBRARIES}
)
