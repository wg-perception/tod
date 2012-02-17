# TOD tests
object_recognition_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.tod)
object_recognition_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.tod)

object_recognition_help_test(${CMAKE_CURRENT_SOURCE_DIR}/../apps/feature_viewer)

# clique test
find_package(GTest)

if(NOT GTEST_FOUND)
  message(STATUS "Disabling testing since gtest was not found.")
  return()
endif()

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${tod_SOURCE_DIR}/src/common)

add_executable(tod-clique-test
  main.cpp
  test_maximum_clique.cpp
  )

target_link_libraries(tod-clique-test
    ${GTEST_LIBRARIES}
    tod_common
    ${pcl_ros_io_LIBRARIES}
)

add_test(tod_clique_tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tod-clique-test)

