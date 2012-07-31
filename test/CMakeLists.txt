find_package(object_recognition_core)
if (object_recognition_core_FOUND)
    # pipeline tests
    object_recognition_core_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.tod)
    object_recognition_core_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.tod)

    object_recognition_core_help_test(${CMAKE_CURRENT_SOURCE_DIR}/../apps/feature_viewer)
endif()

# clique test
include_directories(SYSTEM ${GTEST_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/common)

add_executable(tod-clique-test main.cpp
                               test_maximum_clique.cpp
                               ../src/common/maximum_clique.cpp
)

find_package(PythonLibs REQUIRED)
target_link_libraries(tod-clique-test ${GTEST_LIBRARIES}
                                      ${PYTHON_LIBRARIES}
                                      ${OpenCV_LIBRARIES}
)
set_target_properties(tod-clique-test PROPERTIES COMPILE_FLAGS "-pthread")

add_test(tod_clique_tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tod-clique-test)
