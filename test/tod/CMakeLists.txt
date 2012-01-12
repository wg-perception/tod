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
  tod_detection_ectomodule
  )

add_test(tod_clique_tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tod-clique-test)
