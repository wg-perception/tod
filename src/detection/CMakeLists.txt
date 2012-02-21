# deal with the detection TOD module

find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM
    ${object_recognition_core_INCLUDE_DIRS}
)

ectomodule(tod_detection
  DescriptorMatcher.cpp
  GuessGenerator.cpp
  ModelReader.cpp
  module.cpp
)

link_ecto(tod_detection
  ${OpenCV_LIBS}
  ${ecto_opencv_LIBRARIES}
  ${object_recognition_core_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  opencv_candidate
  tod_common
)

install_ecto_module(tod_detection)
