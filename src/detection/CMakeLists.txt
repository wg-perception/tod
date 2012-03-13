# deal with the detection TOD module

find_package(object_recognition_core REQUIRED)

rosbuild_lite_init()

find_pcl_package()

include_directories(SYSTEM
                    ${object_recognition_core_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
)

ectomodule(ecto_detection
  DescriptorMatcher.cpp
  GuessGenerator.cpp
  ModelReader.cpp
  module.cpp
)

link_ecto(ecto_detection
  ${OpenCV_LIBS}
  ${ecto_opencv_LIBRARIES}
  ${object_recognition_core_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${ecto_opencv_LIBRARIES}
  tod_common
)

install_ecto_module(ecto_detection)
