# deal with the detection TOD module
ectomodule(tod_detection
  DescriptorMatcher.cpp
  GuessGenerator.cpp
  ModelReader.cpp
  module.cpp
)

set_target_properties(tod_detection_ectomodule
      PROPERTIES
      COMPILE_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
      LINK_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
      )

link_ecto(tod_detection
  ${OpenCV_LIBS}
  ${ecto_opencv_LIBRARIES}
  object_recognition_common
  object_recognition_db
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  opencv_candidate
  tod_common
)

install_ecto_module(tod_detection)
