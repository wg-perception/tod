# deal with the training TOD modules

find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM
    ${object_recognition_core_INCLUDE_DIRS}
)

ectomodule(tod_training
    CameraToWorld.cpp
    KeypointValidator.cpp
    ModelStacker.cpp
    ModelFiller.cpp
    module.cpp
    PointMerger.cpp
    PrepareForG2O.cpp
)

set_target_properties(tod_detection_ectomodule
      PROPERTIES
      COMPILE_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
      LINK_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
      )

link_ecto(tod_training
  ${object_recognition_core_LIBRARIES}
  ${OpenCV_LIBS}
  opencv_candidate
  ${pcl_LIBRARIES}
  tod_common
)

install_ecto_module(tod_training)
