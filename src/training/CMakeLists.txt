# deal with the training TOD modules

find_package(object_recognition_core REQUIRED)

rosbuild_lite_init()

find_pcl_package()

include_directories(SYSTEM
                    ${object_recognition_core_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
)

ectomodule(ecto_training
    CameraToWorld.cpp
    KeypointValidator.cpp
    ModelStacker.cpp
    ModelFiller.cpp
    module.cpp
    PointMerger.cpp
    PrepareForG2O.cpp
)

link_ecto(ecto_training
  ${object_recognition_core_LIBRARIES}
  ${OpenCV_LIBS}
  ${ecto_opencv_LIBRARIES}
  ${PCL_LIBRARIES}
  tod_common
)

install_ecto_module(ecto_training)
