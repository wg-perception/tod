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

link_ecto(tod_training
  ${object_recognition_core_LIBRARIES}
  ${OpenCV_LIBS}
  opencv_candidate
  ${pcl_LIBRARIES}
  tod_common
)

install_ecto_module(tod_training)
