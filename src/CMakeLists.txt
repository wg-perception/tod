find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# deal with the detection TOD module
ectomodule(ecto_detection detection/DescriptorMatcher.cpp
                          detection/GuessGenerator.cpp
                          detection/ModelReader.cpp
                          detection/module.cpp
                          common/adjacency_ransac.cpp
                          common/maximum_clique.cpp
)

link_ecto(ecto_detection ${OpenCV_LIBRARIES}
                         ${object_recognition_core_LIBRARIES} 
                         ${Boost_LIBRARIES}
                         ${ecto_opencv_LIBRARIES}
)

install_ecto_module(ecto_detection)


# deal with the training TOD modules
ectomodule(ecto_training training/CameraToWorld.cpp
                         training/KeypointValidator.cpp
                         training/ModelStacker.cpp
                         training/ModelFiller.cpp
                         training/module.cpp
                         training/PointMerger.cpp
                         training/PrepareForG2O.cpp
                         common/adjacency_ransac.cpp
                         common/maximum_clique.cpp
)

link_ecto(ecto_training ${object_recognition_core_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${ecto_opencv_LIBRARIES}
)

install_ecto_module(ecto_training)
