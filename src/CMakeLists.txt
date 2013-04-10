include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# deal with the detection TOD module
ectomodule(ecto_detection DESTINATION ${PROJECT_NAME}
                          INSTALL  
                          detection/DescriptorMatcher.cpp
                          detection/GuessGenerator.cpp
                          detection/ModelReader.cpp
                          detection/module.cpp
                          common/adjacency_ransac.cpp
                          common/maximum_clique.cpp
)

link_ecto(ecto_detection ${OpenCV_LIBRARIES}
                         ${catkin_LIBRARIES} 
                         ${Boost_LIBRARIES}
)

# deal with the training TOD modules
ectomodule(ecto_training DESTINATION ${PROJECT_NAME}
                         INSTALL
                         training/ModelFiller.cpp
                         training/module.cpp
                         training/Trainer.cpp
                         training/training.cpp
                         common/adjacency_ransac.cpp
                         common/maximum_clique.cpp
)

link_ecto(ecto_training ${catkin_LIBRARIES}
                        ${OpenCV_LIBRARIES}
)
