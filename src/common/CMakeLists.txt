# deal with the detection TOD module

find_package(object_recognition_core REQUIRED)

rosbuild_lite_init()

find_pcl_package()

include_directories(SYSTEM
    ${object_recognition_core_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(tod_common SHARED
    maximum_clique.cpp
    adjacency_ransac.cpp
)

if (ROS_ELECTRIC_FOUND)
    find_library(pcl_ros_io_LIBRARIES pcl_ros_io /opt/ros/electric/stacks/perception_pcl/pcl_ros/lib)
    set_target_properties(tod_common
    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed /opt/ros/electric/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
    )

    target_link_libraries(tod_common ${OpenCV_LIBS}
                                     ${PCL_LIBRARIES}
                                     ${pcl_ros_io_LIBRARIES}
    )
else()
    target_link_libraries(tod_common ${OpenCV_LIBRARIES}
                                     ${PCL_LIBRARIES}
    )
endif()

install(TARGETS tod_common
        DESTINATION lib
)
