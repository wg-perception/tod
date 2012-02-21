# deal with the detection TOD module

find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM
    ${object_recognition_core_INCLUDE_DIRS}
)

add_library(tod_common SHARED
    maximum_clique.cpp
    adjacency_ransac.cpp
)

set_target_properties(tod_common
    PROPERTIES
    COMPILE_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
    LINK_FLAGS "-Wl,--no-as-needed /opt/ros/fuerte/lib/libpcl_io.so /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib/libpcl_ros_io.so -Wl,--as-needed"
)

target_link_libraries(tod_common
    ${OpenCV_LIBS}
    ${pcl_LIBRARIES}
    ${pcl_ros_io_LIBRARIES}
)

install(TARGETS tod_common
    DESTINATION lib
)
