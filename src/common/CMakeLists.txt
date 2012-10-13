# deal with the detection TOD module

find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS})

add_library(tod_common SHARED
    maximum_clique.cpp
    adjacency_ransac.cpp
)

target_link_libraries(tod_common ${OpenCV_LIBRARIES})

install(TARGETS tod_common
        DESTINATION lib
)
