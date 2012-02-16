# deal with the detection TOD module

add_library(tod_common
  SHARED
  maximum_clique.cpp
  adjacency_ransac.cpp
  )

find_package(Boost REQUIRED)
find_package(PCL REQUIRED)

target_link_libraries(tod_common
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS tod_common
    DESTINATION lib
    )
