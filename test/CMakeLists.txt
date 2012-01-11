# TOD tests
object_recognition_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.tod)
object_recognition_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.tod)

add_subdirectory(tod)
