cmake_minimum_required(VERSION 3.23)

project(duplicate_finder)

# include(FetchContent)
# set(FETCHCONTENT_QUIET FALSE)
# FetchContent_Declare(
#   opencv
#   GIT_REPOSITORY https://github.com/opencv/opencv.git
#   GIT_TAG 4.8.0
#   GIT_SHALLOW TRUE
#   GIT_PROGRESS TRUE)
# FetchContent_MakeAvailable(opencv)

set(OpenCV_DIR "C:/opencv/build/")

find_package(OpenCV)

#include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} "src/main.cpp")

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/modules/nlohmann_json_cmake_fetchcontent/single_include")
#"${CMAKE_CURRENT_SOURCE_DIR}/modules/opencv_contrib/modules/img_hash/include")
