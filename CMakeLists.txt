# CMake 最低版本号
cmake_minimum_required ( VERSION 2.8 )

# 工程名称
project( SIFT-Human-Track )

find_package( OpenCV REQUIRED )
include_directories( ${root_SOURCE_DIR}/include )


add_subdirectory( src bin )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
