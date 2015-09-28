# CMake 最低版本号
cmake_minimum_required ( VERSION 2.8 )

# 工程名称
project( SIFT-Human-Track )


find_package( OpenCV REQUIRED )
include_directories( ${root_SOURCE_DIR}/include )


aux_source_directory( ${root_SOURCE_DIR}/src DIR_SRCS )


add_executable( SIFT-Human-Track ${DIR_SRCS} )
target_link_libraries( SIFT-Human-Track ${OpenCV_LIBS} )

