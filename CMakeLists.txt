cmake_minimum_required (VERSION 2.8)
project(point_cloud_registration)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)
find_package (Ceres REQUIRED)
find_package (Glog REQUIRED)
find_package (Boost REQUIRED)

# catkin_simple(ALL_DEPS_REQUIRED)
FILE(GLOB_RECURSE LibFiles "include/*")

include_directories(${CMAKE_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDES}
    ${Boost_INCLUDES}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME}
  src/robust_pcl_registration/point_cloud_registration.cc
  src/robust_pcl_registration/pda.cc
  src/external_pcl_registration/gicp.cc
  src/external_pcl_registration/icp.cc
  src/external_pcl_registration/ndt.cc
  ${LibFiles})

target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES} ${PCL_LIBRARIES}
    ${Boost_LIBRARIES})

# Pointcloud alignment using ICP.
add_executable(icp src/main_icp.cc)
target_link_libraries(icp ${PROJECT_NAME})

# Pointcloud alignment using GICP.
add_executable(gicp src/main_gicp.cc)
target_link_libraries(gicp ${PROJECT_NAME})

# Pointcloud alignment using NDT.
add_executable(ndt src/main_ndt.cc)
target_link_libraries(ndt ${PROJECT_NAME})

# Pointcloud alignment using IPDA ("Robust ICP").
add_executable(ipda src/main_ipda.cc)
target_link_libraries(ipda ${PROJECT_NAME})

# Executable to convert pointcloud .xyz to .pcd format.
add_executable(xyz_to_pcd src/main_xyz_to_pcd.cc)
target_link_libraries(xyz_to_pcd ${PROJECT_NAME})

#add_custom_target(headers SOURCES ${LibFiles})

