cmake_minimum_required(VERSION 3.0.2)
project(sensor_fusion)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# System deps
find_package(Eigen3 REQUIRED)

## Find GeographicLib
find_package(PkgConfig REQUIRED)
pkg_check_modules(GEOGRAPHICLIB REQUIRED geographiclib)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GEOGRAPHICLIB_INCLUDE_DIRS}
)

link_directories(${GEOGRAPHICLIB_LIBRARY_DIRS})

add_executable(gps_odom_fuser src/gps_odom_fuser.cpp)
add_dependencies(gps_odom_fuser ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Prefer the imported target; fallback for older setups noted below
target_link_libraries(gps_odom_fuser
  ${catkin_LIBRARIES}
  ${GEOGRAPHICLIB_LIBRARIES}
)

# ---- If your distro doesn't provide the imported target, replace the line above with:
# target_link_libraries(gps_odom_fuser ${catkin_LIBRARIES} Geographic)
