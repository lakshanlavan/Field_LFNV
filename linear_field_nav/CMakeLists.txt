cmake_minimum_required(VERSION 3.10)
project(husky_navigation)

#--------------------------------------------------------------------
# Compiler warnings (only relevant if you later add C++ nodes)
#--------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#--------------------------------------------------------------------
# Dependencies
#--------------------------------------------------------------------
find_package(ament_cmake          REQUIRED)
find_package(ament_cmake_python   REQUIRED)
find_package(rclpy                REQUIRED)
find_package(geometry_msgs        REQUIRED)
find_package(nav_msgs             REQUIRED)
find_package(std_msgs             REQUIRED)
find_package(diagnostic_msgs      REQUIRED)
find_package(sensor_msgs          REQUIRED)
find_package(cv_bridge            REQUIRED)


install(
  PROGRAMS
    scripts/row_manager_fsm.py
    scripts/user_input_node.py
    scripts/path_planning_node.py
    scripts/instruction_generator_node.py
    scripts/plot_mpc_paths.py
    scripts/image_saver.py
    scripts/odom_reset_node.py
    scripts/segment_manager_node.py
    scripts/resnet_pixel_predictor.py
    scripts/pixel_to_odom_converter.py
    scripts/mpc_tracking_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

#--------------------------------------------------------------------
# Launch and configuration files
#--------------------------------------------------------------------
install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

#--------------------------------------------------------------------
# Tests (optional)
#--------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
