#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='husky_navigation',
            executable='odom_reset_node.py',
            name='odom_reset_node',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='husky_navigation',
            executable='row_manager_fsm.py',
            name='row_manager_fsm',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'pose_source': 'gt_ign',
                'world': 'field',
                'model_name_match': 'a200_0000/robot'
            }]
        ),

        Node(
            package='husky_navigation',
            executable='resnet_pixel_predictor.py',
            name='resnet_pixel_predictor',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='husky_navigation',
            executable='pixel_to_odom_converter.py',
            name='pixel_to_odom_converter',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='husky_navigation',
            executable='mpc_tracking_controller.py',
            name='mpc_tracking_controller',
            output='screen',
            emulate_tty=True
        )
    ])
