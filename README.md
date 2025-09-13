# Field_LFNV — Linear Field Navigation (ROS 2 Humble + Ignition)

Clearpath Husky linear field navigation: crop-path prediction, reference-path building,
MPC trajectory tracking, and row-manager FSM.

## Requires
- ROS 2 **Humble** (source): `source /opt/ros/humble/setup.bash`
- **Ignition Gazebo Fortress** (`ign gazebo` on PATH)
- **Clearpath Husky simulation**: `sudo apt install ros-humble-husky-simulator`
- Python deps: `python3-opencv python3-numpy` (plus any you import)

> Vendor models/worlds from `clearpath_gz` are **not copied** here. We reference them via `model://…`.
> Add Clearpath’s share path to Ignition’s resource path (Fortress):
```bash
export IGN_GAZEBO_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH:$(ros2 pkg prefix clearpath_gz)/share"
