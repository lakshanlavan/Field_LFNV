# Field_LFNV — Linear Field Navigation for Agricultural Robotics  
*ROS 2 Humble · Ignition Gazebo Fortress · Clearpath Husky*

---

## Overview
**Field_LFNV** implements a full **linear-field navigation pipeline** for the Clearpath Husky A200.  
It integrates **crop-path perception**, **reference-path generation**, **nonlinear MPC trajectory tracking**,  
and a **finite-state row-manager FSM** to demonstrate robust autonomous navigation in structured fields.

This repository contains **only custom simulation assets and ROS 2 nodes**.  
Vendor Husky models/meshes from `clearpath_gz` are referenced at runtime and **not re-distributed**.

<p align="center">
  <img src="assets/field_layout.png" alt="Linear field Ignition world" width="70%" height="420px">
</p>

*Ignition Gazebo world with crop rows and Husky robot.*

---

## Prerequisites

- **ROS 2 Humble** (Linux)  
- **Ignition Gazebo Fortress** (`ign gazebo` on PATH)  
- **Clearpath Husky simulation packages**  
- Python deps: `python3-opencv python3-numpy` (plus any you import)

### Clearpath Husky Simulation Setup (once)

```bash
# Create workspace
mkdir -p ~/agri_spiral_ws/src
cd ~/agri_spiral_ws/src
```

# Clone Clearpath repos (Humble)
git clone https://github.com/clearpathrobotics/clearpath_common.git    -b humble
git clone https://github.com/clearpathrobotics/clearpath_config.git    -b humble
git clone https://github.com/clearpathrobotics/clearpath_msgs.git      -b humble
git clone https://github.com/clearpathrobotics/clearpath_simulator.git -b humble

# Install deps & build
cd ~/agri_spiral_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
echo "source ~/agri_spiral_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Let Ignition (Fortress) find Clearpath assets
export IGN_GAZEBO_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH:$(ros2 pkg prefix clearpath_gz)/share"
