# Toward Generalizable Robotic Assembly: A Prior-Guided Deep Reinforcement Learning Approach with Multi-Sensor Information

This repository contains the official implementation for the research on a prior-guided deep reinforcement learning (DRL) framework for generalizable robotic assembly, utilizing multi-sensor (vision and force/torque) information as **prior-guided knowledge**. The framework decomposes the complex assembly task into three sequential skills: **Search**, **Alignment**, and **Insertion**, each trained with PG-SAC.


## 📁 Project Structure
The core directories and files are as follows:

- **`algorithms/`**: Core algorithm implementations.
- **`data/`**: Stores visual datasets, experimental data.
- **`draw/`**: Scripts for plotting experimental results and data visualization.
- **`envs/`**: Reinforcement learning training environments.
- **`model/`**: Pre-trained models, including visual feature extraction networks and RL policy models.
- **`robot_control/`**: Robot control interfaces, and handlers for reading/processing data from vision and force/torque sensors.
- **`scenes/`**: CoppeliaSim simulation scenes. The main scene is `SkillGeneralization_vision.ttt`.
- **`support_files/`**: Configuration files for the CoppeliaSim Remote API to enable Python communication.
- **`align_SAC.py`**: Main script to **train or test** the **Align** skill.
- **`insert_SAC.py`**: Main script to **train or test** the **Insert** skill.
- **`search_SAC.py`**: Main script to **train or test** the **Search** skill.
- **`SAC_process.py`**: Main script for the **full-process assembly** execution.
- **`test_cv.py`**: Script for preprocessing **visual feature extraction**.

## 🚀 Getting Started

### Prerequisites
1.  **Simulator:** Install [CoppeliaSim Edu V4.1.0](https://www.coppeliarobotics.com/).
2.  **Python:** Version 3.8 or higher is recommended. Create and activate a virtual environment.

### How to Run
1.  **Launch Simulation:**
   Start CoppeliaSim and open the main scene: `scenes/SkillGeneralization_vision.ttt`.
3.  **Train/Test Individual Skills:**
   Each skill can be trained or tested independently.

- **Search Skill**: `python search_SAC.py`
- **Alignment Skill**: `python align_SAC.py`
- **Insertion Skill**: `python insert_SAC.py`

## 🔧 Key Configuration
*   The mode (Simulation **SIM** or Physical **REAL**) is configured within the `envs/` directory.
*   Update model paths in the main scripts (`SAC_process.py`, `*_SAC.py`) if your trained models are stored in a non-default location.

## 📈 Results and Plots
Trained models are saved in `model/.` To visualize training curves or experimental results, use the scripts in the `draw/`folder. Logged data from experiments is stored in `data/`.
