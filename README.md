# REFLEX: REsilient Fault-tolerant robot control via Linearized Energy-field X-transfer

**REFLEX** is a hierarchical robot control architecture designed to ensure operational continuity under physical joint failures. By decoupling cognitive intent from physical execution, the system maintains task-space goals through a "spinal reflex" mechanism.

## 1. Core Architecture
The system consists of three distinct layers to isolate cognitive reasoning from hardware-specific constraints:
- **FM Layer (Front-End)**: Generates a task-space Potential Gradient based on visual observations, independent of robot morphology.
- **IR Translation (Mid-End)**: Samples the field into a 6-DOF target velocity $\dot{x}_{IR}$ and stiffness parameters ($K_p, K_d$).
- **OSC Layer (Back-End)**: A fault-tolerant Operational Space Controller that maps the IR to joint torques while masking failed joints via Null-space projection.

## 2. Key Hypotheses
- **Cognitive Resilience**: Cognitive models maintain intent stability by ignoring low-level joint-space anomalies.
- **Instant Recovery**: Null-space projection provides sub-millisecond physical adaptation, far exceeding the speed of neural re-planning.

## 3. Installation
```bash
docker build -t reflex-env ./docker
docker run --gpus all -it reflex-env
```

## 4. References & Open-Source Acknowledgements
This project stands on the shoulders of several foundational open-source repositories and academic breakthroughs. We heavily rely on the following works:

### Libraries & Frameworks
- **[robosuite](https://github.com/ARISE-Initiative/robosuite)**: Provides the robust Operational Space Control (OSC) baseline and the MuJoCo-based benchmark environments used for fault injection.
- **[robomimic](https://github.com/ARISE-Initiative/robomimic)**: Utilized for high-quality offline datasets (HDF5) and standard task definitions (`PickPlaceCan`, `NutAssemblySquare`).
- **[torchcfm](https://github.com/atong01/conditional-flow-matching)**: Powers the exact marginal Optimal Transport Conditional Flow Matching (OT-CFM) engine in our brain layer.
- **[torchdiffeq](https://github.com/rtqichen/torchdiffeq)**: Used for fast and stable ODE integration (Euler method) during real-time reflex inference.

### Academic Literature
- **Kinematics & Control**: Khatib, O. (1987). *A unified approach for motion and force control of robot manipulators*. (The theoretical foundation for dynamically consistent null-space projection).
- **SE(3) Generative Models**: Urain, J., et al. (2022). *SE(3)-DiffusionFields: Learning smooth orientations and trajectory generation from demonstration*.
- **Optimal Transport**: Tong, A., et al. (2023). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*.
- **Baseline Architectures**: Chi, C., et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*.