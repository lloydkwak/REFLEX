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
