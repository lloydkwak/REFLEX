import numpy as np
from robosuite.wrappers.wrapper import Wrapper

class FaultInjectionWrapper(Wrapper):
    """
    REFLEX Environment: Physical Fault Injection Wrapper.
    Simulates Out-of-Distribution (OOD) hardware failures by dynamically altering 
    MuJoCo's internal physical properties (friction and damping) during runtime.
    """
    def __init__(self, env, fault_prob=0.005, max_faults=1, fault_type="lock"):
        """
        Initializes the fault injection wrapper.
        
        Args:
            env (robosuite.models.Env): The base robosuite environment.
            fault_prob (float): Probability of a fault occurring at any given timestep.
            max_faults (int): Maximum number of concurrent joint failures allowed.
            fault_type (str): Type of fault to inject (e.g., "lock", "free-swing").
        """
        super().__init__(env)
        self.fault_prob = fault_prob
        self.max_faults = max_faults
        self.fault_type = fault_type
        
        # Track the state of failures
        self.robot = self.env.robots[0]
        self.num_dof = self.robot.dof
        self.fault_mask = np.zeros(self.num_dof)
        
        # Cache original physical parameters for resetting
        self._original_damping = None
        self._original_friction = None

    def reset(self):
        """
        Resets the environment and clears all physical fault states.
        Restores MuJoCo physics parameters to their original operational values.
        """
        obs = super().reset()
        
        # Cache the original physics properties on the first reset
        dof_indices = self.robot._ref_joint_vel_indices
        if self._original_damping is None:
            self._original_damping = np.copy(self.env.sim.model.dof_damping[dof_indices])
            self._original_friction = np.copy(self.env.sim.model.dof_frictionloss[dof_indices])
            
        # Restore original physics properties
        self.env.sim.model.dof_damping[dof_indices] = self._original_damping
        self.env.sim.model.dof_frictionloss[dof_indices] = self._original_friction
        
        # Reset fault mask and notify the controller
        self.fault_mask = np.zeros(self.num_dof)
        self._notify_controller()
        
        return obs

    def step(self, action):
        """
        Advances the simulation by one step. Stochastically injects physical 
        faults before passing the action to the base environment.
        """
        # 1. Stochastic Fault Trigger
        current_faults = np.sum(self.fault_mask)
        if current_faults < self.max_faults and np.random.rand() < self.fault_prob:
            self._inject_fault()

        # 2. Execute Action in Base Environment
        # The FaultTolerantOSC controller inside the base env will inherently 
        # project the action into the null-space of the failed joints.
        obs, reward, done, info = super().step(action)
        
        # Append fault status to the info dict for evaluation metrics
        info["fault_mask"] = np.copy(self.fault_mask)
        info["active_faults"] = np.sum(self.fault_mask)
        
        return obs, reward, done, info

    def _inject_fault(self):
        """
        Executes the physical failure injection mechanism at the MuJoCo engine level.
        Selects a random operational joint and simulates a mechanical lock.
        """
        # Find operational joints
        available_joints = np.where(self.fault_mask == 0)[0]
        if len(available_joints) == 0:
            return
            
        # Select a target joint for failure
        target_joint = np.random.choice(available_joints)
        
        # Map robot joint index to MuJoCo global DOF index
        dof_idx = self.robot._ref_joint_vel_indices[target_joint]
        
        if self.fault_type == "lock":
            # Simulate a locked joint (e.g., broken gearbox or seized bearing)
            # by applying immense damping and friction loss in the physics engine.
            self.env.sim.model.dof_damping[dof_idx] = 1e6
            self.env.sim.model.dof_frictionloss[dof_idx] = 1e6
            print(f"[REFLEX WARNING] Physical Fault Injected: Joint {target_joint} LOCKED.")
            
        self.fault_mask[target_joint] = 1
        self._notify_controller()

    def _notify_controller(self):
        """
        Transmits the hardware fault mask to the underlying operational space controller.
        This enables the 'reflex' mechanism via Null-space projection.
        """
        controller = self.robot.controller
        if hasattr(controller, 'update_fault_mask'):
            controller.update_fault_mask(self.fault_mask)
        else:
            raise NotImplementedError("The base environment must utilize a FaultTolerantOSC controller.")