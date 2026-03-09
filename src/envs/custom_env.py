import numpy as np
from robosuite.wrappers.wrapper import Wrapper

class FaultInjectionWrapper(Wrapper):
    """
    REFLEX Environment: Physical Fault Injection Wrapper.
    
    Simulates Out-of-Distribution (OOD) hardware failures by dynamically altering 
    MuJoCo's internal physical properties (friction and damping) during runtime.
    
    [Research Integrity Fixes Applied]
    1. Pre-reset Physics Restoration: Prevents initial observation corruption.
    2. Multi-fault Scheduling: Supports `max_faults > 1` by generating an array of stochastic trigger steps.
    """
    def __init__(self, env, max_faults=1, fault_type="lock", trigger_range=(0.3, 0.7)):
        """
        Initializes the fault injection wrapper.
        
        Args:
            env (robosuite.models.Env): The base robosuite environment.
            max_faults (int): Maximum number of concurrent joint failures allowed.
            fault_type (str): Type of fault to inject (e.g., "lock", "free-swing").
            trigger_range (tuple): The (min, max) ratio of the episode horizon 
                                   where faults will be stochastically triggered.
        """
        super().__init__(env)
        self.max_faults = max_faults
        self.fault_type = fault_type
        self.trigger_range = trigger_range
        
        # Track the state of the robot and failures
        self.robot = self.env.robots[0]
        self.num_dof = self.robot.dof
        self.fault_mask = np.zeros(self.num_dof)
        
        # Cache original physical parameters for resetting
        self._original_damping = None
        self._original_friction = None
        
        # Step tracking for window-based trigger
        self.current_step = 0
        self.trigger_steps = []  # Changed to array to support max_faults > 1

    def reset(self):
        """
        Resets the environment and clears all physical fault states.
        Restores physics BEFORE super().reset() to ensure mathematical integrity of initial obs.
        """
        # 1. Restore physics BEFORE base reset to prevent corrupting the initial state
        if self._original_damping is not None and hasattr(self, 'robot'):
            dof_indices = self.robot._ref_joint_vel_indices
            self.env.sim.model.dof_damping[dof_indices] = self._original_damping
            self.env.sim.model.dof_frictionloss[dof_indices] = self._original_friction
            
        # 2. Execute base reset (computes initial observation with clean physics)
        obs = super().reset()
        
        # Ensure robot reference is updated in case env recreated it
        self.robot = self.env.robots[0]
        dof_indices = self.robot._ref_joint_vel_indices
        
        # 3. Cache original properties only on the very first reset
        if self._original_damping is None:
            self._original_damping = np.copy(self.env.sim.model.dof_damping[dof_indices])
            self._original_friction = np.copy(self.env.sim.model.dof_frictionloss[dof_indices])
        
        # 4. Reset fault mask and notify the controller
        self.fault_mask = np.zeros(self.num_dof)
        self._notify_controller()
        
        # 5. Schedule exact stochastic steps for multiple fault injections
        self.current_step = 0
        max_steps = getattr(self.env, "horizon", 500)  # Safe fallback if horizon is hidden
        min_trigger = int(max_steps * self.trigger_range[0])
        max_trigger = int(max_steps * self.trigger_range[1])
        
        # Generate 'max_faults' unique random trigger steps within the window
        if max_trigger > min_trigger:
            self.trigger_steps = np.random.choice(
                np.arange(min_trigger, max_trigger), 
                size=self.max_faults, 
                replace=False  # Ensures faults don't trigger on the exact same step
            ).tolist()
        else:
            self.trigger_steps = [min_trigger] * self.max_faults
            
        return obs

    def step(self, action):
        """
        Advances the simulation by one step. Injects a physical fault exactly 
        at the pre-scheduled 'trigger_steps'.
        """
        self.current_step += 1
        
        # 1. Execute Window-based Stochastic Fault Trigger
        if self.current_step in self.trigger_steps:
            current_faults = np.sum(self.fault_mask)
            if current_faults < self.max_faults:
                self._inject_fault()

        # 2. Execute Action in Base Environment
        # The FaultTolerantOSC controller will project the action into the null-space.
        obs, reward, done, info = super().step(action)
        
        # 3. Append fault status to the info dict for robust evaluation metrics
        info["fault_mask"] = np.copy(self.fault_mask)
        info["active_faults"] = np.sum(self.fault_mask)
        info["fault_triggered_now"] = (self.current_step in self.trigger_steps)
        
        return obs, reward, done, info

    def _inject_fault(self):
        """
        Executes the physical failure injection mechanism at the MuJoCo engine level.
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
            # Simulate a mechanically locked joint
            self.env.sim.model.dof_damping[dof_idx] = 1e6
            self.env.sim.model.dof_frictionloss[dof_idx] = 1e6
            
            # Print warning to console for tracking during evaluation
            progress_pct = (self.current_step / getattr(self.env, "horizon", 500)) * 100
            print(f"[REFLEX WARNING] Physical Fault Injected at {progress_pct:.1f}% progression: Joint {target_joint} LOCKED.")
            
        self.fault_mask[target_joint] = 1
        self._notify_controller()

    def _notify_controller(self):
        """
        Transmits the hardware fault mask to the underlying operational space controller.
        """
        controller = self.robot.controller
        if hasattr(controller, 'update_fault_mask'):
            controller.update_fault_mask(self.fault_mask)
        else:
            raise NotImplementedError("The base environment must utilize a FaultTolerantOSC controller to support REFLEX.")