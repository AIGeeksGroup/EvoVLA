"""
DISCOVERSE Environment Wrapper

Provides a unified interface for the DISCOVERSE simulator with:
- Headless rendering with image access
- SE(3) pose extraction for POE
- Stage tracking and gating
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any, Optional, List


class DiscoverseEnv:
    """
    DISCOVERSE Environment Wrapper for EvoVLA Training.
    
    Supports three tasks:
    - bridge: Block Bridge Place (74 stages)
    - jujube_cup: Place Jujube Coffee Cup (19 stages)
    - stack: Stack Block (18 stages)
    """
    
    def __init__(
        self,
        task_name: str,
        headless: bool = True,
        randomize: bool = True,
        stage_window_size: int = 4,
        stage_completion_threshold: float = 0.02,
        stage_metric: str = "delta",
        enable_relaxed_success: bool = True,
    ):
        """
        Args:
            task_name: Task name ("bridge", "jujube_cup", "stack")
            headless: Headless mode (True=no GUI, but can get images)
            randomize: Enable domain randomization
            stage_window_size: Window size for stage gating
            stage_completion_threshold: Threshold for stage advancement
            stage_metric: Gating metric ("delta" or "score")
            enable_relaxed_success: Enable relaxed success criteria
        """
        self.task_name = task_name
        self.randomize = randomize
        self.stage_window_size = max(int(stage_window_size), 1)
        self.stage_completion_threshold = float(stage_completion_threshold)
        self.stage_metric = stage_metric if stage_metric in {"delta", "score"} else "delta"
        self._relaxed_success_enabled = bool(enable_relaxed_success)
        
        # Task configuration
        self._setup_task(task_name, headless)
        
        # State tracking
        self.current_stage = 0
        self.current_step = 0
        self.max_steps = 400
        
        # Stage gating mechanism
        self.stage_reward_window: List[float] = []
        self.stage_score_window: List[float] = []
        
        print(f"✓ DiscoverseEnv initialized: {task_name}")
        print(f"  headless={headless}, max_stages={self.max_stages}")
        print(f"  domain_randomization={self.randomize}")
    
    def _setup_task(self, task_name: str, headless: bool):
        """Set up task-specific configuration."""
        try:
            # Import DISCOVERSE components
            from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
            from discoverse.utils import get_body_tmat
            self.get_body_tmat = get_body_tmat
            
            cfg = AirbotPlayCfg()
            cfg.headless = headless
            cfg.sync = False
            cfg.use_gaussian_renderer = False
            cfg.render_set = {"fps": 20, "width": 448, "height": 448}
            cfg.obs_rgb_cam_id = [0, 1]
            cfg.obs_depth_cam_id = []
            
            if task_name == "bridge":
                from discoverse.examples.tasks_airbot_play.block_bridge_place import SimNode, cfg as task_cfg
                cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_bridge_place.xml"
                self.max_stages = 74
                self.task_description = "build a bridge with green bars and fill with purple blocks"
                self.target_objects = ["block1_green", "block2_green"]
                
            elif task_name == "jujube_cup":
                from discoverse.examples.tasks_airbot_play.place_jujube_coffeecup import SimNode, cfg as task_cfg
                cfg.mjcf_file_path = "mjcf/tasks_airbot_play/place_jujube_coffeecup.xml"
                self.max_stages = 19
                self.task_description = "put the jujube into the coffee cup and place the cup on the plate"
                self.target_objects = ["jujube", "coffeecup_white"]
                
            elif task_name == "stack":
                from discoverse.examples.tasks_airbot_play.stack_block import SimNode, cfg as task_cfg
                cfg.mjcf_file_path = "mjcf/tasks_airbot_play/stack_block.xml"
                self.max_stages = 18
                self.task_description = "stack the green block on the blue block, then stack the red block on top"
                self.target_objects = ["block_green", "block_blue", "block_red"]
            else:
                raise ValueError(f"Unknown task: {task_name}")
            
            # Apply configuration
            for key in ['headless', 'sync', 'use_gaussian_renderer', 'render_set',
                       'obs_rgb_cam_id', 'obs_depth_cam_id', 'mjcf_file_path']:
                if hasattr(cfg, key):
                    setattr(task_cfg, key, getattr(cfg, key))
            
            self.sim_node = SimNode(task_cfg)
            self.cfg = task_cfg
            self._discoverse_available = True
            
        except ImportError:
            print("⚠️ DISCOVERSE not available, using dummy environment")
            self._discoverse_available = False
            self.max_stages = {"bridge": 74, "jujube_cup": 19, "stack": 18}.get(task_name, 18)
            self.task_description = f"Complete the {task_name} task"
            self.target_objects = []
            self.get_body_tmat = lambda *args: np.eye(4)
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment.
        
        Returns:
            obs: Observation dictionary
        """
        if self._discoverse_available:
            try:
                obs_raw = self.sim_node.reset()
                if self.randomize:
                    self.sim_node.domain_randomization()
                obs = self._format_observation(obs_raw)
            except Exception as e:
                print(f"Reset failed, retrying: {e}")
                return self.reset()
        else:
            obs = self._create_dummy_observation()
        
        # Reset state
        self.current_stage = 0
        self.current_step = 0
        self.stage_reward_window = []
        self.stage_score_window = []
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action.
        
        Args:
            action: 7-DoF action array
        
        Returns:
            obs: Observation
            reward: Extrinsic reward
            done: Episode termination flag
            info: Additional information
        """
        self.current_step += 1
        
        if self._discoverse_available:
            try:
                action_array = np.array(action, dtype=np.float32)
                obs_raw, _, _, _, _ = self.sim_node.step(action_array)
                
                if self.sim_node.mj_data.time > 70.0:
                    obs = self._format_observation(obs_raw)
                    info = {
                        'current_stage': self.current_stage,
                        'task_name': self.task_name,
                        'success': False,
                        'timeout': True,
                        'step': self.current_step,
                    }
                    return obs, 0.0, True, info
                
                obs = self._format_observation(obs_raw)
                success = self._check_success()
                reward = 1.0 if success else 0.0
                
            except Exception as e:
                print(f"Step error: {e}")
                obs = self._create_dummy_observation()
                success = False
                reward = 0.0
        else:
            obs = self._create_dummy_observation()
            success = False
            reward = 0.0
        
        max_step_reached = self.current_step >= self.max_steps
        done = success or max_step_reached
        
        info = {
            'current_stage': self.current_stage,
            'task_name': self.task_name,
            'success': success,
            'timeout': max_step_reached,
            'step': self.current_step
        }
        
        return obs, reward, done, info
    
    def _format_observation(self, obs_raw: Any) -> Dict[str, Any]:
        """Format observation for OpenVLA."""
        # Get images
        if hasattr(self.sim_node, 'img_rgb_obs_s') and self.sim_node.img_rgb_obs_s:
            full_image = self.sim_node.img_rgb_obs_s.get(0, np.zeros((448, 448, 3), dtype=np.uint8))
            wrist_image = self.sim_node.img_rgb_obs_s.get(1, full_image)
        else:
            full_image = np.zeros((448, 448, 3), dtype=np.uint8)
            wrist_image = np.zeros((448, 448, 3), dtype=np.uint8)
        
        full_image_pil = Image.fromarray(full_image.astype(np.uint8))
        
        # Extract SE(3) poses
        try:
            ee_pose = self.get_body_tmat(self.sim_node.mj_data, "link6")
            target_obj = self._get_current_target_object()
            obj_pose = self.get_body_tmat(self.sim_node.mj_data, target_obj)
        except:
            ee_pose = np.eye(4)
            obj_pose = np.eye(4)
        
        # Extract proprioception
        state = self.sim_node.mj_data.qpos[:7].copy()
        
        return {
            'image': full_image_pil,
            'full_image': full_image,
            'wrist_image': wrist_image,
            'state': state,
            'task_description': self.task_description,
            'ee_pose': ee_pose,
            'obj_pose': obj_pose
        }
    
    def _create_dummy_observation(self) -> Dict[str, Any]:
        """Create dummy observation when DISCOVERSE is not available."""
        full_image = np.zeros((448, 448, 3), dtype=np.uint8)
        return {
            'image': Image.fromarray(full_image),
            'full_image': full_image,
            'wrist_image': full_image.copy(),
            'state': np.zeros(7),
            'task_description': self.task_description,
            'ee_pose': np.eye(4),
            'obj_pose': np.eye(4)
        }
    
    def _get_current_target_object(self) -> str:
        """Get current target object based on task and stage."""
        if self.target_objects:
            return self.target_objects[0]
        return "block_green"
    
    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        if not self._discoverse_available:
            return False
        try:
            return bool(self.sim_node.check_success())
        except:
            return False
    
    def update_stage_gating(
        self,
        r_stage: float,
        u_k: Optional[float] = None
    ) -> Tuple[bool, int, int]:
        """
        Stage gating mechanism (Paper Appendix).
        
        Maintains sliding window (size=m), advances to next stage when
        average of recent m values exceeds threshold.
        
        Args:
            r_stage: Current stage-aligned reward (from SAR)
            u_k: Current stage score (optional, for score-based gating)
        
        Returns:
            advanced: Whether stage was advanced
            old_stage: Previous stage
            new_stage: Current stage
        """
        metric_value = None
        
        if self.stage_metric == "score":
            if u_k is None:
                return False, self.current_stage, self.current_stage
            
            self.stage_score_window.append(u_k)
            if len(self.stage_score_window) > self.stage_window_size:
                self.stage_score_window.pop(0)
            
            if len(self.stage_score_window) >= self.stage_window_size:
                metric_value = float(np.mean(self.stage_score_window))
        else:
            self.stage_reward_window.append(float(r_stage))
            if len(self.stage_reward_window) > self.stage_window_size:
                self.stage_reward_window.pop(0)
            
            if len(self.stage_reward_window) >= self.stage_window_size:
                metric_value = float(np.mean(self.stage_reward_window))
        
        if (
            metric_value is not None
            and metric_value >= self.stage_completion_threshold
            and self.current_stage < self.max_stages - 1
        ):
            old_stage = self.current_stage
            self.current_stage += 1
            self.stage_reward_window = []
            self.stage_score_window = []
            return True, old_stage, self.current_stage
        
        return False, self.current_stage, self.current_stage
    
    def close(self):
        """Close environment."""
        if hasattr(self, 'sim_node'):
            del self.sim_node

