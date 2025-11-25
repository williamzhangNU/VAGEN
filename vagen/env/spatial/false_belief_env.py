import gymnasium as gym
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.env import SpatialGym
from vagen.env.spatial.Base.tos_base import (
    ExplorationManager,
    HistoryManager,
    BaseAction,
    Agent,
    Room
)
from vagen.env.spatial.Base.tos_base.prompts import PromptManager
from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json, get_room_description
from vagen.env.spatial.Base.tos_base.utils.image_handler import ImageHandler
from vagen.env.spatial.Base.tos_base.actions.actions import ActionSequence, TermAction

from vagen.env.spatial.room_modifier import SingleObjectModifier
from vagen.env.spatial.Base.tos_base.prompts.false_belief_prompts import FALSE_BELIEF_INSTRUCTION

class FalseBeliefEnv(SpatialGym):
    """
    False Belief Task Environment.
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__(config)
        self.target_object_name: Optional[str] = None
        self.history_dir = config.kwargs.get('history_dir')
        if not self.history_dir:
            raise ValueError("history_dir must be provided in config.kwargs for FalseBeliefEnv")
            
        self.replay_steps = 0
        self.current_step_count = 0
        self.target_observed = False
        self.modified_room = None

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super(SpatialGym, self).reset(seed=seed)
        
        # 1. Load history manager
        old_history_manager = HistoryManager.load_from_dir(self.history_dir, eval_override=False)
        
        # 2. Initialize ImageHandler using data from history
        base_dir = os.path.dirname(old_history_manager.image_dir)
        run_name = os.path.basename(old_history_manager.image_dir)
        try:
            run_seed = int(run_name.replace('run', ''))
        except ValueError:
            run_seed = 0
        
        self.image_handler = ImageHandler(base_dir, run_seed, self.config.image_size)
        self.json_data = self.image_handler.json_data
        
        # 3. Restore Room and Agent from history (Initial State)
        self.initial_room = Room.from_dict(old_history_manager.room_dict)
        self.agent = Agent.from_dict(old_history_manager.agent_dict)
        
        # Reset agent to INITIAL position
        self.agent.pos = self.agent.init_pos.copy()
        self.agent.ori = self.agent.init_ori.copy()
        if self.agent.init_room_id is not None:
            self.agent.room_id = self.agent.init_room_id
        self.initial_agent = self.agent.copy()

        # 4. Prepare Modified Room (but don't use it yet)
        modifier = SingleObjectModifier()
        self.modified_room, self.target_object_name = modifier.modify(self.initial_room, self.np_random)
        
        # 5. Setup Managers
        self.prompter = PromptManager(self.config, self.np_random, self.image_handler)
        
        self.remaining_exp_steps = self.config.max_exp_steps
        self.turn_logs = []
        self.current_turn_number = 0
        self.observed_image_paths = []
        self.is_exploration_phase = True
        
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Start with INITIAL room
        self.exploration_manager = ExplorationManager(
            self.initial_room, self.agent,
            grid_size=(self.config.grid_size if hasattr(self.config, 'grid_size') else None),
        )
        
        # 6. Setup New History Manager
        self.history_manager = HistoryManager(
            self.config.get_observation_config(), self.config.get_model_config(),
            old_history_manager.room_dict, self.agent.to_dict(),
            image_dir=self.image_handler.image_dir,
            output_dir=old_history_manager.output_dir, 
            seed=seed,
            eval_override=False,
            all_override=False, 
        )
        
        # Update room dict to modified one for saving
        self.history_manager.room_dict = self.modified_room.to_dict()
        self.history_manager.exploration_path = os.path.join(self.history_manager.output_dir, "false_belief_turn_logs.json")
        self.history_manager.messages_path = os.path.join(self.history_manager.output_dir, "false_belief_messages.json")

        # 7. Get History Responses for Replay
        history_responses = old_history_manager.get_responses()
        if not history_responses:
            raise ValueError("History responses must not be empty for False Belief Task")
        self.replay_steps = len(history_responses)
        self.current_step_count = 0
        self.target_observed = False
        
        # 8. Generate Initial Observation (Standard)
        obs, final_loc = self._generate_initial_observation()
        self.render_cache = obs

        # Initialize messages
        self.history_manager.init_messages(self.prompter.system_prompt())
        self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
        self.history_manager.save_messages(final_loc)
        self.observed_image_paths = []
        
        # Return observation and history
        info = {'history': history_responses}
        return obs, info

    def step(self, llm_response: str):
        """Process agent actions."""
        
        # Execute step (using current room, which is initial_room during replay)
        obs, reward, done, info = super().step(llm_response)
        
        self.current_step_count += 1
        
        # If this was the last replay step, switch to modified room and inject prompt
        if self.current_step_count == self.replay_steps:
            # Switch to modified room
            self.exploration_manager.room = self.modified_room
            
            # Reset agent to INITIAL position and orientation
            self.agent.pos = self.agent.init_pos.copy()
            self.agent.ori = self.agent.init_ori.copy()
            if self.agent.init_room_id is not None:
                self.agent.room_id = self.agent.init_room_id
            
            # Inject False Belief Prompt (REPLACE, not append)
            prompt = FALSE_BELIEF_INSTRUCTION.format(target_object=self.target_object_name)
            obs['obs_str'] = prompt
            
            # Reset step budget for False Belief phase
            self.remaining_exp_steps = self.config.max_exp_steps
                
        elif self.current_step_count > self.replay_steps:
            # False Belief Phase Logic
            
            if done:
                # Check if target is visible ONLY at termination
                is_visible = self._check_target_visibility()
                self.target_observed = is_visible
                info['if_observed'] = self.target_observed
                
                # Check if we should reward
                if self.target_observed:
                    reward = 1.0
                    info['success'] = True
                else:
                    reward = 0.0
                    info['success'] = False
            else:
                # Not done yet, no reward, no observation check needed for reward
                pass

        return obs, reward, done, info
            
    def _check_target_visibility(self) -> bool:
        """Check if target object is currently visible to the agent."""
        if not self.target_object_name or not self.modified_room:
            return False
        target_obj = self.modified_room.get_object_by_name(self.target_object_name)
        if not target_obj:
            return False
            
        # Re-use BaseAction visibility logic
        class DummyAction(BaseAction):
            def execute(self, *args, **kwargs): pass
            def success_message(self, **kwargs): return ""
            def error_message(self, *args): return ""
            
        dummy_action = DummyAction("dummy")
        return dummy_action._is_visible(self.agent, target_obj)
