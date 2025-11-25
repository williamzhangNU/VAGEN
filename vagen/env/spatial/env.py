import gymnasium as gym
import numpy as np
import os
from typing import List, Dict, Any, Optional

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.Base.tos_base import (
    ExplorationManager,
    HistoryManager,
    BaseAction,
    RoomGenerator,
    Agent,
    Room,
)
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_agent_proxy
from vagen.env.spatial.Base.tos_base.prompts import PromptManager
from vagen.env.spatial.Base.tos_base.utils.action_utils import action_results_to_text
from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
from vagen.env.spatial.Base.tos_base.utils.env_logger import EnvTurnLog
from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response
from vagen.env.spatial.Base.tos_base.utils.image_handler import ImageHandler
from vagen.env.spatial.Base.tos_base.actions.actions import ForcedTermAction, ActionSequence
from vagen.env.spatial.Base.tos_base.prompts.false_belief_prompts import FALSE_BELIEF_INSTRUCTION
from vagen.env.spatial.room_modifier import SingleObjectModifier


class SpatialGym(gym.Env):
    """
    Spatial Gym Environment (exploration only).
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.prompter: PromptManager = None

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.initial_agent = None

        # Managers
        self.exploration_manager = None
        self.history_manager = None

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = None
        self.current_turn_number = None
        self.observed_image_paths: List[str] = None
        
        # False belief experiment state
        self.is_false_belief_exp = self._check_false_belief_exp()
        self.in_false_belief_phase = False
        self.target_object_name: Optional[str] = None
        self.modified_room = None
        self.target_observed = False

    def _check_false_belief_exp(self) -> bool:
        """Check if the current task is false_belief_exp."""
        if self.config.eval_tasks:
            for task in self.config.eval_tasks:
                if task.get('task_type') == 'false_belief_exp':
                    return True
        return False

    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        exp_history = {}
        images = []
        final_loc = None
        if self.config.exp_type == 'passive':
            proxy = get_agent_proxy(
                self.config.proxy_agent,
                self.initial_room,
                self.agent,
                grid_size=self.config.grid_size if hasattr(self.config, 'grid_size') else None,
            )            
            proxy.run()
            # Only collect multi-modal data if render_mode is vision
            if self.config.render_mode == 'vision':
                obs_str = proxy.to_text(self.config.image_placeholder)
                image_paths = []
                for t in proxy.turns:
                    if any('observe' in result.action_type for result in t.actions):
                        image, image_path = self._get_multi_modal_data(proxy.mgr, t.pos, t.ori)
                        images.append(image)
                        image_paths.append(image_path)
                assert images is not []
                exp_history['multi_modal_data'] = {self.config.image_placeholder: images}
                exp_history['multi_modal_data_paths'] = image_paths
            else:
                obs_str = proxy.to_text()
            exp_history['obs_str'] = obs_str
            # expose proxy manager so metrics are available via env.get_exp_summary()
            self.exploration_manager = proxy.mgr
            final_loc = (list(proxy.turns[-1].pos), list(proxy.turns[-1].ori))

        intial_prompt, self.observed_image_paths = self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            exp_history=exp_history,
        )
        return intial_prompt, final_loc

    def system_prompt(self) -> str:
        return self.prompter.system_prompt()

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset false belief phase state
        self.in_false_belief_phase = False
        self.target_object_name = None
        self.modified_room = None
        self.target_observed = False

        self.image_handler = ImageHandler(self.config.data_dir, seed, self.config.image_size)
        self.json_data = self.image_handler.json_data

        self.prompter = PromptManager(self.config, self.np_random, self.image_handler)
        # Generate initial room
        self.initial_room, self.agent = initialize_room_from_json(self.json_data)

        # self.prompter = PromptManager(self.config, self.np_random)
        # self.initial_room, self.agent = RoomGenerator.generate_multi_room(
        #     **self.config.get_room_config(),
        #     np_random=self.np_random,
        # )

        self.initial_agent = self.agent.copy()

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps

        # Initialize turn logs
        self.turn_logs = []
        self.current_turn_number = 0
        self.observed_image_paths = []
        # Set exploration phase
        self.is_exploration_phase = True

        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        self.exploration_manager = ExplorationManager(
            self.initial_room, self.agent,
            grid_size=(self.config.grid_size if hasattr(self.config, 'grid_size') else None),
        )
        self.history_manager = HistoryManager(
            self.config.get_observation_config(), self.config.get_model_config(),
            self.initial_room.to_dict(), self.agent.to_dict(),
            image_dir=self.image_handler.image_dir if hasattr(self, 'image_handler') else None,
            output_dir=self.config.kwargs['output_dir'],
            seed=seed,
            eval_override=False,
            all_override=self.config.kwargs.get('all_override', False),
        )
        # Persist the run seed so builders can reproduce evaluation tasks
        info = {}
        if self.history_manager:
            info['history'] = self.history_manager.get_responses()
        # For passive experiments, signal finish so rollout service skips stepping
        if self.config.exp_type == 'passive':
            info['finish'] = True

        obs, final_loc = self._generate_initial_observation()
        self.render_cache = obs

        # initialize message list (system + initial env feedback only; no evaluation question)
        self.history_manager.init_messages(self.prompter.system_prompt())
        self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
        self.history_manager.save_messages(final_loc)
        self.observed_image_paths = []
        return obs, info

    def _get_multi_modal_data(self, room: ExplorationManager, pos: np.ndarray, ori: np.ndarray):
        """Get multi-modal data (images) for current state."""
        assert self.config.render_mode == 'vision', "Cannot get multi-modal data in text mode"
        # Find position: which object is at same location as agent
        # Find position: which object is at same location as agent
        position_name = None if not np.allclose(room.init_pos, pos) else 'agent'
        if position_name is None:
            for obj in room.base_room.all_objects:
                if np.allclose(obj.pos, pos):
                    position_name = obj.name
                    break
        assert position_name is not None, "Agent position not found"
        
        direction = {(0, 1): 'north', (-1, 0): 'west', (0, -1): 'south', (1, 0): 'east'}[tuple(ori)]
        
        img = self.image_handler.get_image(position_name, direction)
        img_path = self.image_handler.get_image_path(position_name, direction)
        return img, img_path
            
    def step(self, llm_response: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        think_content, action, _ = parse_llm_response(
            llm_response, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )

        # Log turn at start with current state
        current_obs = self.render_cache

        # Execute action and get results
        obs, reward, done, info, exp_log, room_state, agent_state = self._execute_action(action)
        
        # Handle false belief phase transition
        if done and self.is_false_belief_exp and not self.in_false_belief_phase:
            # Save exploration turn log before transition
            self._save_turn_log(current_obs, llm_response, think_content, action, 
                               exp_log, room_state, agent_state, reward, info, 
                               is_exploration=True, is_last_exp=True)
            self.history_manager.append_assistant_message(llm_response)
            
            # Transition to false belief phase
            obs, info = self._transition_to_false_belief_phase()
            done = False
            
            # Save transition message
            self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
            self.history_manager.save_messages(None)
            self.observed_image_paths = []
            return obs, reward, done, info
        
        # Handle false belief phase termination - check target visibility
        if done and self.in_false_belief_phase:
            is_visible = self._check_target_visibility()
            self.target_observed = is_visible
            info['if_observed'] = self.target_observed
            info['success'] = self.target_observed
            reward = 1.0 if self.target_observed else 0.0
            
            # Save false belief result to history (similar to map_llm_responses for evaluation)
            fb_turn_log = {
                "is_exploration_phase": False,
                "evaluation_log": {
                    "task_type": "FalseBeliefExp",
                    "user_answer": llm_response,
                    "score": 1.0 if self.target_observed else 0.0,
                    "evaluation_info": {
                        "target_object": self.target_object_name,
                        "target_observed": self.target_observed,
                    },
                    "evaluation_data": {
                        "task_type": "false_belief_exp",
                        "target_object": self.target_object_name,
                    },
                },
                "assistant_raw_message": llm_response,
                "room_state": self.modified_room.to_dict() if self.modified_room else None,
                "agent_state": self.agent.to_dict(),
                "turn_number": self.current_turn_number,
            }
            self.history_manager.update_eval_turn_log(fb_turn_log)
            self.history_manager.save()
            
            done = True
        # Save turn log
        self._save_turn_log(current_obs, llm_response, think_content, action,
                           exp_log, room_state, agent_state, reward, info,
                           is_exploration=not self.in_false_belief_phase, is_last_exp=done)

        # Save messages
        self.history_manager.append_assistant_message(llm_response)
        self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
        self.history_manager.save_messages((agent_state.pos.tolist(), agent_state.ori.tolist()) if agent_state else None)
        self.observed_image_paths = []
        
        return obs, reward, done, info

    def _execute_action(self, action: str):
        """Execute action and return results. Shared by exploration and false belief phases."""
        obs_str, reward, done, info = "", -0.1, False, {'is_valid_action': True}
        obs: Dict[str, Any] = {}
        exp_log = None
        room_state = None
        agent_state = None
        self.remaining_exp_steps -= 1

        action_sequence = ActionSequence.parse(action)
        if self.remaining_exp_steps < 0:
            action_sequence = ActionSequence(motion_actions=[], final_action=ForcedTermAction())

        if not action or not action_sequence:
            obs_str += self.prompter.invalid_action_message() + "\n"
            info['is_valid_action'] = False
            reward += -0.5
        else:
            action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            obs_str += action_results_to_text(
                action_results,
                self.config.image_placeholder if self.config.render_mode == 'vision' else None,
            )
            exp_log = self.exploration_manager.turn_logs[-1]
            if exp_log:
                room_state, agent_state = exp_log.room_state, exp_log.agent_state
                exp_log.room_state = None
                exp_log.agent_state = None
            if action_sequence.final_action and action_sequence.final_action.is_term():
                done = True
                obs = {'obs_str': self.prompter.task_finished_message()}
            else:
                obs_str += "\n" + self.prompter.steps_left_message(self.remaining_exp_steps)
                if self.config.render_mode == 'vision':
                    image, image_path = self._get_multi_modal_data(
                        self.exploration_manager,
                        self.exploration_manager.agent.pos,
                        self.exploration_manager.agent.ori,
                    )
                    obs = {'multi_modal_data': {self.config.image_placeholder: [image]}, 'obs_str': obs_str}
                    self.observed_image_paths.append(image_path)

        if not obs:
            obs = {'obs_str': obs_str}

        if not done:
            obs['obs_str'] += '\n' + self.prompter.get_format_footer(True)

        self.render_cache = obs
        return obs, reward, done, info, exp_log, room_state, agent_state

    def _save_turn_log(self, current_obs, llm_response, think_content, action,
                       exp_log, room_state, agent_state, reward, info,
                       is_exploration=True, is_last_exp=False):
        """Save turn log. Shared by exploration and false belief phases."""
        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs['obs_str'],
            assistant_raw_message=llm_response,
            assistant_think_message=think_content,
            assistant_parsed_message=action,
            is_exploration_phase=is_exploration,
            is_last_exp=is_last_exp,
            exploration_log=exp_log,
            evaluation_log=None,
            room_state=room_state,
            agent_state=agent_state,
            message_images=self.observed_image_paths,
            info={"reward": reward, "is_done": is_last_exp, **info}
        )
        if is_exploration:
            if not self.history_manager.has_exploration(self.current_turn_number - 1):
                self.history_manager.update_turn_log(turn_log.to_dict())
                self.history_manager.save_exploration()
        else:
            self.history_manager.update_turn_log(turn_log.to_dict())
            self.history_manager.save_exploration()
        self.turn_logs.append(turn_log)

    def render(self):
        return self.render_cache

    def close(self):
        return

    # =================== False Belief Experiment Methods ===================
    
    def _transition_to_false_belief_phase(self):
        """Transition from exploration to false belief phase."""
        self.in_false_belief_phase = True
        
        # Modify room - move one object to a new location
        modifier = SingleObjectModifier()
        self.modified_room, self.target_object_name = modifier.modify(self.initial_room, self.np_random)
        
        # Switch exploration manager to use modified room
        self.exploration_manager.room = self.modified_room
        
        # Reset agent to INITIAL position and orientation
        self.agent.pos = self.agent.init_pos.copy()
        self.agent.ori = self.agent.init_ori.copy()
        if self.agent.init_room_id is not None:
            self.agent.room_id = self.agent.init_room_id
        
        # Reset step budget for False Belief phase
        self.remaining_exp_steps = self.config.max_exp_steps
        
        # Update history manager paths for false belief phase
        self.history_manager.exploration_path = os.path.join(self.history_manager.output_dir, "false_belief_turn_logs.json")
        self.history_manager.messages_path = os.path.join(self.history_manager.output_dir, "false_belief_messages.json")
        # Update room dict to modified one for saving
        self.history_manager.room_dict = self.modified_room.to_dict()
        
        # Generate False Belief prompt
        prompt = FALSE_BELIEF_INSTRUCTION.format(target_object=self.target_object_name)
        obs = {'obs_str': prompt}
        obs['obs_str'] += '\n' + self.prompter.get_format_footer(True)
        
        self.render_cache = obs
        info = {'phase': 'false_belief', 'target_object': self.target_object_name}
        
        return obs, info

    def _check_target_visibility(self) -> bool:
        """Check if target object has been observed during exploration."""
        if not self.target_object_name:
            return False
        return self.target_object_name in self.exploration_manager.observed_items

    # =================== Analysis ===================
    
    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY
    
    def get_env_summary(self) -> Dict[str, Any]:
        """Aggregate environment metrics from all turns."""

        return {
            'env_info': self._get_env_info(),
            'env_turn_logs': [turn_log.to_dict() for turn_log in self.turn_logs],
        }

    def _get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "initial_agent": self.initial_agent.to_dict(),
        }











if __name__ == "__main__":
    # Simple test cases for SpatialGym environment

    # TODO: add test cases
    pass

