import gymnasium as gym
import numpy as np
from typing import List, Dict, Any

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.Base.tos_base import (
    ExplorationManager,
    HistoryManager,
    BaseAction,
)
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_agent_proxy
from vagen.env.spatial.Base.tos_base.prompts import PromptManager
from vagen.env.spatial.Base.tos_base.utils.action_utils import action_results_to_text
from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
from vagen.env.spatial.Base.tos_base.utils.env_logger import EnvTurnLog
from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response
from vagen.env.spatial.Base.tos_base.utils.image_handler import ImageHandler
from vagen.env.spatial.Base.tos_base.actions.actions import ForcedTermAction, ActionSequence


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

    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        exp_history = {}
        images = []
        if self.config.exp_type == 'passive' and not self.config.prompt_config['topdown']:
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
                for t in proxy.turns:
                    if any('observe' in result.action_type for result in t.actions):
                        image, image_path = self._get_multi_modal_data(proxy.mgr, t.pos, t.ori)
                        images.append(image)
                        self.observed_image_paths.append(image_path)
                assert images is not []
                exp_history['multi_modal_data'] = {self.config.image_placeholder: images}
            else:
                obs_str = proxy.to_text()
            exp_history['obs_str'] = obs_str
            # expose proxy manager so metrics are available via env.get_exp_summary()
            self.exploration_manager = proxy.mgr

        return self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            eval_manager=None,
            exp_history=exp_history,
        )

    def system_prompt(self) -> str:
        return self.prompter.system_prompt()

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.image_handler = ImageHandler(self.config.data_dir, seed, self.config.image_size)
        self.json_data = self.image_handler.json_data

        self.prompter = PromptManager(self.config, self.np_random, self.image_handler)
        # Generate initial room
        self.initial_room, self.agent = initialize_room_from_json(self.json_data)
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
            image_dir=self.image_handler.image_dir,
            output_dir=self.config.kwargs['output_dir'],
            eval_override=False,
            all_override=self.config.kwargs.get('all_override', False),
            task_type=None,
        )
        # Persist the run seed so builders can reproduce evaluation tasks
        self.history_manager.set_run_seed(seed)
        self.history_manager.save_state()
        info = {}
        if self.history_manager:
            info['history'] = self.history_manager.get_responses()
        # For passive experiments, signal finish so rollout service skips stepping
        if self.config.exp_type == 'passive':
            info['finish'] = True

        obs = self._generate_initial_observation()
        self.render_cache = obs

        # initialize message list (system + initial env feedback only; no evaluation question)
        self.history_manager.init_messages(self.prompter.system_prompt())
        self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
        self.history_manager.save_messages()
        return obs, info

    def _get_multi_modal_data(self, room: ExplorationManager, pos: np.ndarray, ori: np.ndarray):
        """Get multi-modal data (images) for current state."""
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
        """Process agent actions in the spatial gym environment (exploration only)."""
        self.current_turn_number += 1
        think_content, action, _ = parse_llm_response(
            llm_response, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )

        # Log turn at start with current state
        current_obs = self.render_cache

        # Exploration step (merged)
        obs_str, reward, done, info = "", -0.1, False, {'is_valid_action': True}
        obs: Dict[str, Any] = {}
        exp_log = None
        room_state = None
        agent_state = None
        self.remaining_exp_steps -= 1

        action_sequence = ActionSequence.parse(action)
        if self.remaining_exp_steps < 0:
            action_sequence = ActionSequence(motion_actions=[], final_action=ForcedTermAction())

        if not action:
            obs_str += self.prompter.invalid_action_message() + "\n"
            info['is_valid_action'] = False
            reward += -0.5
        elif not action_sequence:
            obs_str += self.prompter.invalid_format_message() + "\n"
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

        # Add footer during exploration, skip when finished
        if not done:
            obs['obs_str'] += '\n' + self.prompter.get_format_footer(True)

        self.render_cache = obs

        # Save turn log (exploration only)
        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs['obs_str'],
            assistant_raw_message=llm_response,
            assistant_think_message=think_content,
            assistant_parsed_message=action,
            is_exploration_phase=True,
            is_last_exp=done,
            exploration_log=exp_log,
            evaluation_log=None,
            room_state=room_state,
            agent_state=agent_state,
            message_images=self.observed_image_paths,
            info={"reward": reward, "is_done": done, **info}
        )
        if not self.history_manager.has_exploration(self.current_turn_number - 1):
            self.history_manager.update_turn_log(turn_log.to_dict())
            self.history_manager.save_exploration()

        # Save message list
        self.history_manager.append_assistant_message(llm_response)
        self.history_manager.append_env_feedback(obs.get('obs_str', ''), self.observed_image_paths or [])
        self.history_manager.save_messages()

        self.observed_image_paths = []
        self.turn_logs.append(turn_log)
        return obs, reward, done, info

    def render(self):
        return self.render_cache

    def close(self):
        return


    


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

