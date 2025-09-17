import gymnasium as gym
import numpy as np
from typing import List, Dict, Any

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.Base.tos_base import (
    EvaluationManager,
    ActionSequence,
    ExplorationManager,
    HistoryManager,
    RoomGenerator,
    BaseAction
)
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_agent_proxy
from vagen.env.spatial.Base.tos_base.prompts import Prompter
from vagen.env.spatial.Base.tos_base.utils.action_utils import action_results_to_text
from vagen.env.spatial.utils.initialize_room import initialize_room_from_json
from vagen.env.spatial.Base.tos_base.utils.env_logger import EnvTurnLog
from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response
from vagen.env.spatial.utils.image_handler import ImageHandler
from vagen.env.spatial.Base.tos_base.actions.actions import ForcedTermAction, ActionSequence


class SpatialGym(gym.Env):
    """
    Spatial Gym Environment with exploration and evaluation phases.

    This environment uses an EvaluationManager to handle all evaluation tasks,
    separating evaluation logic from the main environment logic.
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.prompter: Prompter = None

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.initial_agent = None

        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None
        self.cognitive_map_manager = None
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
                self.config.proxy_agent_config["type"],
                self.initial_room,
                self.agent,
                delegate=self.config.proxy_agent_config.get("delegate"),
                observer_delegate=self.config.proxy_agent_config.get("observer_delegate"), # TODO change name
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
            eval_manager=self.evaluation_manager,
            exp_history=exp_history,
        )

    def system_prompt(self) -> str:
        return "You are an AI assistant that answers visual questions based on images."

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.image_handler = ImageHandler(self.config.base_dir, seed, self.config.image_size)
        self.json_data = self.image_handler.json_data

        self.prompter = Prompter(self.config, self.np_random, self.image_handler)
        # Generate initial room
        # self.initial_room, self.agent = RoomGenerator.generate_room(
        #     **self.config.get_room_config(),
        #     np_random=self.np_random,
        # )
        self.initial_room, self.agent = initialize_room_from_json(self.json_data)
        self.initial_agent = self.agent.copy()

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps

        # Initialize turn logs
        self.turn_logs = []
        self.current_turn_number = 0
        self.observed_image_paths = []
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type == 'active'

        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        self.exploration_manager = ExplorationManager(
            self.initial_room, self.agent,
            enable_information_gain=getattr(self.config, 'calculate_information_gain', False),
            grid_size=(self.config.grid_size if hasattr(self.config, 'grid_size') else None),
            enable_exploration_quality=getattr(self.config, 'calculate_exploration_quality', False)
        )
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room, self.agent) if len(self.config.eval_tasks) > 0 else None
        self.history_manager = HistoryManager(
            self.config.get_observation_config(),self.config.get_model_config(), 
            self.initial_room.to_dict(), self.agent.to_dict(), 
            override=self.config.kwargs['override'], output_dir= self.config.kwargs['output_dir']
        )
        info = {}
        if self.history_manager and self.history_manager.is_history_exist():
            info['history'] = self.history_manager.get_responses()
            
        obs = self._generate_initial_observation()
        self.render_cache = obs
        return obs, info

    def _step_exploration(self, action: str):
        """
        Handle exploration phase step with parsed result and shared info.
        """
        obs_str = ""
        reward = -0.1 
        self.remaining_exp_steps -= 1
        exp_log = None
        obs={}
        info = {'is_valid_action': True}
        action_sequence = ActionSequence.parse(action)
        if self.remaining_exp_steps < 0:
            action_sequence = ActionSequence(motion_actions=[], final_action=ForcedTermAction())
        if not action_sequence:
            obs_str += "Invalid action\n"
            reward += -0.5 # invalid action penalty
            info['is_valid_action'] = False
        else:
            # execute action
            _ , action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            obs_str += action_results_to_text(action_results, self.config.image_placeholder if self.config.render_mode == 'vision' else None)
            exp_log = self.exploration_manager.turn_logs[-1]
            if action_sequence.final_action and action_sequence.final_action.is_term():
                self.is_exploration_phase = False
                obs_str += self.prompter.get_evaluation_prompt(self.evaluation_manager)
            else:
                obs_str += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."
                # Only get multi-modal data if render_mode is vision
                if self.config.render_mode == 'vision':
                    image, image_path = self._get_multi_modal_data(self.exploration_manager, self.exploration_manager.agent.pos, self.exploration_manager.agent.ori)
                    obs = {'multi_modal_data': {self.config.image_placeholder: [image]}}
                    self.observed_image_paths.append(image_path)

        return {**obs, 'obs_str': obs_str}, reward, False, info, exp_log

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
            

    def _step_evaluation(self, action: str):
        """Handle evaluation phase step with parsed result and shared info."""

        correct, _ = self.evaluation_manager.evaluate_answer(action)
        eval_log = self.evaluation_manager.turn_logs[-1]
        reward = 1 if correct else 0

        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question()
            assert next_question, "No question found after evaluation phase"
            return {'obs_str': next_question}, reward, False, {}, eval_log

        return {'obs_str': "Task finished"}, reward, True, {}, eval_log

    def step(self, llm_response: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        exp_log, eval_log = None, None
        think_content, action, parsed_ok = parse_llm_response(
            llm_response, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )
        room_state = None
        agent_state = None

        # Log turn at start with current state
        current_obs = self.render_cache
        is_exploration_phase = self.is_exploration_phase # so termiante action is included in exploration log
        # step the environment
        if parsed_ok:
            if self.is_exploration_phase:
                obs, reward, done, step_info, exp_log = self._step_exploration(action)
                if exp_log:
                    room_state, agent_state = exp_log.room_state, exp_log.agent_state
                    exp_log.room_state = None
                    exp_log.agent_state = None
            else:
                obs, reward, done, step_info, eval_log = self._step_evaluation(action)
                room_state, agent_state = eval_log.room_state, eval_log.agent_state
                eval_log.room_state = None
                eval_log.agent_state = None
        else:
            reward, obs, done, step_info = -0.5, {'obs_str': "Invalid input format.\n"}, False, {}


        obs['obs_str'] += '\n' + self.prompter.FORMAT_PROMPT
        self.render_cache = obs

        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs['obs_str'],
            assistant_raw_message=llm_response,
            assistant_think_message=think_content,
            assistant_parsed_message=action,
            is_exploration_phase=is_exploration_phase,
            observed_items=list(self.exploration_manager.observed_items),
            exploration_log=exp_log,
            evaluation_log=eval_log,
            room_state=room_state,
            agent_state=agent_state,
            message_images=self.observed_image_paths,
            info={"reward": reward, "is_done": done, **step_info}
        )
        if is_exploration_phase:
            if not self.history_manager.is_history_exist():
                self.history_manager.update_turn_log(turn_log.to_dict())
        else:
            self.history_manager.update_turn_log(turn_log.to_dict())
            self.history_manager.save()
        self.observed_image_paths = []
        self.turn_logs.append(turn_log)
        return obs, reward, done, step_info

    def render(self):
        return self.render_cache

    def close(self):
        return


    


    # =================== Analysis ===================
    
    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY
    
    def get_eval_summary(self):
        """Get evaluation performance metrics."""
        return self.evaluation_manager.get_eval_summary() if self.evaluation_manager else EvaluationManager.DEFAULT_EVAL_SUMMARY.copy()
    
    def get_env_summary(self) -> Dict[str, Any]:
        """Aggregate environment metrics from all turns."""

        return {
            'env_info': self._get_env_info(),
            'env_turn_logs': [turn_log.to_dict() for turn_log in self.turn_logs],
            # 'summary': {
            #     'total_turns': len(self.turn_logs),
            #     'exp_summary': self.get_exp_summary(),
            #     'eval_summary': self.get_eval_summary(),
            #     'cogmap_summary': {},
            # }
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

