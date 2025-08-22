import gymnasium as gym
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.Base.tos_base import (
    EvaluationManager,
    EvaluationTurnLog,
    Room,
    ActionSequence,
    ExplorationManager,
    ExplorationTurnLog,
    CognitiveMapManager,
    CognitiveMapTurnLog,
    RoomGenerator,
    BaseAction,
    ObserveAction,
    Agent,
)
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_agent_proxy
from vagen.env.spatial.prompts import Prompter
from vagen.env.spatial.Base.tos_base.utils.action_utils import action_results_to_text
from vagen.env.spatial.utils.initialize_room import initialize_room_from_json
from vagen.env.utils.parse_utils import parse_freethink
from vagen.env.spatial.utils.image_handler import ImageHandler


@dataclass
class EnvTurnLog:
    """Log data for a single environment turn."""
    turn_number: int
    user_message: str = ""  # Environment observation
    assistant_raw_message: str = ""  # Raw assistant input
    assistant_think_message: str = ""  # Think part of assistant message
    assistant_parsed_message: str = ""  # Parsed assistant action
    is_exploration_phase: bool = False
    exploration_log: Optional["ExplorationTurnLog"] = None
    evaluation_log: Optional["EvaluationTurnLog"] = None
    cogmap_log: Optional["CognitiveMapTurnLog"] = None
    room_state: Optional["Room"] = None
    agent_state: Optional["Agent"] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "assistant_raw_message": self.assistant_raw_message,
            "assistant_think_message": self.assistant_think_message,
            "assistant_parsed_message": self.assistant_parsed_message,
            "is_exploration_phase": self.is_exploration_phase,
            "exploration_log": self.exploration_log.to_dict() if self.exploration_log else {},
            "evaluation_log": self.evaluation_log.to_dict() if self.evaluation_log else {},
            "cogmap_log": self.cogmap_log.to_dict() if self.cogmap_log else {},
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "info": self.info
        }

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

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = None
        self.current_turn_number = None

    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        exp_history_data = {}
        if self.config.exp_type == 'passive' and not self.config.prompt_config['topdown']:
            strategy = getattr(self.config, 'passive_agent_strategy', 'oracle')
            proxy = get_agent_proxy(strategy, self.initial_room, self.agent)
            proxy.run()

            # Get exploration history with images
            exp_history_data = {} 
            obs_str = proxy.to_text(self.config.image_placeholder)
            images = []
            for t in proxy.turns:
                if any(result.action_type == 'observe' for result in t.actions):
                    images.append(self._get_multi_modal_data(proxy.mgr, t.pos, t.ori))
            exp_history_data['obs_str'] = obs_str
            exp_history_data['multi_modal_data'] = {self.config.image_placeholder: images}
            # expose proxy manager so metrics are available via env.get_exp_summary()
            self.exploration_manager = proxy.mgr

        return self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            eval_manager=self.evaluation_manager,
            cogmap_manager=self.cognitive_map_manager,
            exp_history=exp_history_data,
        )

    def system_prompt(self) -> str:
        return "You are an AI assistant that answers visual questions based on images."

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.image_handler = ImageHandler(self.config.base_dir, seed, self.config.image_size)
        self.image_dir = self.image_handler.image_dir
        self.json_data = self.image_handler.json_data
        self.prompter = Prompter(self.config, self.image_handler, self.np_random)
        # Generate initial room
        # self.initial_room, self.agent = RoomGenerator.generate_room(
        #     **self.config.get_room_config(),
        #     np_random=self.np_random,
        # )
        mask = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  0, -1],
            [-1,  0,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  0, -1],
            [-1,  0,  1,  1,  1,  1,  1, 101, 2,  2,  2,  2,  2,  2,  0, -1],
            [-1,  0,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  0, -1],
            [-1,  0,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  0, -1],
            [-1,  0,  0,  0, 100, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  3,  3,  3,  3,  3,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  3,  3,  3,  3,  3,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  3,  3,  3,  3,  3,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  3,  3,  3,  3,  3,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  3,  3,  3,  3,  3,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        self.initial_room, self.agent = initialize_room_from_json(self.json_data, mask)
        self.initial_agent = self.agent.copy()

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps

        # Initialize turn logs
        self.turn_logs = []
        self.current_turn_number = 0

        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type == 'active'

        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        # Set observation mode: default to 'full' (dir+degree+distance), allow override via config
        mode = getattr(self.config, 'observation_mode', 'full')
        ObserveAction.MODE = 'full' if mode == 'full' else 'dir'

        # Initialize managers
        # create decoupled agent with initial pose (0,0,N) and store init pose
        # always create exploration manager (also used to generate passive history)
        self.exploration_manager = ExplorationManager(self.initial_room, self.agent)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room, self.agent) if len(self.config.eval_tasks) > 0 else None
        self.cognitive_map_manager = CognitiveMapManager() if self.config.prompt_config['cogmap'] else None

        # Generate initial observation
        obs = self._generate_initial_observation()
        self.render_cache = obs
        return obs, {}

    def _step_exploration(self, result: dict, info: dict):
        """
        Handle exploration phase step with parsed result and shared info.
        """
        obs_str = ""
        reward = -0.1 # per step penalty
        include_visual = False
        self.remaining_exp_steps -= 1
        exp_log = None

        action = result['actions'][0]
        action_sequence = ActionSequence.parse(action)
        if not action_sequence:
            obs_str += "Invalid action\n"
            reward += -0.5 # invalid action penalty
            info['metrics']['action_is_valid'] = False
            info['metrics']['action_is_effective'] = False
        else:
            # execute action
            exp_info, action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            reward += -1 if exp_info.get('redundant', False) else 0 # redundant observe penalty
            obs_str += action_results_to_text(action_results, self.config.image_placeholder)
            exp_log = self.exploration_manager.turn_logs[-1]
            include_visual = True

        # End exploration phase
        if self.remaining_exp_steps < 0 or (action_sequence and action_sequence.final_action.is_term()):
            self.is_exploration_phase = False
            obs_str += "Exploration phase ended\n"
            obs_str += self.prompter.get_evaluation_prompt(self.evaluation_manager)
        else:
            obs_str += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."

        obs = {'multi_modal_data': self._get_multi_modal_data(self.exploration_manager, self.exploration_manager.agent.pos, self.exploration_manager.agent.ori)} if include_visual else {}
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
        
        direction = {(0, 1): 'north', (1, 0): 'west', (0, -1): 'south', (-1, 0): 'east'}[tuple(ori)]
        
        img = self.image_handler.get_image(position_name, direction)
        return img
            

    def _step_evaluation(self, result: dict, info: dict):
        """Handle evaluation phase step with parsed result and shared info."""

        action = result['actions'][0]

        correct, _ = self.evaluation_manager.evaluate_answer(action)
        eval_log = self.evaluation_manager.turn_logs[-1]
        reward = 1 if correct else 0
        info['metrics']['success'] = correct

        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question()
            assert next_question, "No question found after evaluation phase"
            return {'obs_str': next_question}, reward, False, info, eval_log

        return {'obs_str': "Task finished"}, reward, True, info, eval_log

    def step(self, llm_raw_response: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        exp_log, eval_log, cogmap_log = None, None, None
        result = parse_freethink(llm_raw_response, action_sep="|", max_actions=1)
        room_state_last_turn = next((turn_log.room_state for turn_log in self.turn_logs[::-1] if turn_log.room_state), self.initial_room)
        agent_state_last_turn = next((turn_log.agent_state for turn_log in self.turn_logs[::-1] if turn_log.agent_state), self.agent)

        current_obs = self.render_cache

        info = {
            "metrics": {
                'success': bool(result['actions']),
                'action_is_effective': bool(result['actions']),
                'action_is_valid': bool(result['actions']),
            },
            "llm_raw_response": llm_raw_response,
            "llm_response": result['llm_response'],
        }

        # step the environment
        if result['actions'] and result['think_content']:
            # think
            if self.cognitive_map_manager:
                self.cognitive_map_manager.evaluate_cognitive_map(result['think_content'], room_state_last_turn, agent_state_last_turn)
                cogmap_log = self.cognitive_map_manager.turn_logs[-1]
            # action
            if self.is_exploration_phase:
                obs, reward, done, _, exp_log = self._step_exploration(result, info)
            else:
                obs, reward, done, _, eval_log = self._step_evaluation(result, info)
        else:
            obs = {'obs_str': 'Invalid input format'}
            reward = -0.5 # invalid input penalty
            done = False

        # post-process the observation
        if self.is_exploration_phase:
            obs['obs_str'] += '\n' + self.prompter.COGMAP_EXP_REQUIRED_INSTRUCTION if self.config.prompt_config['cogmap'] else ''
        else:
            obs['obs_str'] += '\n' + self.prompter.COGMAP_EVAL_REQUIRED_INSTRUCTION if self.config.prompt_config['cogmap'] else ''
        obs['obs_str'] += '\n' + self.prompter.FORMAT_PROMPT
        self.render_cache = obs

        # Get room state from turn logs
        room_state = room_state_last_turn
        agent_state = agent_state_last_turn
        if exp_log and exp_log.room_state:
            room_state = exp_log.room_state
            agent_state = exp_log.agent_state or agent_state
        elif eval_log and eval_log.room_state:
            room_state = eval_log.room_state
            agent_state = eval_log.agent_state or agent_state

        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs['obs_str'],
            assistant_raw_message=llm_raw_response,
            assistant_think_message=result['think_content'],
            assistant_parsed_message=result['action_content'],
            is_exploration_phase=self.is_exploration_phase,
            room_state=room_state,
            agent_state=agent_state,
            exploration_log=exp_log,
            evaluation_log=eval_log,
            cogmap_log=cogmap_log,
            info={"reward": reward, "is_done": done, **{k: v for k, v in info.items() if 'response' not in k}}
        )
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
    
    def get_eval_summary(self):
        """Get evaluation performance metrics."""
        return self.evaluation_manager.get_eval_summary() if self.evaluation_manager else EvaluationManager.DEFAULT_EVAL_SUMMARY.copy()
    
    def get_cogmap_summary(self):
        """Get cognitive map summary."""
        return self.cognitive_map_manager.get_cogmap_summary() if self.cognitive_map_manager else CognitiveMapManager.DEFAULT_COGMAP_SUMMARY.copy()
    
    def get_env_summary(self) -> Dict[str, Any]:
        """Aggregate environment metrics from all turns."""

        return {
            'env_info': self._get_env_info(),
            'env_turn_logs': [turn_log.to_dict() for turn_log in self.turn_logs],
            'summary': {
                'total_turns': len(self.turn_logs),
                'exp_summary': self.get_exp_summary(),
                'eval_summary': self.get_eval_summary(),
                'cogmap_summary': self.get_cogmap_summary()
            }
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

