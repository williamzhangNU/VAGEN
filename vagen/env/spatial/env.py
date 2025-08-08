import gymnasium as gym
import numpy as np
import re
import os
import json
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.utils.parse_utils import parse_freethink

from vagen.env.spatial.Base.tos_base import (
    EvaluationManager,
    ActionSequence,
    ExplorationManager,
    BaseAction,
    Room,
    ExplorationTurnLog,
    EvaluationTurnLog,
    CognitiveMapManager,
    CognitiveMapTurnLog
)
from vagen.env.spatial.utils.initialize_room import initialize_room_from_json
from vagen.env.spatial.utils.action_utils import action_results_to_text
from vagen.env.spatial.utils.image_handler import ImageHandler
from vagen.env.spatial.prompts import Prompter


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
            "info": self.info
        }

class SpatialGym(gym.Env):

    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.prompter = None

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.final_room = None
        
        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None
        self.cognitive_map_manager = None

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = None
        self.current_turn_number = None
        


    def _init_data(self, seed: int = None):
        """Initialize data and image handler."""
        self.image_handler = ImageHandler(self.config.base_dir, seed, self.config.image_size)
        self.image_dir = self.image_handler.image_dir
        self.json_data = self.image_handler.json_data


    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        return self.prompter.get_initial_observation_prompt(
            room=self.initial_room, 
            eval_manager=self.evaluation_manager,
            cogmap_manager=self.cognitive_map_manager
        )


    def _get_multi_modal_data(self) -> List[Image.Image]:
        img = self.image_handler.get_image(self.current_position, self.current_direction)
        return [img]
    
    def _update_agent_state(self) -> Tuple[str, str]:
        """Update current position and direction of agent."""

        assert self.exploration_manager is not None, "Exploration manager is not set"
        # TODO: check for evaluation state
            
        room = self.exploration_manager.exploration_room
        agent = room.agent
        assert np.array_equal(agent.pos, np.array([0, 0])), f"Agent position is not (0,0), got {agent.pos}"
        assert np.array_equal(agent.ori, np.array([0, 1])), f"Agent orientation is not (0,1), got {agent.ori}"
        
        # 1. Find position: which object is at same location as agent (0,0)
        position_name = None if not np.allclose(room.initial_pos.pos, agent.pos) else 'agent'
        if position_name is None:
            for obj in room.objects:
                if np.allclose(obj.pos, agent.pos):
                    position_name = obj.name
                    break
        assert position_name is not None, "Agent position not found"
                
        # 2. Find direction: initial_pos always faces north
        initial_ori = tuple(room.initial_pos.ori)
        
        # Map orientation difference to direction
        direction_name = {(0, 1): 'north', (1, 0): 'west', (0, -1): 'south', (-1, 0): 'east'}[initial_ori]
        
        self.current_position = position_name
        self.current_direction = direction_name





    def system_prompt(self) -> str:
        return "You are an AI assistant that answers visual questions based on images."

    def reset(self, seed: int = None):
        super().reset(seed=seed)

        self._init_data(seed=seed)
        self.current_position = 'agent'
        self.current_direction = 'north'

        # Generate initial room
        self.initial_room = initialize_room_from_json(self.json_data)

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps

        # Initialize turn logs
        self.turn_logs = []
        self.current_turn_number = 0
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type != 'passive'
        
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Initialize managers
        self.exploration_manager = ExplorationManager(self.initial_room) if self.config.exp_type == 'active' else None
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room) if len(self.config.eval_tasks) > 0 else None
        self.cognitive_map_manager = CognitiveMapManager(self.initial_room) if self.config.exp_type == 'active' else None
        self.prompter = Prompter(self.config, self.image_handler, self.np_random)

        obs = self._generate_initial_observation()
        self.render_cache = obs
        return obs, {}


    def step(self, llm_raw_response: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        exp_log, eval_log, cogmap_log = None, None, None
        result = parse_freethink(llm_raw_response, action_sep="|", max_actions=1)
        room_state_last_turn = next((turn_log.room_state for turn_log in self.turn_logs[::-1] if turn_log.room_state), self.initial_room)
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
                self.cognitive_map_manager.evaluate_cognitive_map(result['think_content'], room_state_last_turn)
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
            obs['obs_str'] += '\n' + self.prompter.COGMAP_EXP_REQUIRED_INSTRUCTION if self.config.prompt_config['cogmap'] else ""
        else:
            obs['obs_str'] += '\n' + self.prompter.COGMAP_EVAL_REQUIRED_INSTRUCTION if self.config.prompt_config['cogmap'] else ""
        obs['obs_str'] += "\n" + self.prompter.FORMAT_PROMPT
        self.render_cache = obs

        # Get room state from turn logs
        room_state = room_state_last_turn
        if exp_log and exp_log.room_state:
            room_state = exp_log.room_state
        elif eval_log and eval_log.room_state:
            room_state = eval_log.room_state

        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs['obs_str'],
            assistant_raw_message=llm_raw_response,
            assistant_think_message=result['think_content'],
            assistant_parsed_message=result['action_content'],
            is_exploration_phase=self.is_exploration_phase,
            exploration_log=exp_log,
            evaluation_log=eval_log,
            cogmap_log=cogmap_log,
            room_state=room_state,
            info={"reward": reward, "is_done": done, **{k: v for k, v in info.items() if 'response' not in k}}
        )
        self.turn_logs.append(turn_log)
        return obs, reward, done, info

    def _step_exploration(self, result: Dict[str, Any], info: Dict[str, Any]):
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
            info['metrics']['action_is_effective'] = False  # TODO check actual effect
        else:
            # execute action
            exp_info, action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            reward += -1 if exp_info.get('redundant', False) else 0 # redundant observe penalty
            obs_str += action_results_to_text(action_results, self.config.image_placeholder)
            exp_log = self.exploration_manager.turn_logs[-1]

            include_visual = True
            self._update_agent_state()

        # End exploration phase
        if self.remaining_exp_steps < 0 or (action_sequence and action_sequence.final_action.is_term()):
            self.is_exploration_phase = False
            obs_str += "Exploration phase ended\n"
            self.final_room = self.exploration_manager.finish_exploration()
            obs_str += self.prompter.get_evaluation_prompt(self.evaluation_manager)
        else:
            obs_str += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."

        obs = {'multi_modal_data': self._get_multi_modal_data()} if include_visual else {}
        return {**obs, 'obs_str': obs_str}, reward, False, info, exp_log

    def _step_evaluation(self, result: Dict[str, Any], info: Dict[str, Any]):
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
            "final_room": self.final_room.to_dict() if self.final_room else None,
        }











if __name__ == "__main__":
    
    def test_passive_mode():
        """Test passive exploration mode."""
        print("Testing Passive Mode...")
        
        config = SpatialGymConfig(
            exp_type='passive',
            eval_tasks=[{"task_type": "dir", "task_kwargs": {}}],
            max_exp_steps=1
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        print(obs)
        
        # Should start in evaluation phase
        assert not env.is_exploration_phase
        assert 'Exploration History' in obs['obs_str']
        assert 'multi_modal_data' in obs  # Visual data from exploration history
        
        # Test evaluation step
        answer = "left"
        obs, reward, done, info = env.step(answer)
        assert reward in [0, 1]  # Binary reward
        
        print("✓ Passive mode test passed\n")

    def test_active_mode():
        """Test active exploration mode."""
        print("Testing Active Mode...")
        
        config = SpatialGymConfig(
            exp_type='active',
            eval_tasks=[{"task_type": "dir", "task_kwargs": {}}],
            max_exp_steps=5
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        
        # Should start in exploration phase
        assert env.is_exploration_phase
        assert 'multi_modal_data' in obs  # Visual in initial obs
        
        # Test valid action with visual
        obs, reward, done, info = env.step("Movement: [Rotate(90)]\nFinal: Observe()")
        assert 'multi_modal_data' in obs  # Visual for valid action
        
        # Test invalid action without visual
        obs, reward, done, info = env.step("Invalid action")
        assert 'multi_modal_data' not in obs  # No visual for invalid action
        
        print("✓ Active mode test passed\n")

    def test_visual_conditions():
        """Test visual observation conditions."""
        print("Testing Visual Conditions...")
        
        config = SpatialGymConfig(exp_type='active', max_exp_steps=3)
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        
        # Valid action in exploration → has visual
        obs, _, _, _ = env.step("Movement: []\nFinal: Observe()")
        assert env.is_exploration_phase
        assert 'multi_modal_data' in obs
        
        # Invalid action in exploration → no visual
        obs, _, _, _ = env.step("Invalid format")
        assert 'multi_modal_data' not in obs
        
        # End exploration
        obs, _, _, _ = env.step("Movement: []\nFinal: Term()")
        assert not env.is_exploration_phase
        assert 'multi_modal_data' not in obs  # No visual in evaluation
        
        print("✓ Visual conditions test passed\n")

    def test_phase_transitions():
        """Test exploration to evaluation transition."""
        print("Testing Phase Transitions...")
        
        config = SpatialGymConfig(exp_type='active', max_exp_steps=2)
        env = SpatialGym(config)
        env.reset(seed=42)
        
        assert env.is_exploration_phase
        assert env.remaining_exp_steps == 2
        
        # Use one step
        env.step("Movement: []\nFinal: Observe()")
        assert env.is_exploration_phase
        assert env.remaining_exp_steps == 1
        
        # Terminate early
        obs, _, _, _ = env.step("Movement: []\nFinal: Term()")
        assert not env.is_exploration_phase
        assert "Exploration phase ended" in obs['obs_str']
        
        print("✓ Phase transitions test passed\n")

    def test_configuration():
        """Test different configurations."""
        print("Testing Configuration...")
        
        # Test active vs passive
        active_config = SpatialGymConfig(exp_type='active')
        passive_config = SpatialGymConfig(exp_type='passive')
        
        active_env = SpatialGym(active_config)
        passive_env = SpatialGym(passive_config)
        
        active_obs, _ = active_env.reset(seed=42)
        passive_obs, _ = passive_env.reset(seed=42)
        
        # Active starts with exploration
        assert active_env.is_exploration_phase
        assert 'Action Instructions' in active_obs['obs_str']
        
        # Passive starts with evaluation
        assert not passive_env.is_exploration_phase
        assert 'Exploration History' in passive_obs['obs_str']
        
        print("✓ Configuration test passed\n")

    def test_metrics():
        """Test exploration and evaluation metrics."""
        print("Testing Metrics...")
        
        config = SpatialGymConfig(exp_type='active', max_exp_steps=3)
        env = SpatialGym(config)
        env.reset(seed=42)
        
        # Initial metrics
        exp_metrics = env.get_exp_summary()
        eval_metrics = env.get_eval_summary()
        
        assert isinstance(exp_metrics, dict)
        assert isinstance(eval_metrics, dict)
        
        # After some exploration
        env.step("Movement: []\nFinal: Observe()")
        env.step("Movement: []\nFinal: Term()")
        
        # Should transition to evaluation
        assert not env.is_exploration_phase
        
        print("✓ Metrics test passed\n")

    def test_error_handling():
        """Test error handling for invalid inputs."""
        print("Testing Error Handling...")
        
        config = SpatialGymConfig(exp_type='active')
        env = SpatialGym(config)
        env.reset(seed=42)
        
        # Test various invalid inputs
        invalid_inputs = [
            "",  # Empty
            "Just text",  # No action format
            "Movement: [InvalidAction()]\nFinal: Observe()",  # Invalid action
        ]
        
        for invalid_input in invalid_inputs:
            obs, reward, done, info = env.step(invalid_input)
            assert reward == 0  # No reward for invalid input
            assert 'Invalid' in obs['obs_str']
        
        print("✓ Error handling test passed\n")

    # Run all tests
    print("=" * 50)
    print("SPATIAL GYM TESTS")
    print("=" * 50)
    
    try:
        test_passive_mode()
        # test_active_mode()
        # test_visual_conditions()
        # test_phase_transitions()
        # test_configuration()
        # test_metrics()
        # test_error_handling()
        
        print("=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    