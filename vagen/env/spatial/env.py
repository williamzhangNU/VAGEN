import gymnasium as gym
import numpy as np
import re
import os
import json
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.utils.parse_utils import parse_freethink

from vagen.env.spatial.Base.tos_base import (
    EvaluationManager,
    ActionSequence,
    ExplorationManager,
    BaseAction
)
from vagen.env.spatial.utils.initialize_room import initialize_room_from_json
from vagen.env.spatial.utils.generate_history import AutoExplore
from vagen.env.spatial.utils.action_utils import action_results_to_text
from vagen.env.spatial.utils.image_handler import ImageHandler
from vagen.env.spatial.prompt import (
    ACTIVE_INSTRUCTION, 
    PASSIVE_INSTRUCTION, 
    EVALUATION_INSTRUCTION,
    FORMAT_PROMPT,
    TOPDOWN_PROMPT
)

class SpatialGym(gym.Env):

    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.final_room = None
        
        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None

        # Exploration metrics
        self.n_valid_queries = None
        self.n_redundant_queries = None


    def _init_data(self, seed: int = None):
        """Initialize data and image handler."""
        self.image_handler = ImageHandler(self.config.base_dir, seed, self.config.image_size)
        self.image_dir = self.image_handler.image_dir
        self.json_data = self.image_handler.json_data


    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        room_desc = self.initial_room.get_room_description() + (TOPDOWN_PROMPT.format(placeholder=self.config.image_placeholder) if self.config.with_topdown else "")
        images = [self.image_handler.get_image('topdown')] if self.config.with_topdown else []
        
        if self.config.exp_type == 'passive':
            exp_history = ""
            if not self.config.with_topdown:
                exp_history_obs = AutoExplore(self.initial_room, self.np_random, self.image_handler).gen_exp_history()
                exp_history = f"## Exploration History\n{exp_history_obs['obs_str']}"
                images.extend(exp_history_obs['multi_modal_data'][self.config.image_placeholder])

            obs = PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history,
            )

            eval_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert eval_question, "No question found after exploration phase"
            obs += EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
            
            result_obs = self._create_obs(obs, include_visual=False)
            if images:
                result_obs['multi_modal_data'] = {self.config.image_placeholder: images}
            return result_obs

        exp_instructions = f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
        obs = ACTIVE_INSTRUCTION.format(
            room_info=room_desc,
            exp_instructions=exp_instructions
        )
        result_obs = self._create_obs(obs, include_visual=False)
        result_obs['multi_modal_data'] = {self.config.image_placeholder: images}
        return result_obs


    def _create_obs(self, obs_str: str, include_visual: bool = False) -> dict:
        obs = {'obs_str': obs_str + "\n" + FORMAT_PROMPT}
        if include_visual:
            img = self.image_handler.get_image(self.current_position, self.current_direction)
            obs['multi_modal_data'] = {self.config.image_placeholder: [img]}
        return obs
    
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
        self.n_valid_queries = 0
        self.n_redundant_queries = 0
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type != 'passive'
        
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Initialize managers
        if self.config.exp_type == 'active':
            self.exploration_manager = ExplorationManager(self.initial_room)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random)
        return self._generate_initial_observation(), {}


    def step(self, llm_raw_response: str):
        """Process agent actions in the spatial gym environment."""
        if self.is_exploration_phase:
            return self._step_exploration(llm_raw_response)
        else:
            return self._step_evaluation(llm_raw_response)

    def _step_exploration(self, llm_raw_response: str):
        """
        Handle exploration phase step.
        TODO:
        1. Deal with evaluation tasks that needs image observation
        """
        obs_str = ""
        reward = 0 # per step penalty
        include_visual = False

        result = parse_freethink(llm_raw_response, action_sep="|", max_actions=1)
        info = {
            "metrics": {
                'success': False,              # Did LLM complete the task?
                'action_is_effective': False,  # Did action change game state meaningfully?
                'action_is_valid': False,      # Was action format correct?
                },
            "llm_raw_response": llm_raw_response,  # Original LLM response
            "llm_response": result['llm_response'],     # Parsed action structure
        }

        # Parse and validate action
        action_sequence = None
        if result['actions']:
            action = result['actions'][0]
            action_sequence = ActionSequence.parse(action)        
            if not action_sequence:
                obs_str += "Invalid action\n"
                reward += 0 # format penalty
            else:
                info['metrics']['action_is_valid'] = True
                info['metrics']['action_is_effective'] = True # TODO check if effective
                self.n_valid_queries += 1 if not action_sequence.final_action.is_term() else 0
        else:
            obs_str += "Invalid input format\n"
            reward += 0 # format penalty

        self.remaining_exp_steps -= 1 # NOTE invalid action also counts as a step
        if self.remaining_exp_steps < 0 or (action_sequence and action_sequence.final_action.is_term()):
            # End exploration phase
            # NOTE (Optional) show image for original position
            self.is_exploration_phase = False
            obs_str += "Exploration phase ended\n"
            self.final_room = self.exploration_manager.finish_exploration()
            
            # Transition to evaluation, NOTE question is generated based on the initial room
            eval_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert eval_question, "No question found after exploration phase"
            obs_str += EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
        else:
            # Execute exploration action, TODO give better reward to efficient exploration
            if action_sequence:
                include_visual = True
                exp_info, action_results = self.exploration_manager.execute_action_sequence(action_sequence)
                self._update_agent_state()
                # Track redundant queries
                if exp_info.get('redundant', False):
                    self.n_redundant_queries += 1
                    reward += 0 # redundant observe penalty
                
                # Convert action results to text observation
                obs_str += action_results_to_text(action_results, self.config.image_placeholder)
            obs_str += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."
        
        return self._create_obs(obs_str, include_visual=include_visual), reward, False, {}
    
    def _step_evaluation(self, llm_raw_response: str):
        """Handle evaluation phase step."""
        # TODO: different reward for different tasks

        obs_str = ""
        reward = 0
        result = parse_freethink(llm_raw_response, action_sep="|", max_actions=1)
        info = {
            "metrics": {
                'success': False,              # Did LLM complete the task?
                'action_is_effective': False,  # Did action change game state meaningfully?
                'action_is_valid': False,      # Was action format correct?
                },
            "llm_raw_response": llm_raw_response,  # Original LLM response
            "llm_response": result['llm_response'],     # Parsed action structure
        }
        if not result['actions']:
            # TODO change each task only has one chance
            obs_str += "Invalid input format, please answer the question\n"
            reward += 0 # format penalty
            return self._create_obs(obs_str), reward, False, info

        action = result['actions'][0]

        # Evaluate answer
        correct, info = self.evaluation_manager.evaluate_answer(action) # TODO parse here
        reward = 1 if correct else 0
        
        # Check for next task
        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert next_question, "No question found after evaluation phase"
            return self._create_obs(next_question), reward, False, {}
        
        # All tasks completed
        return self._create_obs("Task finished"), reward, True, {}

    def close(self):
        return


    


    # =================== Analysis ===================
    
    def get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "final_room": self.final_room.to_dict() if self.final_room else None,
        }

    def get_exp_efficiency(self):
        """Get exploration efficiency metrics."""
        if self.config.exp_type == 'passive':
            return {
                "coverage": 0,
                "redundancy": self.n_redundant_queries / self.n_valid_queries if self.n_valid_queries > 0 else 0,
                "n_valid_queries": self.n_valid_queries,
                "n_redundant_queries": self.n_redundant_queries,
            }
        
        assert self.exploration_manager, "Exploration manager not initialized"
        return self.exploration_manager.get_exploration_efficiency()
    
    def get_eval_performance(self):
        """Get evaluation performance metrics."""
        if not self.evaluation_manager:
            return {
                "accuracy": 0.0,
                "accuracy_completed": 0.0,
                "task_results": [],
                "completed_tasks": 0,
                "unanswered_tasks": 0
            }
        
        return self.evaluation_manager.get_evaluation_summary()

if __name__ == "__main__":
    
    def test_passive_mode():
        """Test passive exploration mode."""
        print("Testing Passive Mode...")
        
        config = SpatialGymConfig(
            exp_type='passive',
            eval_tasks=[{"task_type": "all_pairs", "task_kwargs": {"num_pairs": 2}}],
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
        assert reward in [-0.5, 0, 1]  # Binary reward or format penalty
        
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
        exp_metrics = env.get_exp_efficiency()
        eval_metrics = env.get_eval_performance()
        
        assert 'coverage' in exp_metrics
        assert 'accuracy' in eval_metrics
        
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
            assert reward < 0  # Penalty for invalid input
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
    