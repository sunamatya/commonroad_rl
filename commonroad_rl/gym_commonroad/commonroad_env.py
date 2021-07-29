import warnings

from commonroad_rl.gym_commonroad.reward import reward_constructor
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.termination import Termination

"""
Module for the CommonRoad Gym environment
"""
import os
import gym
import glob
import yaml
import pickle
import random
import logging
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple, Union
from collections import defaultdict

# import from commonroad-drivability-checker
from commonroad_dc.collision.visualization import draw_dispatch as crdc_draw_dispatch

# import from commonroad-io
from commonroad.scenario.scenario import ScenarioID
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.draw_dispatch_cr import draw_object

# import from commonroad-rl
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.gym_commonroad.action import ContinuousAction, DiscretePMJerkAction
from commonroad_rl.gym_commonroad.reward import reward_constructor
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.termination import Termination

matplotlib.use("AGG")
LOGGER = logging.getLogger(__name__)


class CommonroadEnv(gym.Env):
    """
    Description:
        This environment simulates the ego vehicle in a traffic scenario using commonroad environment. The task of
        the ego vehicle is to reach the predefined goal without going off-road, collision with other vehicles, and
        finish the task in specific time frame. Please consult `commonroad_rl/gym_commonroad/README.md` for details.
    """

    metadata = {"render.modes": ["human"]}

    # For the current configuration check the ./configs.yaml file
    def __init__(
            self,
            meta_scenario_path=PATH_PARAMS["meta_scenario"],
            train_reset_config_path=PATH_PARAMS["train_reset_config"],
            test_reset_config_path=PATH_PARAMS["test_reset_config"],
            visualization_path=PATH_PARAMS["visualization"],
            logging_path=None,
            test_env=False,
            play=False,
            config_file=PATH_PARAMS["configs"]["commonroad-v1"],
            logging_mode=1,
            **kwargs,
    ) -> None:
        """
        Initialize environment, set scenario and planning problem.
        """
        # Set logger if not yet exists
        LOGGER.setLevel(logging_mode)

        if not len(LOGGER.handlers):
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging_mode)
            stream_handler.setFormatter(formatter)
            LOGGER.addHandler(stream_handler)

            if logging_path is not None:
                file_handler = logging.FileHandler(filename=os.path.join(logging_path, "console_copy.txt"))
                file_handler.setLevel(logging_mode)
                file_handler.setFormatter(formatter)
                LOGGER.addHandler(file_handler)

        LOGGER.debug("Initialization started")

        # Default configuration
        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Assume default environment configurations
        self.configs = config["env_configs"]

        # Overwrite environment configurations if specified
        if kwargs is not None:
            for k, v in kwargs.items():
                assert k in self.configs, f"Configuration item not supported: {k}"
                # TODO: update only one term in configs
                if isinstance(v, dict):
                    self.configs[k].update(v)
                else:
                    self.configs.update({k: v})

        # Make environment configurations as attributes
        self.vehicle_params: dict = self.configs["vehicle_params"]
        self.action_configs: dict = self.configs["action_configs"]
        self.render_configs: dict = self.configs["render_configs"]
        self.reward_type: str = self.configs["reward_type"]

        # change configurations when using point mass vehicle model
        if self.vehicle_params["vehicle_model"] == 0:
            self.observe_heading = False
            self.observe_steering_angle = False
            self.observe_global_turn_rate = False
            self.observe_distance_goal_long_lane = False

        # Flag for popping out scenarios
        self.play = play

        # Load scenarios and problems
        self.meta_scenario_path = meta_scenario_path
        self.all_problem_dict = dict()
        self.planning_problem_set_list = []

        # Accelerator structures
        # self.cache_goal_obs = dict()

        meta_scenario_reset_dict_path = os.path.join(self.meta_scenario_path, "meta_scenario_reset_dict.pickle")
        with open(meta_scenario_reset_dict_path, "rb") as f:
            self.meta_scenario_reset_dict = pickle.load(f)

        # problem_meta_scenario_dict_path = os.path.join(self.meta_scenario_path, "problem_meta_scenario_dict.pickle")
        # with open(problem_meta_scenario_dict_path, "rb") as f:
        #     self.problem_meta_scenario_dict = pickle.load(f)

        self.train_reset_config_path = train_reset_config_path

        if not test_env and not play:
            fns = glob.glob(os.path.join(train_reset_config_path, "*.pickle"))
            for fn in fns:
                with open(fn, "rb") as f:
                    self.all_problem_dict[os.path.basename(fn).split(".")[0]] = pickle.load(f)
            self.is_test_env = False
            LOGGER.info(f"Training on {train_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")
        else:
            fns = glob.glob(os.path.join(test_reset_config_path, "*.pickle"))
            for fn in fns:
                with open(fn, "rb") as f:
                    self.all_problem_dict[os.path.basename(fn).split(".")[0]] = pickle.load(f)
            LOGGER.info(f"Testing on {test_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")

        self.visualization_path = visualization_path
        self.current_step = 0

        self.termination = Termination(self.configs)
        self.terminated = False
        self.termination_reason = None

        if self.action_configs['action_type'] == "continuous":
            self.ego_action: ContinuousAction = ContinuousAction(self.vehicle_params, self.action_configs)
        else:
            self.ego_action: DiscretePMJerkAction = DiscretePMJerkAction(self.vehicle_params,
                                                                         self.action_configs['long_steps'],
                                                                         self.action_configs['lat_steps'])

        # Action space remove
        # TODO initialize action space with class
        if self.action_configs['action_type'] == "continuous":
            action_high = np.array([1.0, 1.0])
            self.action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype="float32")
        else:
            # action_count = self.action_configs['long_steps'] + self.action_configs['lat_steps']
            self.action_space = gym.spaces.MultiDiscrete([self.action_configs['long_steps'],
                                                          self.action_configs['lat_steps']])

        # Observation space
        self.observation_collector = ObservationCollector(self.configs)

        # Reward function
        self.reward_function: Reward = reward_constructor.make_reward(self.configs)

        # TODO initialize reward class

        LOGGER.debug(f"Meta scenario path: {meta_scenario_path}")
        LOGGER.debug(f"Training data path: {train_reset_config_path}")
        LOGGER.debug(f"Testing data path: {test_reset_config_path}")
        LOGGER.debug("Initialization done")

    @property
    def observation_space(self):
        return self.observation_collector.observation_space

    @property
    def observation_dict(self):
        return self.observation_collector.observation_dict

    def seed(self, seed=Union[None, int]):
        self.action_space.seed(seed)

    def reset(self, benchmark_id=None) -> np.ndarray:
        """
        Reset the environment.
        :param benchmark_id: benchmark id used for reset to specific scenario
        :return: observation
        """

        self._set_scenario_problem(benchmark_id)
        self.ego_action.reset(self.planning_problem.initial_state, self.scenario.dt)
        self.observation_collector.reset(self.scenario, self.planning_problem, self.reset_config, self.benchmark_id)
        # TODO: remove self._set_goal()
        self._set_goal()

        self.current_step = 0
        self.terminated = False

        initial_observation = self.observation_collector.observe(self.ego_action.vehicle)
        self.reward_function.reset(self.observation_dict, self.ego_action)
        self.termination.reset(self.observation_dict, self.ego_action)

        return initial_observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Propagate to next time step, compute next observations, reward and status.

        :param action: vehicle acceleration, vehicle steering velocity
        :return: observation, reward, status and other information
        """
        self.current_step += 1
        if self.action_configs['action_type'] == "continuous":
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # Make action and observe result
        self.ego_action.step(action, local_ccosy=self.observation_collector.local_ccosy)
        observation = self.observation_collector.observe(self.ego_action.vehicle)

        # Check for termination
        done, reason, termination_info = self.termination.is_terminated(self.observation_dict, self.ego_action)
        if reason is not None:
            self.termination_reason = reason

        if done:
            self.terminated = True

        # Calculate reward
        reward = self.reward_function.calc_reward(self.observation_dict, self.ego_action)

        info = {
            "scenario_name": self.benchmark_id,
            "chosen_action": action,
            "current_episode_time_step": self.current_step,
            "max_episode_time_steps": self.observation_collector.episode_length,
        }
        info.update(termination_info)

        return observation, reward, done, info

    def render(self, mode: str = "human") -> None:
        """
        Generate images for visualization.

        :param mode: default as human for visualization
        :return: None
        """
        # Render only every xth timestep, the first and the last
        if not (self.current_step % self.render_configs["render_skip_timesteps"] == 0 or self.terminated):
            return

        # Draw scenario, goal, sensing range and detected obstacles
        draw_params = {
            "scenario": {"time_begin": self.current_step,
                         "lanelet_network": {"lanelet": {"show_label": False,
                                                         "fill_lanelet": True},
                                             "traffic_sign": {"draw_traffic_signs": False,
                                                              "show_traffic_signs": "all",
                                                              "show_label": False,
                                                              'scale_factor': 0.1, },
                                             "intersection": {"draw_intersections": True}},
                         "dynamic_obstacle": {"show_label": False}}}
        draw_object(self.scenario, draw_params=draw_params)
        # draw_params={"time_begin": self.current_step,
        #              "scenario": {"lanelet_network": {"lanelet": {"show_label": False,
        #                                                           "fill_lanelet": True}},
        #                           "dynamic_obstacle": {"show_label": True}}})

        # Draw certain objects only once
        if not self.render_configs["render_combine_frames"] or self.current_step == 0:
            draw_object(self.planning_problem)
            self.observation_collector.render(self.render_configs)

        # Draw ego vehicle
        crdc_draw_dispatch.draw_object(self.ego_action.vehicle.collision_object,
                                       draw_params={"collision": {"facecolor": "green", "zorder": 30}})

        plt.gca().set_aspect("equal")
        plt.autoscale()

        # Save figure, only if frames should not be combined or simulation is over
        os.makedirs(os.path.join(self.visualization_path, self.scenario.benchmark_id), exist_ok=True)
        if not self.render_configs["render_combine_frames"] or self.terminated:
            plt.savefig(os.path.join(self.visualization_path, self.scenario.benchmark_id,
                                     self.file_name_format % self.current_step) + ".png",
                        format="png",
                        dpi=300,
                        bbox_inches="tight")
            plt.close()

    # =================================================================================================================
    #
    #                                    reset functions
    #
    # =================================================================================================================
    def _set_scenario_problem(self, benchmark=None) -> None:
        """
        Select scenario and planning problem.

        :return: None
        """
        if self.play:
            # pop instead of reusing
            LOGGER.info(f"Number of scenarios left {len(list(self.all_problem_dict.keys()))}")
            self.benchmark_id = random.choice(list(self.all_problem_dict.keys()))
            problem_dict = self.all_problem_dict.pop(self.benchmark_id)
        else:
            if benchmark is not None:
                self.benchmark_id = benchmark
                problem_dict = self.all_problem_dict[benchmark]
            else:
                self.benchmark_id, problem_dict = random.choice(list(self.all_problem_dict.items()))

        # Set reset config dictionary
        scenario_id = ScenarioID.from_benchmark_id(self.benchmark_id, "2020a")
        map_id = parse_map_name(scenario_id)
        self.reset_config = self.meta_scenario_reset_dict[map_id]
        # meta_scenario = self.problem_meta_scenario_dict[self.benchmark_id]
        self.scenario = restore_scenario(self.reset_config["meta_scenario"], problem_dict["obstacle"], scenario_id)
        self.planning_problem: PlanningProblem = random.choice(
            list(problem_dict["planning_problem_set"].planning_problem_dict.values())
        )

        # Set name format for visualization
        self.file_name_format = self.benchmark_id + "_ts_%03d"

    def _set_goal(self) -> None:
        """
        Set ego vehicle and initialize its status.

        :return: None
        """
        self.goal = self.observation_collector.goal_observation

        # Compute initial distance to goal for normalization if required
        if self.reward_type == "dense_reward":  # or "hybrid_reward":
            self.observation_collector._create_navigator()
            distance_goal_long, distance_goal_lat = self.goal.get_long_lat_distance_to_goal(
                self.ego_action.vehicle.state.position, self.observation_collector.navigator
            )
            self.initial_goal_dist = np.sqrt(distance_goal_long ** 2 + distance_goal_lat ** 2)

            # Prevent cases where the ego vehicle starts in the goal region
            if self.initial_goal_dist < 1.0:
                warnings.warn("Ego vehicle starts in the goal region")
                self.initial_goal_dist = 1.0


if __name__ == "__main__":
    # /data/inD-dataset-v1.0/cr_scenarios
    env = gym.make("commonroad-v1",
                    meta_scenario_path="/home/xiao/projects/commonroad-rl/pickles/highD/tmp/meta_scenario",
                    train_reset_config_path="/home/xiao/projects/commonroad-rl/pickles/highD/tmp/problem")
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.render()
        observations, rewards, done, info = env.step(action)
