"""
Module tests of the module gym_commonroad
"""
import os
import random
import timeit
import numpy as np
from stable_baselines.common.env_checker import check_env
from commonroad_rl.gym_commonroad import *
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.non_functional import function_to_string
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

resource_path = resource_root("test_gym_commonroad")
pickle_xml_scenarios(
    input_dir=os.path.join(resource_path),
    output_dir=os.path.join(resource_path, "pickles")
)

meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")

output_path = output_root("test_gym_commonroad")
visualization_path = os.path.join(output_path, "visualization")


@pytest.mark.parametrize(("num_of_checks", "test_env", "play"),
                         [(15, False, False),
                          (15, False, True),
                          (15, True, False),
                          (15, True, True)])
@module_test
@functional
def test_check_env(num_of_checks, test_env, play):
    # Run more circles of checking to search for sporadic issues
    for idx in range(num_of_checks):
        print(f"Checking progress: {idx + 1}/{num_of_checks}")
        env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                       test_reset_config_path=problem_path, visualization_path=visualization_path, test_env=False,
                       play=False, )
        check_env(env)


@pytest.mark.parametrize(("reward_type"),
                         [("hybrid_reward"),
                          ("sparse_reward"),
                          ("dense_reward")])
@module_test
@functional
def test_step(reward_type):
    env = gym.make("commonroad-v1",
                   meta_scenario_path=meta_scenario_path,
                   train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path,
                   visualization_path=visualization_path,
                   reward_type=reward_type)
    env.reset()
    done = False
    while not done:
        # for i in range(50):
        action = env.action_space.sample()
        # action = np.array([0., 0.1])
        obs, reward, done, info = env.step(action)
        # TODO: define reference format and assert
        # print(f"step {i}, reward {reward:2f}")


@pytest.mark.parametrize(("reward_type"),
                         [("dense_reward"),
                          ("sparse_reward"),
                          ("hybrid_reward")])
@module_test
@functional
def test_observation_order(reward_type):
    env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path, flatten_observation=False)

    # set random seed to make the env choose the same planning problem
    random.seed(0)
    obs_dict = env.reset()

    # collect observation in other format
    env = gym.make("commonroad-v1", meta_scenario_path=meta_scenario_path, train_reset_config_path=problem_path,
                   test_reset_config_path=problem_path, flatten_observation=True)

    # seed needs to be reset before function call
    random.seed(0)
    obs_flatten = env.reset()
    obs_flatten_exp = np.zeros(env.observation_space.shape)

    # flatten the dictionary observation
    index = 0
    for obs_dict_value in obs_dict.values():
        size = np.prod(obs_dict_value.shape)
        obs_flatten_exp[index: index + size] = obs_dict_value.flat
        index += size

    # compare 2 observation
    assert np.allclose(obs_flatten_exp, obs_flatten), "Two observations don't have the same order"


@pytest.mark.parametrize(("reward_type"),
                         [("dense_reward"),
                          ("sparse_reward"),
                          ("hybrid_reward")])
@module_test
@nonfunctional
def test_step_time(reward_type):
    # Define reference time
    reference_time = 15.0

    def measurement_setup():
        import gym
        import numpy as np

        env = gym.make("commonroad-v1", meta_scenario_path="{meta_scenario_path}", train_reset_config_path="{problem_path}",
                       test_reset_config_path="{problem_path}", visualization_path="{visualization_path}",
                       reward_type="{reward_type}", )
        env.reset()
        action = np.array([0.0, 0.0])

    def measurement_code(env, action):
        env.step((action))

    setup_str = function_to_string(measurement_setup)
    code_str = function_to_string(measurement_code)

    times = timeit.repeat(setup=setup_str, stmt=code_str, repeat=1, number=1000)
    min_time = np.amin(times)

    # TODO: Set exclusive CPU usage for this thread, because other processes influence the result  # assert
    #  average_time < reference_time, f"The step is too slow, average time was {average_time}"
