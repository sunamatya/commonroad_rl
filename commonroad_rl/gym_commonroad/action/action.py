"""
Module containing the action base class
"""
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from commonroad_rl.gym_commonroad.action.vehicle import *


def _rotate_to_curvi(vector: np.ndarray, local_ccosy: CurvilinearCoordinateSystem, pos: np.ndarray)\
        -> np.ndarray:
    """
    Function to rotate a vector in the curvilinear system to its counterpart in the normal coordinate system

    :param vector: The vector in question
    :returns: The rotated vector
    """
    try:
        long, _ = local_ccosy.convert_to_curvilinear_coords(pos[0], pos[1])
    except ValueError:
        long = 0.

    tangent = local_ccosy.tangent(long)
    theta = np.math.atan2(tangent[1], tangent[0])
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    return np.matmul(rot_mat, vector)


class Action(ABC):
    """
    Description:
        Abstract base class of all action spaces
    """

    def __init__(self):
        """ Initialize empty object """
        super().__init__()
        self.vehicle = None

    @abstractmethod
    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        pass


class DiscreteAction(Action):
    """
    Description:
        Abstract base class of all discrete action spaces. Each high-level discrete
        action is converted to a low-level trajectory by a specified planner.
    """

    def __init__(self, params_dict: dict):
        """ Initialize empty object """
        super().__init__()
        self.vehicle = DiscreteVehicle(params_dict)
        self.local_ccosy = None

    def reset(self, initial_state: State, dt: float) -> None:
        """
        resets the vehicle
        :param initial_state: initial state
        :param dt: time step size of scenario
        """
        self.vehicle.reset(initial_state, dt)

    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        self.local_ccosy = local_ccosy
        self.vehicle.current_time_step += 1
        state = self._get_new_state(action)
        self.vehicle.set_current_state(state)
        self.vehicle.update_collision_object()

    @abstractmethod
    def _get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """function which return new states given the action and current state"""
        pass


class DiscretePMJerkAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model and jerk control
    """

    def __init__(self, params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete longitudinal jerk steps
        :param lat_steps: number of discrete lateral jerk steps
        """
        super().__init__(params_dict)
        if VehicleModel(params_dict["vehicle_model"]) != VehicleModel.PM:
            raise ValueError('ERROR in ACTION INITIALIZATION: '
                             'DiscretePMAction can only be used with the PM vehicle_type')
        self.vehicle = ContinuousVehicle(params_dict)
        if long_steps % 2 == 0 or lat_steps % 2 == 0:
            raise ValueError('ERROR in ACTION INITIALIZATION: '
                             'The discrete steps for longitudinal and lateral jerk '
                             'have to be odd numbers, so constant velocity without turning is an possible action')
        self.j_max = 10  # set the maximum jerk
        self.long_step_size = (self.j_max * 2) / (long_steps - 1)
        self.action_mapping_long = {}
        for idx in range(long_steps):
            self.action_mapping_long[idx] = (self.j_max - (idx * self.long_step_size))
        self.lat_step_size = (self.j_max * 2) / (lat_steps - 1)
        self.action_mapping_lat = {}
        for idx in range(lat_steps):
            self.action_mapping_lat[idx] = (self.j_max - (idx * self.lat_step_size))

        a_max = self.vehicle.parameters.longitudinal.a_max
        self._rescale_factor = np.array([a_max, a_max])

    def _get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """
        # map discrete action to jerk and calculate a
        # correct rescale in order to make 0 acceleration achievable again when sign of acc switches
        a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration
        if self.vehicle.state.acceleration != 0 and np.sign(a_long) != np.sign(self.vehicle.state.acceleration) and \
                (np.abs(a_long) % (self.long_step_size * self.vehicle.dt)) != 0:
            if a_long > 0:
                a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration - \
                         (np.abs(a_long) % (self.long_step_size * self.vehicle.dt))
            else:
                a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration + \
                         (np.abs(a_long) % (self.long_step_size * self.vehicle.dt))

        a_lat = self.action_mapping_lat[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y
        if self.vehicle.state.acceleration_y != 0 and np.sign(a_lat) != np.sign(self.vehicle.state.acceleration_y) and \
                (np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt)) != 0:
            if a_lat > 0:
                a_lat = self.action_mapping_long[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y - (
                        np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt))
            else:
                a_lat = self.action_mapping_long[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y + (
                        np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt))
        # add rotation if necessary
        control_input = np.array([a_long, a_lat])
        if np.linalg.norm(control_input) > self.vehicle.parameters.longitudinal.a_max:
            control_input = (control_input / np.linalg.norm(control_input)) * self._rescale_factor

        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, self.vehicle.state.position)

        # get the next state from the PM model
        next_state = self.vehicle.get_new_state(control_input, "acceleration")
        return next_state


class DiscretePMAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model
    """

    def __init__(self, params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete acceleration steps
        :param lat_steps: number of discrete turning steps
        """
        super().__init__(params_dict)
        if VehicleModel(params_dict["vehicle_model"]) != VehicleModel.PM:
            print('ERROR in ACTION INITIALIZATION: DiscretePMAction can only be used with the PM vehicle_type')
            raise ValueError
        self.vehicle = ContinuousVehicle(params_dict)
        if lat_steps % 2 == 0 or long_steps % 2 == 0:
            raise ValueError('ERROR in ACTION INITIALIZATION: The discrete steps for turning and accelerating '
                             'have to be odd numbers, so constant velocity without turning is a possible action')
        a_max = self.vehicle.parameters.longitudinal.a_max
        a_steps = (a_max * 2) / (long_steps - 1)
        self.action_mapping_dict = {}
        for idx in range(long_steps):
            self.action_mapping_dict[idx] = np.array([a_max - (idx * a_steps), 0])
        i = 0
        turn_steps = (a_max * 2) / (lat_steps - 1)
        for j in range(lat_steps):
            if a_max - (j * turn_steps) != 0.0:
                self.action_mapping_dict[long_steps + i] = np.array([0, a_max - (j * turn_steps)])
                i += 1

    def _get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """

        # map discrete action to control inputs
        control_input = self.action_mapping_dict[action]

        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, self.vehicle.state.position)

        # get the next state from the PM model
        next_state = self.vehicle.get_new_state(control_input, "acceleration")
        return next_state


class ContinuousAction(Action):
    """
    Description:
        Module for continuous action space; actions correspond to vehicle control inputs
    """

    def __init__(self, params_dict: dict, action_dict: dict):
        """ Initialize object """
        super().__init__()
        # create vehicle object
        self.vehicle = ContinuousVehicle(params_dict)
        self.action_base = action_dict['action_base']
        self._set_rescale_factors(initial=True)

    def _set_rescale_factors(self, initial=False):

        a_max = self.vehicle.parameters.longitudinal.a_max
        # rescale factors for PM model
        if self.vehicle.vehicle_model == VehicleModel.PM:
            self._rescale_factor = np.array([a_max, a_max])
            self._rescale_bias = 0.0
        # rescale factors for KS model
        elif self.vehicle.vehicle_model == VehicleModel.KS:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2.0, a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2.0, 0.])
        # rescale factors for YawRate model
        elif self.vehicle.vehicle_model == VehicleModel.YawRate:
            if not initial:
                yaw_rate_max = self.vehicle.parameters.yaw.v_max = np.abs(
                    self.vehicle.parameters.longitudinal.a_max / (self.vehicle.state.velocity + 1e-6))
                yaw_rate_min = self.vehicle.parameters.yaw.v_min = -self.vehicle.parameters.yaw.v_max
            else:
                yaw_rate_max = -2.
                yaw_rate_min = 2.
            self._rescale_factor = np.array([a_max, (yaw_rate_max - yaw_rate_min) / 2.0])
            self._rescale_bias = np.array([0.0, (yaw_rate_max + yaw_rate_min) / 2.0])

    def reset(self, initial_state: State, dt: float) -> None:
        self.vehicle.reset(initial_state, dt)

    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        :return: New state of ego vehicle
        """

        self.vehicle.current_time_step += 1
        rescaled_action = self.rescale_action(action)
        new_state = self.vehicle.get_new_state(rescaled_action, self.action_base)
        self.vehicle.set_current_state(new_state)

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param action: action from the CommonroadEnv.
        :return: rescaled action
        """
        if self.vehicle.vehicle_model == VehicleModel.YawRate:
            # update rescale factors
            self._set_rescale_factors()

        return self._rescale_factor * action + self._rescale_bias

    # @staticmethod
    # def _get_new_state(action: np.ndarray, vehicle) -> State:
    #     # generate the next state for the given action
    #     new_state = vehicle.get_new_state(action)
    #     return new_state
