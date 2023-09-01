from dataclasses import dataclass
from enum import IntEnum, auto, unique

import numpy as np
import sympy as sm


@unique
class SteerWith(IntEnum):
    """Enumeration of options for controlling the bicycle steering.

    Members
    =======
    STEER_TORQUE
        Will include T7 and use this to control the steer.
    ELBOW_TORQUE
        Will include T13 and T16 and use these to control the steer.
    MUSCLES
        Will add bicep and tricep musculotendons to both arms and use the
        excitation of these to control the steer

    """
    STEER_TORQUE = auto()
    ELBOW_TORQUE = auto()
    MUSCLES = auto()


@dataclass
class Metadata:
    duration: float
    longitudinal_displacement: float
    lateral_displacement: float
    num_nodes: float
    interval_value: float
    steer_with: SteerWith
    include_roll_torque: bool
    target_q2: callable
    constants: np.array


@dataclass
class ForOpty:
    """Dataclass for passing the equations of motion."""
    time: sm.Symbol
    state_vars: sm.Matrix
    input_vars: sm.Matrix
    equations_of_motion: sm.Matrix
    parameters: sm.Matrix

    @property
    def t(self):
        return self.time

    @property
    def x(self):
        return self.state_vars

    @property
    def r(self):
        return self.input_vars

    @property
    def eom(self):
        return self.equations_of_motion

    @property
    def p(self):
        return self.parameters
