"""A trajectory tracking OCP with a muscle-driven steered bicycle."""

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

from generate_eom import SteerWith, constants_values, gen_eom_for_opty


DURATION = 2.0
LONGITUDINAL_DISPLACEMENT = 10.0
LATERAL_DISPLACEMENT = 1.0
NUM_NODES = 100
INTERVAL_VALUE = DURATION / (NUM_NODES - 1)

STEER_WITH = SteerWith.STEER_TORQUE
INCLUDE_ROLL_TORQUE = False


def target_q2(q1):
    return 0.5 * LATERAL_DISPLACEMENT * (1 - np.cos(np.pi * q1 / LONGITUDINAL_DISPLACEMENT))


def obj(free):
    """Minimize the sum of the squares of the muscle activations."""
    q1 = free[:NUM_NODES]
    q2 = free[NUM_NODES:2*NUM_NODES]
    err = (target_q2(q1) - q2)
    return INTERVAL_VALUE*(np.sum(err**2))


def obj_grad(free):
    q1 = free[:NUM_NODES]
    q2 = free[NUM_NODES:2*NUM_NODES]
    grad = np.zeros_like(free)
    dJdq1 = np.pi*LATERAL_DISPLACEMENT*(LATERAL_DISPLACEMENT*(1 - cos(np.pi*q1/LONGITUDINAL_DISPLACEMENT))/2 - q2)*sin(pi*q1/LONGITUDINAL_DISPLACEMENT)/LONGITUDINAL_DISPLACEMENT
    dJdq2 = LATERAL_DISPLACEMENT*(cos(pi*q1/LONGITUDINAL_DISPLACEMENT) - 1) + 2*q2
    grad[:NUM_NODES] = dJdq1
    grad[NUM_NODES:2*NUM_NODES] = dJdq2
    return grad


model = gen_eom_for_opty(STEER_WITH, INCLUDE_ROLL_TORQUE)
constants = constants_values(STEER_WITH)

q1, q2, q3, q4, q5, q6, q7, q8, q11, q12, q13, q14, q15, q16 = model.x[:14]
u1, u2, u3, u4, u5, u6, u7, u8, u11, u12, u13, u14, u15, u16 = model.x[14:28]
if STEER_WITH is SteerWith.MUSCLES:
    a1, a2, a3, a4 = model.x[28:]
    if INCLUDE_ROLL_TORQUE:
        T4, T6 = model.r[:2]
        e1, e2, e3, e4 = model.r[2:]
    else:
        T6 = model.r[0]
        e1, e2, e3, e4 = model.r[1:]
elif STEER_WITH is SteerWith.ELBOW_TORQUE:
    if INCLUDE_ROLL_TORQUE:
        T4, T6, T13, T16 = model.r
    else:
        T6, T13, T16 = model.r
elif STEER_WITH is SteerWith.STEER_TORQUE:
    if INCLUDE_ROLL_TORQUE:
        T4, T6, T7 = model.r
    else:
        T6, T7 = model.r
else:
    raise NotImplementedError

instance_constraints = (
    q1.replace(model.t, 0.0),
    q1.replace(model.t, DURATION) - LONGITUDINAL_DISPLACEMENT,
    q2.replace(model.t, 0.0),
    q2.replace(model.t, DURATION) - LATERAL_DISPLACEMENT,
    q3.replace(model.t, 0.0) - q3.replace(model.t, DURATION),  # periodic yaw
    q4.replace(model.t, 0.0) - q4.replace(model.t, DURATION),  # periodic roll
    q5.replace(model.t, 0.0) - q5.replace(model.t, DURATION),  # periodic pitch
    q7.replace(model.t, 0.0) - q7.replace(model.t, DURATION),  # periodic steering rotation angle
    q11.replace(model.t, 0.0) - q11.replace(model.t, DURATION),
    q12.replace(model.t, 0.0) - q12.replace(model.t, DURATION),
    q13.replace(model.t, 0.0) - q13.replace(model.t, DURATION),
    q14.replace(model.t, 0.0) - q14.replace(model.t, DURATION),
    q15.replace(model.t, 0.0) - q15.replace(model.t, DURATION),
    q16.replace(model.t, 0.0) - q16.replace(model.t, DURATION),
    u1.replace(model.t, 0.0) - u1.replace(model.t, DURATION),
    u2.replace(model.t, 0.0) - u2.replace(model.t, DURATION),
    u3.replace(model.t, 0.0) - u3.replace(model.t, DURATION),
    u4.replace(model.t, 0.0) - u4.replace(model.t, DURATION),
    u5.replace(model.t, 0.0) - u5.replace(model.t, DURATION),
    u6.replace(model.t, 0.0) - u6.replace(model.t, DURATION),
    u7.replace(model.t, 0.0) - u7.replace(model.t, DURATION),
    u8.replace(model.t, 0.0) - u8.replace(model.t, DURATION),
    u11.replace(model.t, 0.0) - u11.replace(model.t, DURATION),
    u12.replace(model.t, 0.0) - u12.replace(model.t, DURATION),
    u13.replace(model.t, 0.0) - u13.replace(model.t, DURATION),
    u14.replace(model.t, 0.0) - u14.replace(model.t, DURATION),
    u15.replace(model.t, 0.0) - u15.replace(model.t, DURATION),
    u16.replace(model.t, 0.0) - u16.replace(model.t, DURATION),
)

bounds = {
    q1: (0.0, LONGITUDINAL_DISPLACEMENT),
    q2: (0.0, LATERAL_DISPLACEMENT),
    q3: (-100.0, 100.0),
    q4: (-100.0, 100.0),
    q5: (-100.0, 100.0),
    q6: (-100.0, 100.0),
    q7: (-100.0, 100.0),
    q8: (-100.0, 100.0),
    q11: (-100.0, 100.0),
    q12: (-100.0, 100.0),
    q13: (-100.0, 100.0),
    q14: (-100.0, 100.0),
    q15: (-100.0, 100.0),
    q16: (-100.0, 100.0),
    u1: (-100.0, 100.0),
    u2: (-100.0, 100.0),
    u3: (-100.0, 100.0),
    u4: (-100.0, 100.0),
    u5: (-100.0, 100.0),
    u6: (-100.0, 100.0),
    u7: (-100.0, 100.0),
    u8: (-100.0, 100.0),
    u11: (-100.0, 100.0),
    u12: (-100.0, 100.0),
    u13: (-100.0, 100.0),
    u14: (-100.0, 100.0),
    u15: (-100.0, 100.0),
    u16: (-100.0, 100.0),
}
if STEER_WITH is SteerWith.STEER_TORQUE:
    bounds = {
        **bounds,
        T6: (-100.0, 100.0),
        T7: (100.0, 100.0),
    }
else:
    raise NotImplementedError


print('Beginning to instantiate the `opty.Problem`.')
start = timer()

problem = Problem(
    obj,
    obj_grad,
    model.eom,
    model.state_vars,
    NUM_NODES,
    INTERVAL_VALUE,
    known_parameter_map=dict(zip(model.p, constants)),
    instance_constraints=instance_constraints,
    bounds=bounds,
    integration_method='midpoint',
    tmp_dir='/Users/sambrockie/Documents/Delft/Code/muscle-driven-bicycle-paper/codegen'
)

stop = timer()
print(f'`opty.Problem` instantiated in {stop-start}s.')

# Generate a sensible initial guess
# np.array of length problem.num_free
initial_guess = np.zeros(problem.num_free)

# Find the optimal solution.
sol, info = problem.solve(initial_guess)

q1_sol = sol[:NUM_NODES]
q2_sol = sol[NUM_NODES:2*NUM_NODES]
q1_target = np.linspace(0.0, LONGITUDINAL_DISPLACEMENT)
q2_target = target_q2(q1_target)

plt.plot(q1_sol, q2_sol, label='Solution')
plt.plot(q1_target, q2_target, label='Target')
plt.show()
