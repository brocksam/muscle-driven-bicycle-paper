"""A trajectory tracking OCP with a muscle-driven steered bicycle."""

from timeit import default_timer as timer
import logging

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import parse_free

from container import Metadata, SteerWith
from generate_eom import constants_values, gen_eom_for_opty
from generate_init_guess import gen_init_guess_for_opty
from utils import plot_trajectories

logging.basicConfig(level=logging.INFO)


SPEED = 3.0  # m/s
LONGITUDINAL_DISPLACEMENT = 10.0
LATERAL_DISPLACEMENT = 2.0
DURATION = LONGITUDINAL_DISPLACEMENT / SPEED
NUM_NODES = 301
INTERVAL_VALUE = DURATION / (NUM_NODES - 1)
WEIGHT = 0.95

#STEER_WITH = SteerWith.STEER_TORQUE
#STEER_WITH = SteerWith.ELBOW_TORQUE
STEER_WITH = SteerWith.MUSCLES
INCLUDE_ROLL_TORQUE = False

if STEER_WITH.name == "STEER_TORQUE":
    NUM_INPUTS = 2  # T6, T7
    NUM_STATES = 28
elif STEER_WITH.name == "ELBOW_TORQUE":
    NUM_INPUTS = 3  # T6, T13, T16
    NUM_STATES = 28
elif STEER_WITH.name == "MUSCLES":
    NUM_INPUTS = 5  # T6, e1, e2, e3, e4
    NUM_STATES = 32


def target_q2(q1):
    return 0.5 * LATERAL_DISPLACEMENT * (1 - np.cos(np.pi * q1 / LONGITUDINAL_DISPLACEMENT))


def obj(free):
    """Minimize the sum of the squares of the muscle activations."""
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q1, q2 = x[0], x[1]
    err = (target_q2(q1) - q2)
    # TODO : drive torque shouldn't be summed with muscle excitation, as they
    # are different units (also could solve with setting T6 = 0).
    return INTERVAL_VALUE*(WEIGHT*np.sum(err**2) + (1.0-WEIGHT)*np.sum(r.flatten())**2)


def obj_grad(free):
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q1, q2 = x[0], x[1]
    grad = np.zeros_like(free)
    # TODO : Add WEIGHT to q1, q2. Also, why isn't INTERVAL_VALUE in the q1, q2
    # derivs?
    dJdq1 = WEIGHT*INTERVAL_VALUE*np.pi*LATERAL_DISPLACEMENT*(LATERAL_DISPLACEMENT*(1 - np.cos(np.pi*q1/LONGITUDINAL_DISPLACEMENT))/2 - q2)*np.sin(np.pi*q1/LONGITUDINAL_DISPLACEMENT)/LONGITUDINAL_DISPLACEMENT
    dJdq2 = WEIGHT*INTERVAL_VALUE*(LATERAL_DISPLACEMENT*(np.cos(np.pi*q1/LONGITUDINAL_DISPLACEMENT) - 1) + 2*q2)
    grad[0:1*NUM_NODES] = dJdq1
    grad[1*NUM_NODES:2*NUM_NODES] = dJdq2
    grad[NUM_STATES*NUM_NODES:(NUM_STATES + NUM_INPUTS)*NUM_NODES] = 2.0*(1.0-WEIGHT)*INTERVAL_VALUE*r.flatten()
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
    q6.replace(model.t, 0.0),
    q7.replace(model.t, 0.0) - q7.replace(model.t, DURATION),  # periodic steering rotation angle
    q8.replace(model.t, 0.0),
    q11.replace(model.t, 0.0) - q11.replace(model.t, DURATION),
    q12.replace(model.t, 0.0) - q12.replace(model.t, DURATION),
    q13.replace(model.t, 0.0) - q13.replace(model.t, DURATION),
    q14.replace(model.t, 0.0) - q14.replace(model.t, DURATION),
    q15.replace(model.t, 0.0) - q15.replace(model.t, DURATION),
    q16.replace(model.t, 0.0) - q16.replace(model.t, DURATION),
    u1.func(0.0),
    u2.func(0.0),
    #u1.replace(model.t, 0.0) - LONGITUDINAL_DISPLACEMENT/DURATION,
    #u1.replace(model.t, DURATION) - LONGITUDINAL_DISPLACEMENT/DURATION,
    #u2.replace(model.t, 0.0) - u2.replace(model.t, DURATION),
    u3.replace(model.t, 0.0) - u3.replace(model.t, DURATION),
    u4.replace(model.t, 0.0) - u4.replace(model.t, DURATION),
    u5.replace(model.t, 0.0) - u5.replace(model.t, DURATION),
    u6.func(0.0),
    #u6.replace(model.t, 0.0) + LONGITUDINAL_DISPLACEMENT/(DURATION*constants[-1]),
    #u6.replace(model.t, DURATION) + LONGITUDINAL_DISPLACEMENT/(DURATION*constants[-1]),
    u7.replace(model.t, 0.0) - u7.replace(model.t, DURATION),
    u8.func(0.0),
    #u8.replace(model.t, 0.0) + LONGITUDINAL_DISPLACEMENT/(DURATION*constants[-2]),
    #u8.replace(model.t, DURATION) + LONGITUDINAL_DISPLACEMENT/(DURATION*constants[-2]),
    u11.replace(model.t, 0.0) - u11.replace(model.t, DURATION),
    u12.replace(model.t, 0.0) - u12.replace(model.t, DURATION),
    u13.replace(model.t, 0.0) - u13.replace(model.t, DURATION),
    u14.replace(model.t, 0.0) - u14.replace(model.t, DURATION),
    u15.replace(model.t, 0.0) - u15.replace(model.t, DURATION),
    u16.replace(model.t, 0.0) - u16.replace(model.t, DURATION),
)

bounds = {
    q1: (-0.1, LONGITUDINAL_DISPLACEMENT + 0.1),
    q2: (-0.1, LATERAL_DISPLACEMENT + 0.1),
    q3: (-1.0, 1.0),
    q4: (-1.0, 1.0),
    q5: (-1.0, 1.0),
    q6: (-100.0, 100.0),
    q7: (-1.0, 1.0),
    q8: (-100.0, 100.0),
    q11: (-1.0, 1.0),
    q12: (-1.0, 1.0),
    q13: (0.0, 3.0),
    q14: (-1.0, 1.0),
    q15: (-1.0, 1.0),
    q16: (0.0, 3.0),
    u1: (0.0, 10.0),
    u2: (-5.0, 5.0),
    u3: (-4.0, 4.0),
    u4: (-4.0, 4.0),
    u5: (-4.0, 4.0),
    u6: (-20.0, 0.0),
    u7: (-4.0, 4.0),
    u8: (-20.0, 0.0),
    u11: (-4.0, 4.0),
    u12: (-4.0, 4.0),
    u13: (-4.0, 4.0),
    u14: (-4.0, 4.0),
    u15: (-4.0, 4.0),
    u16: (-4.0, 4.0),
}
if INCLUDE_ROLL_TORQUE:
    bounds = {
        **bounds,
        T4: (-10.0, 10.0),
    }
if STEER_WITH is SteerWith.STEER_TORQUE:
    bounds = {
        **bounds,
        T6: (-100.0, 100.0),
        T7: (-10.0, 10.0),
    }
elif STEER_WITH is SteerWith.ELBOW_TORQUE:
    bounds = {
        **bounds,
        T6: (-100.0, 100.0),
        T13: (-10.0, 10.0),
        T16: (-10.0, 10.0),
    }
elif STEER_WITH is SteerWith.MUSCLES:
    bounds = {
        **bounds,
        a1: (0.0, 1.0),
        a2: (0.0, 1.0),
        a3: (0.0, 1.0),
        a4: (0.0, 1.0),
        T6: (-100.0, 100.0),
        e1: (0.0, 1.0),
        e2: (0.0, 1.0),
        e3: (0.0, 1.0),
        e4: (0.0, 1.0),
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
    #integration_method='midpoint',
    parallel=True,
    tmp_dir='codegen',
)

problem.add_option('nlp_scaling_method', 'gradient-based')
problem.add_option('max_iter', 5000)
#problem.add_option('linear_solver', 'spral')

stop = timer()
print(f'`opty.Problem` instantiated in {stop-start}s.')

metadata = Metadata(DURATION, LONGITUDINAL_DISPLACEMENT, LATERAL_DISPLACEMENT,
                    NUM_NODES, INTERVAL_VALUE, STEER_WITH, INCLUDE_ROLL_TORQUE,
                    target_q2, constants)

# Generate a sensible initial guess
# np.array of length problem.num_free
initial_guess = gen_init_guess_for_opty(model, problem, metadata)

# Find the optimal solution.
sol, info = problem.solve(initial_guess)

q1_sol = sol[:NUM_NODES]
q2_sol = sol[NUM_NODES:2*NUM_NODES]
q1_target = np.linspace(0.0, LONGITUDINAL_DISPLACEMENT)
q2_target = target_q2(q1_target)

fig, ax = plt.subplots()
ax.plot(q1_target, q2_target, label='Target', linewidth=4)
ax.plot(q1_sol, q2_sol, label='Solution', linewidth=2, linestyle='dashed')
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Distance [m]')
ax.legend()

solx, solr, solp = parse_free(sol, NUM_STATES, NUM_INPUTS, NUM_NODES)
solt = np.linspace(0.0, DURATION, num=NUM_NODES)
plot_trajectories(solt, solx, solr, model.x,
                  problem.collocator.input_trajectories, skip_first=True)

plt.show()
