from dataclasses import dataclass
from enum import IntEnum, auto, unique

import numpy as np
import sympy as sm
import sympy.physics.mechanics as mec
from sympy.physics._biomechanics import (
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)

from utils import ReferenceFrame, ExtensorPathway


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


def gen_eom_for_opty(steer_with=SteerWith.MUSCLES, include_roll_torque=False):
    """Returns a dictionary with necessary symbolic expressions and variables
    for use in Opty.

    Parameters
    ==========

    steer_with : SteerWith
        One of the three members of the ``SteerWith`` enumeration. Choosing the
        ``STEER_TORQUE`` member will include ``T7`` in the model and use this
        to steer directly. Choosing the ``ELBOW_TORQUE`` will include ``T13``
        and ``T16`` at the two elbows and use these to steer. Choosing
        ``MUSCLES`` (default) will add a bicep and tricep musculotendon to each
        arm, the excitation of which will be used to steer.
    include_roll_torque : bool
        Boolean to select whether the roll torque ``T4`` should be included in
        the formulation. The default is ``False``.

    See Also
    ========

    SteerWith : Enumeration of steering options.

    """

    ##################
    # Reference Frames
    ##################

    print('Defining reference frames.')

    # Newtonian Frame
    N = ReferenceFrame('N')
    # Yaw Frame
    A = ReferenceFrame('A')
    # Roll Frame
    B = ReferenceFrame('B')
    # Rear Frame
    C = ReferenceFrame('C')
    # Rear Wheel Frame
    D = ReferenceFrame('D')
    # Front Frame
    E = ReferenceFrame('E')
    # Front Wheel Frame
    F = ReferenceFrame('F')
    # Right Upper Arm
    G = ReferenceFrame('G')
    # Right Lower Arm
    H = ReferenceFrame('H')
    # Left Upper Arm
    I = ReferenceFrame('I')
    # Left Lower Arm
    J = ReferenceFrame('J')

    ####################################
    # Generalized Coordinates and Speeds
    ####################################

    # All the following are a function of time.
    t = mec.dynamicsymbols._t

    # q1: perpendicular distance from the n2> axis to the rear contact
    #     point in the ground plane
    # q2: perpendicular distance from the n1> axis to the rear contact
    #     point in the ground plane
    # q3: frame yaw angle
    # q4: frame roll angle
    # q5: frame pitch angle
    # q6: rear wheel rotation angle
    # q7: steering rotation angle
    # q8: front wheel rotation angle
    # q9: perpendicular distance from the n2> axis to the front contact
    #     point in the ground plane
    # q10: perpendicular distance from the n1> axis to the front contact
    #     point in the ground plane
    # q11,q12: right shoulder angles
    # q13: right elbow angle
    # q14,q15: left shoulder angles
    # q16: left elbow angle

    print('Defining time varying symbols.')

    q1, q2, q3, q4 = mec.dynamicsymbols('q1 q2 q3 q4')
    q5, q6, q7, q8 = mec.dynamicsymbols('q5 q6 q7 q8')
    q11, q12, q13 = mec.dynamicsymbols('q11, q12, q13')
    q14, q15, q16 = mec.dynamicsymbols('q14, q15, q16')

    u1, u2, u3, u4 = mec.dynamicsymbols('u1 u2 u3 u4')
    u5, u6, u7, u8 = mec.dynamicsymbols('u5 u6 u7 u8')
    u11, u12, u13 = mec.dynamicsymbols('u11, u12, u13')
    u14, u15, u16 = mec.dynamicsymbols('u14, u15, u16')

    u1d, u2d, u3d, u4d = mec.dynamicsymbols('u1d u2d u3d u4d')
    u5d, u6d, u7d, u8d = mec.dynamicsymbols('u5d u6d u7d u8d')
    u11d, u12d, u13d = mec.dynamicsymbols('u11d, u12d, u13d')
    u14d, u15d, u16d = mec.dynamicsymbols('u14d, u15d, u16d')

    #################################
    # Orientation of Reference Frames
    #################################

    print('Orienting frames.')

    # The following defines a 3-1-2 Tait-Bryan rotation with yaw (q3), roll
    # (q4), pitch (q5) angles to orient the rear frame relative to the ground.
    # The front frame is then rotated through the steer angle (q7) about the
    # rear frame's 3 axis.

    # rear frame yaw
    A.orient(N, 'Axis', (q3, N['3']))
    # rear frame roll
    B.orient(A, 'Axis', (q4, A['1']))
    # rear frame pitch
    C.orient(B, 'Axis', (q5, B['2']))
    # rear wheel angle
    D.orient(C, 'Axis', (q6, C['2']))
    # front frame steer
    E.orient(C, 'Axis', (q7, C['3']))
    # front wheel angle
    F.orient(E, 'Axis', (q8, E['2']))
    # right upper arm
    G.orient_body_fixed(C, (q11, q12, 0), '232')
    # right lower arm
    H.orient_axis(G, q13, G['2'])
    # left upper arm
    I.orient_body_fixed(C, (q14, q15, 0), '232')
    # left lower arm
    J.orient_axis(I, q16, I['2'])

    ###########
    # Constants
    ###########

    print('Defining constants.')

    # geometry
    # rf: radius of front wheel
    # rr: radius of rear wheel
    # d1: the perpendicular distance from the steer axis to the center
    #     of the rear wheel (rear offset)
    # d2: the distance between wheels along the steer axis
    # d3: the perpendicular distance from the steer axis to the center
    #     of the front wheel (fork offset)
    # l1: the distance in the c1> direction from the center of the rear
    #     wheel to the frame center of mass
    # l2: the distance in the c3> direction from the center of the rear
    #     wheel to the frame center of mass
    # l3: the distance in the e1> direction from the front wheel center to
    #     the center of mass of the fork
    # l4: the distance in the e3> direction from the front wheel center to
    #     the center of mass of the fork
    # d4, d5, d6: locates right shoulder from rear wheel center
    # d7 : length of upper arm
    # d8 : length of lower arm
    # d9, d10, d11 : locates right handgrip from front wheel center
    rf, rr = sm.symbols('rf, rr')
    d1, d2, d3, d4, d5, d6 = sm.symbols('d1, d2, d3, d4, d5, d6')
    d7, d8, d9, d10, d11 = sm.symbols('d7, d8, d9, d10, d11')
    l1, l2, l3, l4 = sm.symbols('l1, l2, l3, l4')

    # acceleration due to gravity
    g = sm.symbols('g')

    # mass
    mc, md, me, mf, mg, mh, mi, mj = sm.symbols('mc, md, me, mf, mg, mh, mi, mj')

    # inertia components
    ic11, ic22, ic33, ic31 = sm.symbols('ic11, ic22, ic33, ic31')
    id11, id22 = sm.symbols('id11, id22')
    ie11, ie22, ie33, ie31 = sm.symbols('ie11, ie22, ie33, ie31')
    if11, if22 = sm.symbols('if11, if22')

    ###########
    # Specified
    ###########

    # control torques
    # T4 : roll torque
    # T6 : rear wheel torque
    # T7 : steer torque
    # T13 : right elbow torque
    # T16 : left elbow torque
    T4 = mec.dynamicsymbols('T4') if include_roll_torque else sm.S.Zero
    T6 = mec.dynamicsymbols('T6')
    if steer_with is SteerWith.STEER_TORQUE:
        T7 = mec.dynamicsymbols('T7')
    else:
        T7 = sm.S.Zero
    if steer_with is SteerWith.ELBOW_TORQUE:
        T13, T16 = mec.dynamicsymbols('T13, T16')
    else:
        T13, T16 = sm.S.Zero, sm.S.Zero

    ##################
    # Position Vectors
    ##################

    print('Defining position vectors.')

    # ground origin
    O = mec.Point('O')

    # rear wheel contact point
    dn = mec.Point('dn')
    dn.set_pos(O, q1*N['1'] + q2*N['2'])

    # rear wheel contact point to rear wheel center
    do = mec.Point('do')
    do.set_pos(dn, -rr*B['3'])

    # rear wheel center to bicycle frame center
    co = mec.Point('co')
    co.set_pos(do, l1*C['1'] + l2*C['3'])

    # rear wheel center to steer axis point
    ce = mec.Point('ce')
    ce.set_pos(do, d1*C['1'])

    ## right arm
    # rear wheel center to right shoulder
    cgr = mec.Point('cgr')
    cgr.set_pos(do, d4*C['1'] + d5*C['2'] + d6*C['3'])

    # right shoulder to elbow
    gh = mec.Point('gh')
    gh.set_pos(cgr, d7*G['3'])

    # right shoulder to upper arm mass center
    go = mec.Point('go')
    go.set_pos(cgr, d7/2*G['3'])

    # right shoulder to upper arm muscle attachment
    gm = mec.Point('gm')
    gm.set_pos(cgr, 1*d7/10*G['3'])

    # right elbow to lower arm muscle atachment
    hm = mec.Point('hm')
    hm.set_pos(gh, 2*d8/10*H['3'])

    # right elbow to lower arm mass center
    ho = mec.Point('ho')
    ho.set_pos(gh, d8/2*H['3'])

    # elbow to hand
    hc = mec.Point('hc')
    hc.set_pos(gh, d8*H['3'])

    ## left arm
    # rear wheel center to left shoulder
    cgl = mec.Point('cgl')
    cgl.set_pos(do, d4*C['1'] - d5*C['2'] + d6*C['3'])

    # left shoulder to elbow
    ji = mec.Point('ji')
    ji.set_pos(cgl, d7*I['3'])

    # left shoulder to upper arm mass center
    io = mec.Point('io')
    io.set_pos(cgl, d7/2*I['3'])

    # left shoulder to upper arm muscle attachment
    im = mec.Point('im')
    im.set_pos(cgl, 1*d7/10*I['3'])

    # left elbow to lower arm muscle attachment
    jm = mec.Point('jm')
    jm.set_pos(ji, 2*d8/10*J['3'])

    # left elbow to lower arm mass center
    jo = mec.Point('jo')
    jo.set_pos(ji, d8/2*J['3'])

    # elbow to hand
    jc = mec.Point('jc')
    jc.set_pos(ji, d8*J['3'])

    # steer axis point to the front wheel center
    fo = mec.Point('fo')
    fo.set_pos(ce, d2*E['3'] + d3*E['1'])

    # front wheel center to right handgrip
    ch_r = mec.Point('chr')
    ch_r.set_pos(ce, d9*E['1'] + d10*E['2'] + d11*E['3'])

    # front wheel center to left handgrip
    ch_l = mec.Point('chl')
    ch_l.set_pos(ce, d9*E['1'] - d10*E['2'] + d11*E['3'])

    # front wheel center to front frame center
    eo = mec.Point('eo')
    eo.set_pos(fo, l3*E['1'] + l4*E['3'])

    # front wheel contact point
    fn = mec.Point('fn')
    fn.set_pos(fo, rf*E['2'].cross(A['3']).cross(E['2']).normalize())

    ######################
    # Holonomic Constraint
    ######################

    print('Defining holonomic constraints.')

    # this constraint is enforced so that the front wheel contacts the ground
    holonomic_wheel = sm.Matrix([fn.pos_from(dn).dot(A['3'])])
    holonomic_handr = (hc.pos_from(co) - ch_r.pos_from(co)).to_matrix(C)
    holonomic_handl = (jc.pos_from(co) - ch_l.pos_from(co)).to_matrix(C)
    holonomic = holonomic_wheel.col_join(holonomic_handr).col_join(holonomic_handl)

    print('The holonomic constraint is a function of these dynamic variables:')
    print(list(sm.ordered(mec.find_dynamicsymbols(holonomic))))

    ####################################
    # Kinematical Differential Equations
    ####################################

    print('Defining kinematical differential equations.')

    kinematical = [
        q1.diff(t) - u1,  # rear wheel contact x location
        q2.diff(t) - u2,  # rear wheel contact y location
        q3.diff(t) - u3,  # yaw
        q4.diff(t) - u4,  # roll
        q5.diff(t) - u5,  # pitch
        q6.diff(t) - u6,  # rear wheel rotation
        q7.diff(t) - u7,  # steer
        q8.diff(t) - u8,  # front wheel rotation
        q11.diff(t) - u11,  # right shoulder extension
        q12.diff(t) - u12,  # right shoulder rotation
        q13.diff(t) - u13,  # right elbow extension
        q14.diff(t) - u14,  # left shoulder extension
        q15.diff(t) - u15,  # left shoulder rotation
        q16.diff(t) - u16,  # left elbow extension
    ]

    ####################
    # Angular Velocities
    ####################

    print('Defining angular velocities.')

    # Note that the wheel angular velocities are defined relative to the frame
    # they are attached to.

    A.set_ang_vel(N, u3*N['3'])  # yaw rate
    B.set_ang_vel(A, u4*A['1'])  # roll rate
    C.set_ang_vel(B, u5*B['2'])  # pitch rate
    D.set_ang_vel(C, u6*C['2'])  # rear wheel rate
    E.set_ang_vel(C, u7*C['3'])  # steer rate
    F.set_ang_vel(E, u8*E['2'])  # front wheel rate
    G.set_ang_vel(C, u11*C['2'] + u12*G['3'])
    H.set_ang_vel(G, u13*G['2'])
    I.set_ang_vel(C, u14*C['2'] + u15*I['3'])
    J.set_ang_vel(I, u16*I['2'])

    ###################
    # Linear Velocities
    ###################

    print('Defining linear velocities.')

    O.set_vel(N, 0)

    # rear wheel contact stays in ground plane and does not slip
    dn.set_vel(N, u1*N['1'] + u2*N['2'])

    # mass centers
    do.set_vel(N, do.pos_from(O).dt(N).xreplace({q1.diff(): u1, q2.diff(): u2}))
    co.v2pt_theory(do, N, C)
    ce.v2pt_theory(do, N, C)
    fo.v2pt_theory(ce, N, E)
    eo.v2pt_theory(fo, N, E)
    go.v2pt_theory(cgr, N, G)
    ho.v2pt_theory(gh, N, H)
    io.v2pt_theory(cgl, N, I)
    jo.v2pt_theory(ji, N, J)

    # arm & handlebar joints
    cgr.v2pt_theory(co, N, C)
    gh.v2pt_theory(cgr, N, G)
    hc.v2pt_theory(gh, N, H)
    ch_r.v2pt_theory(fo, N, F)

    cgl.v2pt_theory(co, N, C)
    ji.v2pt_theory(cgl, N, I)
    jc.v2pt_theory(ji, N, J)
    ch_l.v2pt_theory(fo, N, F)

    # rear and front wheel contact velocity
    N_v_dn = do.vel(N) + D.ang_vel_in(N).cross(dn.pos_from(do))
    N_v_fn = fo.vel(N) + F.ang_vel_in(N).cross(fn.pos_from(fo))

    ####################
    # Motion Constraints
    ####################

    print('Defining nonholonomic constraints.')

    nonholonomic = sm.Matrix([
        N_v_dn.dot(A['1']),
        N_v_dn.dot(A['2']),
        N_v_fn.dot(A['1']),
        N_v_fn.dot(A['3']),
        N_v_fn.dot(A['2']),
        holonomic_handr[0].diff(t),
        holonomic_handr[1].diff(t),
        holonomic_handr[2].diff(t),
        holonomic_handl[0].diff(t),
        holonomic_handl[1].diff(t),
        holonomic_handl[2].diff(t),
    ])



    #########
    # Inertia
    #########

    print('Defining inertia.')

    # NOTE : You cannot define the wheel inertias with respect to their
    # respective frames because the generalized inertia force calcs will fail
    # because there is no direction cosine matrix relating the wheel frames
    # back to the other reference frames so I define them here with respect to
    # the rear and front frames.

    # NOTE : Changing 0.0 to 0 or sm.S(0) changes the floating point errors.

    Ic = mec.inertia(C, ic11, ic22, ic33, 0.0, 0.0, ic31)
    Id = mec.inertia(C, id11, id22, id11, 0.0, 0.0, 0.0)
    Ie = mec.inertia(E, ie11, ie22, ie33, 0.0, 0.0, ie31)
    If = mec.inertia(E, if11, if22, if11, 0.0, 0.0, 0.0)
    Ig = mec.inertia(G, mg/12*d7**2, mg/12*d7**2, mg/2*(d7/10)**2)
    Ih = mec.inertia(H, mh/12*d8**2, mh/12*d8**2, mh/2*(d8/10)**2)
    Ii = mec.inertia(I, mi/12*d7**2, mi/12*d7**2, mi/2*(d7/10)**2)
    Ij = mec.inertia(J, mj/12*d8**2, mj/12*d8**2, mj/2*(d8/10)**2)

    ##############
    # Rigid Bodies
    ##############

    print('Defining the rigid bodies.')

    rear_frame = mec.RigidBody('Rear Frame', co, C, mc, (Ic, co))
    rear_wheel = mec.RigidBody('Rear Wheel', do, D, md, (Id, do))
    front_frame = mec.RigidBody('Front Frame', eo, E, me, (Ie, eo))
    front_wheel = mec.RigidBody('Front Wheel', fo, F, mf, (If, fo))
    rupper_arm = mec.RigidBody('Right Upper Arm', go, G, mg, (Ig, go))
    rlower_arm = mec.RigidBody('Right Lower Arm', ho, H, mh, (Ih, ho))
    lupper_arm = mec.RigidBody('Left Upper Arm', io, I, mi, (Ii, io))
    llower_arm = mec.RigidBody('Left Lower Arm', jo, J, mj, (Ij, jo))

    bodies = [rear_frame, rear_wheel, front_frame, front_wheel, rupper_arm,
              rlower_arm, lupper_arm, llower_arm]

    if steer_with is SteerWith.MUSCLES:

        ################
        # Musculotendons
        ################

        print('Defining the musculotendons')

        F_M_max_bicep, F_M_max_tricep = sm.symbols('F_M_max_bicep, F_M_max_tricep')
        l_M_opt_bicep, l_M_opt_tricep = sm.symbols('l_M_opt_bicep, l_M_opt_tricep')
        l_T_slack_bicep, l_T_slack_tricep = sm.symbols('l_T_slack_bicep, l_T_slack_tricep')
        v_M_max, alpha_opt, beta = sm.symbols('v_M_max, alpha_opt, beta')
        tau_a, tau_d, b_tanh = sm.symbols('tau_a, tau_d, b_tanh')

        bicep_right_pathway = mec.LinearPathway(gm, hm)
        bicep_right_activation = FirstOrderActivationDeGroote2016(
            'bi_r',
            activation_time_constant=tau_a,
            deactivation_time_constant=tau_d,
            smoothing_rate=b_tanh,
        )
        bicep_right = MusculotendonDeGroote2016(
            'bi_r',
            bicep_right_pathway,
            activation_dynamics=bicep_right_activation,
            tendon_slack_length=l_T_slack_bicep,
            peak_isometric_force=F_M_max_bicep,
            optimal_fiber_length=l_M_opt_bicep,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )

        bicep_left_pathway = mec.LinearPathway(im, jm)
        bicep_left_activation = FirstOrderActivationDeGroote2016(
            'bi_l',
            activation_time_constant=tau_a,
            deactivation_time_constant=tau_d,
            smoothing_rate=b_tanh,
        )
        bicep_left = MusculotendonDeGroote2016(
            'bi_l',
            bicep_left_pathway,
            activation_dynamics=bicep_left_activation,
            tendon_slack_length=l_T_slack_bicep,
            peak_isometric_force=F_M_max_bicep,
            optimal_fiber_length=l_M_opt_bicep,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )

        tricep_right_pathway = ExtensorPathway(gm, hm, gh, G['2'], -G['3'],
                                               H['3'], d8/10, q13)
        tricep_right_activation = FirstOrderActivationDeGroote2016(
            'tri_r',
            activation_time_constant=tau_a,
            deactivation_time_constant=tau_d,
            smoothing_rate=b_tanh,
        )
        tricep_right = MusculotendonDeGroote2016(
            'tri_r',
            tricep_right_pathway,
            activation_dynamics=tricep_right_activation,
            tendon_slack_length=l_T_slack_tricep,
            peak_isometric_force=F_M_max_tricep,
            optimal_fiber_length=l_M_opt_tricep,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )

        tricep_left_pathway = ExtensorPathway(im, jm, ji, I['2'], -I['3'],
                                              J['3'], d8/10, q16)
        tricep_left_activation = FirstOrderActivationDeGroote2016(
            'tri_l',
            activation_time_constant=tau_a,
            deactivation_time_constant=tau_d,
            smoothing_rate=b_tanh,
        )
        tricep_left = MusculotendonDeGroote2016(
            'tri_l',
            tricep_left_pathway,
            activation_dynamics=tricep_left_activation,
            tendon_slack_length=l_T_slack_tricep,
            peak_isometric_force=F_M_max_tricep,
            optimal_fiber_length=l_M_opt_tricep,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )

        musculotendons = [bicep_right, bicep_left, tricep_right, tricep_left]

    ###########################
    # Generalized Active Forces
    ###########################

    print('Defining the forces and torques.')

    # gravity
    Fco = (co, mc*g*A['3'])
    Fdo = (do, md*g*A['3'])
    Feo = (eo, me*g*A['3'])
    Ffo = (fo, mf*g*A['3'])
    Fgo = (go, mg*g*A['3'])
    Fho = (ho, mh*g*A['3'])
    Fio = (io, mi*g*A['3'])
    Fjo = (jo, mj*g*A['3'])

    # input torques
    Tc = (C, T4*A['1'] - T6*B['2'] - T7*C['3'])
    Td = (D, T6*C['2'])
    Te = (E, T7*C['3'])
    Tg = (G, -T13*G['2'])
    Th = (H, T13*G['2'])
    Ti = (I, -T16*I['2'])
    Tj = (J, T16*I['2'])

    forces = [Fco, Fdo, Feo, Ffo, Fgo, Fho, Fio, Fjo]
    if include_roll_torque:
        forces.append(Tc)
    forces.append(Td)
    if steer_with is SteerWith.STEER_TORQUE:
        forces.append(Te)
    elif steer_with is SteerWith.ELBOW_TORQUE:
        forces.extend([Tg, Th, Ti, Tj])
    elif steer_with is SteerWith.MUSCLES:
        for musculotendon in musculotendons:
            forces.extend(musculotendon.to_loads())

    # Manually compute the ground contact velocities.
    kindiffeqs = [q1.diff(t), q2.diff(t), q3.diff(t), q4.diff(t), q5.diff(t),
                  q6.diff(t), q7.diff(t), q8.diff(t), q11.diff(t), q12.diff(t),
                  q13.diff(t), q14.diff(t), q15.diff(t), q16.diff(t)]
    kindiffdict = sm.solve(kinematical, kindiffeqs, dict=True)[0]
    nonholonomic = nonholonomic.xreplace(kindiffdict)
    print('The nonholonomic constraints are a function of these dynamic variables:')
    print(list(sm.ordered(mec.find_dynamicsymbols(sm.Matrix(nonholonomic)))))

    ###############################
    # Prep symbolic data for output
    ###############################

    q_ind = (q1, q2, q3, q4, q6, q7, q8)  # yaw, roll, steer
    q_dep = (q5, q11, q12, q13, q14, q15, q16)  # pitch
    q_ign = None
    u_ind = (u4, u6, u7)  # roll rate, rear wheel rate, steer rate
    u_dep = (u1, u2, u3, u5, u8, u11, u12, u13, u14, u15, u16)  # yaw rate, pitch rate, front wheel rate
    p = sm.Matrix([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, g, ic11, ic22,
                   ic31, ic33, id11, id22, ie11, ie22, ie31, ie33, if11, if22,
                   l1, l2, l3, l4, mc, md, me, mf, mg, mh, mi, mj, rf, rr])

    if steer_with is SteerWith.MUSCLES:
        p_muscles = sm.Matrix([
            F_M_max_bicep,
            l_M_opt_bicep,
            l_T_slack_bicep,
            F_M_max_tricep,
            l_M_opt_tricep,
            l_T_slack_tricep,
            v_M_max,
            alpha_opt,
            beta,
            tau_a,
            tau_d,
            b_tanh,
        ])
        p = p.col_join(p_muscles)

        e = sm.Matrix.vstack(*[m.r for m in musculotendons])
        a = sm.Matrix.vstack(*[m.x for m in musculotendons])
        ad = sm.Matrix.vstack(*[m.rhs() for m in musculotendons])

    T = sm.Matrix([T4, T6]) if include_roll_torque else sm.Matrix([T6])
    if steer_with is SteerWith.MUSCLES:
        r = sm.Matrix.vstack(T, e)
    elif steer_with is SteerWith.ELBOW_TORQUE:
        r = sm.Matrix.vstack(T, sm.Matrix([T13, T16]))
    elif steer_with is SteerWith.STEER_TORQUE:
        r = sm.Matrix.vstack(T, sm.Matrix([T7]))

    q = sm.Matrix([q1, q2, q3, q4, q5, q6, q7, q8, q11, q12, q13, q14, q15, q16])
    u = sm.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u11, u12, u13, u14, u15, u16])

    points = (
        O,
        dn,  # rear contact
        do,  # rear wheel center
        cgr,  # right shoulder
        go,
        gm,
        gh,  # right elbow
        hm,
        ho,
        hc,  # right hand
        ch_r,
        ch_l,
        jc,  # left hand
        jo,
        jm,
        ji,
        im,
        io,
        cgl,
        do,  # rear wheel center
        ce,  # steer axis
        fo,  # front wheel center
        fn,
    )

    ###############
    # Kane's Method
    ###############

    print("Generating Kane's equations.")

    kane = mec.KanesMethod(
        N,
        q_ind,
        u_ind,
        kd_eqs=kinematical,
        q_dependent=q_dep,
        configuration_constraints=holonomic,
        u_dependent=u_dep,
        velocity_constraints=nonholonomic,
        constraint_solver='CRAMER',
    )

    Fr, Frs = kane.kanes_equations(bodies, loads=forces)

    x = q.col_join(u)
    x_all = kane.q.col_join(kane.u)
    eom = kane.mass_matrix_full*x_all - kane.forcing_full

    if steer_with is SteerWith.MUSCLES:
        x = x.col_join(a)
        eom = eom.col_join(ad)

    return ForOpty(t, x, r, eom, p)


def constants_values(steer_with=SteerWith.MUSCLES):

    bike_p_vals = np.array([
        0.9534570696121849,  # d1
        0.2676445084476887,  # d2
        0.03207142672761929,  # d3
        0.8,  # d4
        0.2,  # d5, shoulder half width
        -1.0,  # d6
        0.3,  # d7, upper arm length
        0.35,  # d8, lower arm length
        0.06,  # d9
        0.25,  # d10, handlebar half width
        -0.5,  # d11
        9.81,  # g
        7.178169776497895,  # ic11
        11.0,  # ic22
        3.8225535938357873,  # ic31
        4.821830223502103,  # ic33
        0.0603,  # id11
        0.12,  # id22
        0.05841337700152972,  # ie11
        0.06,  # ie22
        0.009119225261946298,  # ie31
        0.007586622998470264,  # ie33
        0.1405,  # if11
        0.28,  # if22
        0.4707271515135145,  # l1
        -0.47792881146460797,  # l2
        -0.00597083392418685,  # l3
        -0.3699518200282974,  # l4
        85.0,  # mc
        2.0,  # md
        4.0,  # me
        3.0,  # mf
        2.3,  # mg
        1.7,  # mh
        2.3,  # mi
        1.7,  # mj
        0.35,  # rf
        0.3,  # rr
    ])

    muscle_p_vals = np.array([
        500.0,  # F_M_max_bicep
        0.18,  # l_M_opt_bicep
        0.17,  # l_T_slack_bicep
        500.0,  # F_M_max_tricep
        0.18,  # l_M_opt_tricep
        0.19,  # l_T_slack_tricep
        10.0,  # v_M_max
        0.0,  # alpha_opt
        0.1,  # beta
        0.015,  # tau_a
        0.060,  # tau_d
        10.0,  # b_tanh
    ])

    if steer_with is SteerWith.MUSCLES:
        return np.hstack((bike_p_vals, muscle_p_vals))
    else:
        return bike_p_vals
