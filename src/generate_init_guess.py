import numpy as np
import sympy as sm
import sympy.physics.mechanics as mec


def gen_init_guess_for_opty(model, problem, metadata):

    guesses = []
    for var in sm.Matrix.vstack(model.x, model.r):
        guesses.append(DISPATCHER[var](model, problem, metadata) + 0.01*np.random.random())

    init_guess = np.concatenate(guesses)

    return init_guess


def _q1_guess(model, problem, metadata):
    """q1: perpendicular distance from the n2> axis to the rear contact point
    in the ground plane."""
    return np.linspace(0.0, metadata.longitudinal_displacement, metadata.num_nodes)


def _q2_guess(model, problem, metadata):
    """q2: perpendicular distance from the n1> axis to the rear contact point
    in the ground plane."""
    return metadata.target_q2(np.linspace(0.0, metadata.longitudinal_displacement, metadata.num_nodes))


def _q3_guess(model, problem, metadata):
    """q3: frame yaw angle."""
    return np.arctan(_q2_guess(model, problem, metadata) / (_q1_guess(model, problem, metadata) + 0.01*np.random.random()))


def _q4_guess(model, problem, metadata):
    """q4: frame roll angle."""
    return 0.01*np.ones(metadata.num_nodes)


def _q5_guess(model, problem, metadata):
    """q5: frame pitch angle."""
    return 0.01*np.ones(metadata.num_nodes)


def _q6_guess(model, problem, metadata):
    """q6: rear wheel rotation angle."""
    return -_q1_guess(model, problem, metadata) / (5.0 / metadata.constants[-1])


def _q7_guess(model, problem, metadata):
    """q7: steering rotation angle."""
    return 0.01*np.ones(metadata.num_nodes)


def _q8_guess(model, problem, metadata):
    """q8: front wheel rotation angle."""
    return -_q1_guess(model, problem, metadata) / (5.0 / metadata.constants[-2])


def _q11_guess(model, problem, metadata):
    """q11: first (swing) right shoulder angle."""
    return -0.21607176*np.ones(metadata.num_nodes)


def _q12_guess(model, problem, metadata):
    """q12: second (rotation) right shoulder angle."""
    return 0.15706218*np.ones(metadata.num_nodes)


def _q13_guess(model, problem, metadata):
    """q13: right elbow angle."""
    return 1.15113191*np.ones(metadata.num_nodes)


def _q14_guess(model, problem, metadata):
    """q14: first (swing) left shoulder angle."""
    return -0.21607176*np.ones(metadata.num_nodes)


def _q15_guess(model, problem, metadata):
    """q15: second (rotation) left shoulder angle."""
    return -0.15706218*np.ones(metadata.num_nodes)


def _q16_guess(model, problem, metadata):
    """q16: left elbow angle."""
    return 1.15113191*np.ones(metadata.num_nodes)


def _u1_guess(model, problem, metadata):
    """u1: perpendicular distance from the n2> axis to the rear contact point
    in the ground plane."""
    return np.gradient(_q1_guess(model, problem, metadata))


def _u2_guess(model, problem, metadata):
    """u2: perpendicular distance from the n1> axis to the rear contact point
    in the ground plane."""
    return np.gradient(_q2_guess(model, problem, metadata))


def _u3_guess(model, problem, metadata):
    """u3: frame yaw angle."""
    return np.gradient(_q3_guess(model, problem, metadata))


def _u4_guess(model, problem, metadata):
    """u4: frame roll angle."""
    return np.gradient(_q4_guess(model, problem, metadata))


def _u5_guess(model, problem, metadata):
    """u5: frame pitch angle."""
    return np.gradient(_q5_guess(model, problem, metadata))


def _u6_guess(model, problem, metadata):
    """u6: rear wheel rotation angle."""
    return np.gradient(_q6_guess(model, problem, metadata))


def _u7_guess(model, problem, metadata):
    """u7: steering rotation angle."""
    return np.gradient(_q7_guess(model, problem, metadata))


def _u8_guess(model, problem, metadata):
    """u8: front wheel rotation angle."""
    return np.gradient(_q8_guess(model, problem, metadata))


def _u11_guess(model, problem, metadata):
    """u11: first (swing) right shoulder angle."""
    return np.gradient(_q11_guess(model, problem, metadata))


def _u12_guess(model, problem, metadata):
    """u12: second (rotation) right shoulder angle."""
    return np.gradient(_q12_guess(model, problem, metadata))


def _u13_guess(model, problem, metadata):
    """u13: right elbow angle."""
    return np.gradient(_q13_guess(model, problem, metadata))


def _u14_guess(model, problem, metadata):
    """u14: first (swing) left shoulder angle."""
    return np.gradient(_q14_guess(model, problem, metadata))


def _u15_guess(model, problem, metadata):
    """u15: second (rotation) left shoulder angle."""
    return np.gradient(_q15_guess(model, problem, metadata))


def _u16_guess(model, problem, metadata):
    """u16: left elbow angle."""
    return np.gradient(_q16_guess(model, problem, metadata))


def _T4_guess(model, problem, metadata):
    """."""
    return 0.01*np.ones(metadata.num_nodes)


def _T6_guess(model, problem, metadata):
    """."""
    return 0.01*np.ones(metadata.num_nodes)


def _T7_guess(model, problem, metadata):
    """."""
    return 0.01*np.ones(metadata.num_nodes)


DISPATCHER = {
    mec.dynamicsymbols('q1'): _q1_guess,
    mec.dynamicsymbols('q2'): _q2_guess,
    mec.dynamicsymbols('q3'): _q3_guess,
    mec.dynamicsymbols('q4'): _q4_guess,
    mec.dynamicsymbols('q5'): _q5_guess,
    mec.dynamicsymbols('q6'): _q6_guess,
    mec.dynamicsymbols('q7'): _q7_guess,
    mec.dynamicsymbols('q8'): _q8_guess,
    mec.dynamicsymbols('q11'): _q11_guess,
    mec.dynamicsymbols('q12'): _q12_guess,
    mec.dynamicsymbols('q13'): _q13_guess,
    mec.dynamicsymbols('q14'): _q14_guess,
    mec.dynamicsymbols('q15'): _q15_guess,
    mec.dynamicsymbols('q16'): _q16_guess,
    mec.dynamicsymbols('u1'): _u1_guess,
    mec.dynamicsymbols('u2'): _u2_guess,
    mec.dynamicsymbols('u3'): _u3_guess,
    mec.dynamicsymbols('u4'): _u4_guess,
    mec.dynamicsymbols('u5'): _u5_guess,
    mec.dynamicsymbols('u6'): _u6_guess,
    mec.dynamicsymbols('u7'): _u7_guess,
    mec.dynamicsymbols('u8'): _u8_guess,
    mec.dynamicsymbols('u11'): _u11_guess,
    mec.dynamicsymbols('u12'): _u12_guess,
    mec.dynamicsymbols('u13'): _u13_guess,
    mec.dynamicsymbols('u14'): _u14_guess,
    mec.dynamicsymbols('u15'): _u15_guess,
    mec.dynamicsymbols('u16'): _u16_guess,
    mec.dynamicsymbols('T4'): _T4_guess,
    mec.dynamicsymbols('T6'): _T6_guess,
    mec.dynamicsymbols('T7'): _T7_guess,
}
