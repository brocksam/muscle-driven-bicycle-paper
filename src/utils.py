import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import numpy as np


class ReferenceFrame(me.ReferenceFrame):
    """Subclass that enforces the desired unit vector indice style."""

    def __init__(self, *args, **kwargs):

        kwargs.pop('indices', None)
        kwargs.pop('latexs', None)

        lab = args[0].lower()
        tex = r'\hat{{{}}}_{}'

        super(ReferenceFrame, self).__init__(*args, indices=('1', '2', '3'),
                                             latexs=(tex.format(lab, '1'),
                                                     tex.format(lab, '2'),
                                                     tex.format(lab, '3')),
                                             **kwargs)


class ExtensorPathway(me.PathwayBase):

    def __init__(self, origin, insertion, axis_point, axis, parent_axis,
                 child_axis, radius, coordinate):
        """A custom pathway that wraps a circular arc around a pin joint.

        This is intended to be used for extensor muscles. For example, a
        triceps wrapping around the elbow joint to extend the upper arm at
        the elbow.

        Parameters
        ==========
        origin : Point
            Muscle origin point fixed on the parent body (A).
        insertion : Point
            Muscle insertion point fixed on the child body (B).
        axis_point : Point
            Pin joint location fixed in both the parent and child.
        axis : Vector
            Pin joint rotation axis.
        parent_axis : Vector
            Axis fixed in the parent frame (A) that is directed from the pin
            joint point to the muscle origin point.
        child_axis : Vector
            Axis fixed in the child frame (B) that is directed from the pin
            joint point to the muscle insertion point.
        radius : sympyfiable
            Radius of the arc that the muscle wraps around.
        coordinate : sympfiable function of time
            Joint angle, zero when parent and child frames align. Positive
            rotation about the pin joint axis, B with respect to A.

        Notes
        =====

        Only valid for coordinate >= 0.

        """
        super().__init__(origin, insertion)

        self.origin = origin
        self.insertion = insertion
        self.axis_point = axis_point
        self.axis = axis.normalize()
        self.parent_axis = parent_axis.normalize()
        self.child_axis = child_axis.normalize()
        self.radius = radius
        self.coordinate = coordinate

        self.origin_distance = axis_point.pos_from(origin).magnitude()
        self.insertion_distance = axis_point.pos_from(insertion).magnitude()
        self.origin_angle = sm.asin(self.radius/self.origin_distance)
        self.insertion_angle = sm.asin(self.radius/self.insertion_distance)

    @property
    def length(self):
        """Length of the pathway.

        Length of two fixed length line segments and a changing arc length
        of a circle.

        """

        angle = self.origin_angle + self.coordinate + self.insertion_angle
        arc_length = self.radius*angle

        origin_segment_length = self.origin_distance*sm.cos(self.origin_angle)
        insertion_segment_length = (self.insertion_distance *
                                    sm.cos(self.insertion_angle))

        return origin_segment_length + arc_length + insertion_segment_length

    @property
    def extension_velocity(self):
        """Extension velocity of the pathway.

        Arc length of circle is the only thing that changes when the elbow
        flexes and extends.

        """
        return self.radius*self.coordinate.diff(me.dynamicsymbols._t)

    def to_loads(self, force_magnitude):
        """Loads in the correct format to be supplied to `KanesMethod`.

        Forces applied to origin, insertion, and P from the muscle wrapped
        over circular arc of radius r.

        """

        parent_tangency_point = me.Point('Aw')  # fixed in parent
        child_tangency_point = me.Point('Bw')  # fixed in child

        parent_tangency_point.set_pos(
            self.axis_point,
            -self.radius*sm.cos(self.origin_angle)*self.parent_axis.cross(self.axis)
            + self.radius*sm.sin(self.origin_angle)*self.parent_axis,
        )
        child_tangency_point.set_pos(
            self.axis_point,
            self.radius*sm.cos(self.insertion_angle)*self.child_axis.cross(self.axis)
            + self.radius*sm.sin(self.insertion_angle)*self.child_axis),

        parent_force_direction_vector = self.origin.pos_from(parent_tangency_point)
        child_force_direction_vector = self.insertion.pos_from(child_tangency_point)
        force_on_parent = force_magnitude*parent_force_direction_vector.normalize()
        force_on_child = force_magnitude*child_force_direction_vector.normalize()
        loads = [
            me.Force(self.origin, force_on_parent),
            me.Force(self.axis_point, -(force_on_parent + force_on_child)),
            me.Force(self.insertion, force_on_child),
        ]
        return loads


def plot_trajectories(t, x, r, state_syms, input_syms, skip_first=False):

    q = x[0:14].copy()
    u = x[14:28].copy()
    if skip_first:
        q[:, 0] = np.nan
        u[:, 0] = np.nan
        r[:, 0] = np.nan
    q_sym = state_syms[0:14]
    u_sym = state_syms[14:28]

    qfig, qaxes = plt.subplots(7, 2, sharex=True)
    for i, (qi, ax, q_symi) in enumerate(zip(q, qaxes.T.flatten(), q_sym)):
        if i > 1:
            ax.plot(t, np.rad2deg(qi))
        else:
            ax.plot(t, qi)
        ax.set_ylabel(sm.latex(q_symi, mode='inline'))
    qaxes[-1, 0].set_xlabel('Time [s]')
    qaxes[-1, 1].set_xlabel('Time [s]')
    qfig.tight_layout()

    ufig, uaxes = plt.subplots(7, 2, sharex=True)
    for i, (ui, ax, u_symi) in enumerate(zip(u, uaxes.T.flatten(), u_sym)):
        if i > 1:
            ax.plot(t, np.rad2deg(ui))
        else:
            ax.plot(t, ui)
        ax.set_ylabel(sm.latex(u_symi, mode='inline'))
    uaxes[-1, 0].set_xlabel('Time [s]')
    uaxes[-1, 1].set_xlabel('Time [s]')
    ufig.tight_layout()

    rfig, raxes = plt.subplots(len(input_syms), 1, sharex=True)
    for ri, ax, r_symi in zip(r, raxes, input_syms):
        ax.plot(t, ri)
        ax.set_ylabel(sm.latex(r_symi, mode='inline'))
    ax.set_xlabel('Time [s]')
    rfig.tight_layout()
