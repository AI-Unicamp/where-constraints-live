"""Inverse CSG with typed slots (Chapter 3 of the dissertation).

Building blocks, from points to a shape occupancy:

1. :func:`to_local` - points in each primitive's frame;
2. :func:`primitive_sdf` - signed distance per primitive type;
3. :func:`volume_radius` and :func:`sdf_to_occupancy` - calibrated soft occupancy;
4. :func:`type_weights` and :func:`combine_types` - typed-slot competition
   (product union over the types of a slot);
5. an assembler (:func:`nary_dnf`, :func:`csg_stump`, :func:`csg_stump_minmax`,
   :func:`binary_tree`) - the Boolean program over the slots.

:class:`TypedSlots` bundles the learnable slot parameters and steps 1-4.
"""
from .primitives import TYPES, primitive_sdf, sphere_sdf, box_sdf, cylinder_sdf, wedge_sdf, corner_wedge_sdf
from .volume import primitive_volume, volume_radius
from .fields import euler_to_rotation, to_local, sdf_to_occupancy
from .types import SELECTORS, type_weights, combine_types, onehot_occupancy, multihot_mask
from .assemblers import nary_dnf, csg_stump, csg_stump_minmax, binary_tree
from .slots import TypedSlots
