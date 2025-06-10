"""OVERCAST 3D Cloud Field Generator for CLAVR-X."""

import logging

from overcast_preprocessing import construct_3d_field

LOG = logging.getLogger(__name__)

interface = "algorithms"
family = "xarray_to_xarray"
name = "construct_3d_cloud_field"


def call(xobj, number_of_levels=41, max_height=20):
    return construct_3d_field.compute_3d_clavrx_cloud_phase(
        xobj,
        number_of_levels=number_of_levels,
        max_height=max_height,
    )
