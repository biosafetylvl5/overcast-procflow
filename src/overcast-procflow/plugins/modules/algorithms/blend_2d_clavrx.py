"""OVERCAST Blending for CLAVR-X.

This module-based plugin provides a wrapper to blend 2D CLAVR-X data using
a specified weighting function. It utilizes functionality from
`overcast_preprocessing.blending`.
"""

import logging

from overcast_preprocessing import blending

LOG = logging.getLogger(__name__)

# Specify internal data used by GeoIPS to create procflows
interface = "algorithms"
family = "xarray_to_xarray"
name = "blend_2d_clavrx"


def call(
    xobj,
    weight_function=blending.hard_cutoff_weights,
    cloud_phase_weight_function=None,
    **kwargs,
):
    """Blend 2D CLAVR-X data using a specified weighting function.

    Parameters
    ----------
    xobj : xarray.Dataset or xarray.DataArray
        The xarray object containing CLAVR-X data to be blended.
    weight_function : callable, optional
        The function that calculates weights for blending. Defaults
        to `blending.hard_cutoff_weights`.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The blended xarray object, with the same type as the input.
    """
    if type(weight_function) == str:
        weight_function = getattr(blending, weight_function)
    if type(cloud_phase_weight_function) == str:
        cloud_phase_weight_function = getattr(blending, cloud_phase_weight_function)
    # Perform the 2D blend using the specified weighting function.
    # The actual blending logic is handled by the blend_2d_clavrx function.
    print(xobj, weight_function, cloud_phase_weight_function)
    return blending.blend_2d_clavrx(
        xobj,
        weight_function=weight_function,
        cloud_phase_weight_function=cloud_phase_weight_function,
        **kwargs,
    )
