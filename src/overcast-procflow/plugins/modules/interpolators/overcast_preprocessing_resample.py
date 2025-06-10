"""OVERCAST Preprocessing Interpolation for CLAVR-X."""

### WARNING: THIS FILE SHOULD DISAPPEAR! PLEASE CRY IF IT MAKES IT INTO THE CODEBASE
import logging
import os

from overcast_preprocessing import resampling

LOG = logging.getLogger(__name__)

interface = "interpolators"
family = "xarray_to_xarray"
name = "overcast_preprocessing_resample"


def call(data, use_cache=False, cache_key="", cache_location=None):
    # Generate a destination grid with a grid spacing of 0.02 degrees
    N_PROCESSORS = 40
    geolims = resampling.GeoLimits(lonmin=-180, lonmax=180, latmin=-70, latmax=70)
    destination_grid = resampling.generate_global_swath(geolims, 0.02)

    if use_cache:
        if not cache_location:
            pass  # TODO: Implement caching

        NN_N_NEIGHBORS = 1
        GAUSS_N_NEIGHBORS = 4
        NN_CACHE_FILENAME = f"{cache_key}-nn_resample_cache.pkl"
        GAUSS_CACHE_FILENAME = f"{cache_key}-gauss_resample_cache.pkl"
        if not all(map(os.path.exists, [GAUSS_CACHE_FILENAME, NN_CACHE_FILENAME])):
            # Precompute and save nearest neighbor resampling
            resampling.save_resampling_info(
                data,
                NN_N_NEIGHBORS,
                NN_CACHE_FILENAME,
                global_grid=destination_grid,
                nprocs=N_PROCESSORS,
            )
            # Precompute and save Gaussian resampling
            resampling.save_resampling_info(
                data,
                GAUSS_N_NEIGHBORS,
                GAUSS_CACHE_FILENAME,
                global_grid=destination_grid,
                nprocs=N_PROCESSORS,
            )
        return resampling.resample_clavrx(
            data,
            global_grid=destination_grid,
            nprocs=N_PROCESSORS,
            precomp_nn_resample_info=NN_CACHE_FILENAME,
            precomp_gauss_resample_info=GAUSS_CACHE_FILENAME,
        )

    else:
        return resampling.resample_clavrx(
            data,
            global_grid=destination_grid,
            nprocs=N_PROCESSORS,
        )
