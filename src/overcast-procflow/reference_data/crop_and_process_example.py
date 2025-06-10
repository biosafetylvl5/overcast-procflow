import matplotlib.pyplot as plt
import xarray as xr
from clavrx_input import hdf4_input
from overcast_preprocessing import (
    construct_3d_field,
    parallax_correction,
    resampling,
    sanitization,
)

CLAVRX_VARS = [
    "sensor_azimuth_angle",
    "sensor_zenith_angle",
    "surface_elevation",
    "cloud_phase",
    "cld_height_base",
    "cld_height_acha",
    "cld_cwp_dcomp",
    "freezing_altitude",
    "freezing_altitude_253",
    "freezing_altitude_268",
]

full_ds = hdf4_input.read(
    "/mnt/overcastnas1/GEO_clavrx/GOES16_ABI/RadF/output/2024053/"
    "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20240532000204.level2.hdf",  # type: ignore
    var_names=CLAVRX_VARS,
)
cwc_path = (
    "/mnt/overcastnas1/cwhite/cwc_golden_cases/2024053_2000/"
    "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20240532000204.CWC.h5"
)

cropped_ds = full_ds.isel(y=range(3000, 3100), x=range(0, 100))
cropped_ds.to_netcdf("cropped_raw_input.nc")

sanitized_ds = sanitization.sanitize_clavrx(cropped_ds)
sanitized_ds.to_netcdf("cropped_sanitized.nc")

geo_limits = {"latmin": -7.5, "latmax": -6, "lonmin": -151, "lonmax": -144}
geolim_clipped_ds = sanitization.clip_clavrx_to_geolims(cropped_ds, geo_limits)
geolim_clipped_ds.to_netcdf("cropped_clipped.nc")

pc_corrected_ds = parallax_correction.parallax_correct_clavrx(sanitized_ds)
pc_corrected_ds.to_netcdf("cropped_parallax_corrected.nc")

cwc_ds = xr.open_dataset(cwc_path)
cwc_profiles = cwc_ds["standardized_profiles"].rename(
    {"phony_dim_0": "y", "phony_dim_1": "x", "phony_dim_2": "levels"},
)
cropped_cwc = cwc_profiles.isel(y=range(3000, 3100), x=range(0, 100))
cwc_added_ds = pc_corrected_ds.assign({"cwc_standardized_profiles": cropped_cwc})
cwc_added_ds.to_netcdf("cropped_pc_with_cwc.nc")

test_geo_lims = resampling.GeoLimits(lonmin=-157, lonmax=-142, latmin=-9, latmax=-5)
test_swath = resampling.generate_global_swath(test_geo_lims, 0.02)
resampled_ds = resampling.resample_clavrx(cwc_added_ds, global_grid=test_swath)
resampled_ds.to_netcdf("cropped_resampled.nc")

cloud_phase_3d_ds = construct_3d_field.compute_3d_clavrx_cloud_phase(
    sanitized_ds,
    number_of_levels=41,
    max_height=20,
)
cloud_phase_3d_ds.to_netcdf("cropped_3d.nc")

cwc_3d_ds = construct_3d_field.interpolate_cwc_to_altitude_levels(
    cwc_added_ds,
    rounding_precision=0.5,
    number_of_levels=41,
    max_height=20,
)
cwc_3d_ds.to_netcdf("cropped_3d_cwc.nc")

F, ax = plt.subplots(1, 4, figsize=(9, 3))
ax[0].imshow(cropped_ds.cld_height_acha, interpolation="none")
ax[0].set_title("Raw input")
ax[1].imshow(sanitized_ds.cld_height_acha, interpolation="none")
ax[1].set_title("Sanitized")
ax[2].imshow(geolim_clipped_ds.cld_height_acha, interpolation="none")
ax[2].set_title("Clipped")
ax[3].imshow(resampled_ds.cld_height_acha, interpolation="none", origin="lower")
ax[3].set_title("Resampled")
F.savefig("example_fig.png", bbox_inches="tight")
