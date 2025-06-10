import matplotlib.pyplot as plt
import xarray as xr
from clavrx_input import hdf4_input
from overcast_preprocessing import (
    blending,
    parallax_correction,
    resampling,
    sanitization,
)

platform_list = ["GOES-16", "Meteosat-10", "Meteosat-9", "HIM9", "GOES-18"]

manual_boundaries = {
    "GOES-16_Meteosat-10": -37.6,
    "Meteosat-10_Meteosat-9": 22.75,
    "Meteosat-9_HIM9": 93.1,
    "HIM9_GOES-18": -178.2,
    "GOES-18_GOES-16": -106.2,
}


def preprocess_file(ds: xr.Dataset, cwc_fn: str) -> xr.Dataset:
    out_ds = sanitization.sanitize_clavrx(ds)
    out_ds = parallax_correction.parallax_correct_clavrx(out_ds)
    out_ds = sanitization.ingest_cwc(out_ds, cwc_fn)
    out_ds = resampling.resample_clavrx(out_ds)
    return out_ds


REPROCESS_INPUTS = False

if REPROCESS_INPUTS:
    print("Loading GOES-16")
    G16_full = hdf4_input.read(
        "/mnt/overcastnas1/GEO_clavrx/GOES16_ABI/RadF/output/"
        "2023345/clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20233450600210.level2.hdf",  # type: ignore
    )
    print("Loading GOES-18")
    G18_full = hdf4_input.read(
        "/mnt/overcastnas1/GEO_clavrx/GOES18_ABI/RadF/output/"
        "2023345/clavrx_OR_ABI-L1b-RadF-M6C01_G18_s20233450600224.level2.hdf",  # type: ignore
    )
    G16_cwc_fn = "/mnt/overcastnas1/cwhite/cwc_golden_cases/2023345_0600/clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20233450600210.CWC.h5"
    G18_cwc_fn = "/mnt/overcastnas1/cwhite/cwc_golden_cases/2023345_0600/clavrx_OR_ABI-L1b-RadF-M6C01_G18_s20233450600224.CWC.h5"
    print("\nPreprocessing GOES-16")
    G16_full = preprocess_file(G16_full, G16_cwc_fn)
    print("Preprocessing GOES-18")
    G18_full = preprocess_file(G18_full, G18_cwc_fn)

    print("\nCropping files")
    G16_cropped = G16_full.isel(y=range(3600, 3800), x=range(3600, 3800))
    G18_cropped = G18_full.isel(y=range(3600, 3800), x=range(3600, 3800))

    print("\nSaving input files")
    G16_cropped.to_netcdf("blending_G16_input.nc")
    G18_cropped.to_netcdf("blending_G18_input.nc")
else:
    print("Loading G16 and G18 cropped and preprocessed from file")
    G16_cropped = xr.open_dataset("blending_G16_input.nc")
    G18_cropped = xr.open_dataset("blending_G18_input.nc")

print("\nBlending using hard cutoffs")
hard_cutoff_blended_ds = blending.blend_2d_clavrx(
    [G16_cropped, G18_cropped],
    weight_function=blending.hard_cutoff_weights,
)
print("Saving...")
hard_cutoff_blended_ds.to_netcdf("blending_hard_cutoffs.nc")

print("\nBlending using unweighted averaging")
unweighted_blended_ds = blending.blend_2d_clavrx(
    [G16_cropped, G18_cropped],
    weight_function=blending.unweighted_averaging_weights,
    cloud_phase_weight_function=blending.hard_cutoff_weights,
)
print("Saving...")
unweighted_blended_ds.to_netcdf("blending_unweighted.nc")

print("\nBlending using manual cutoffs")
manual_blended_ds = blending.blend_2d_clavrx(
    [G16_cropped, G18_cropped],
    weight_function=blending.manual_sensor_cutoff_weights,
    weight_function_kwargs={
        "platforms": platform_list,
        "boundaries": manual_boundaries,
    },
)
print("Saving...")
manual_blended_ds.to_netcdf("blending_manual.nc")

print("\nBlending using cosine of SZA")
cosine_blended_ds = blending.blend_2d_clavrx(
    [G16_cropped, G18_cropped],
    weight_function=blending.cosine_of_sza_weights,
)
print("Saving...")
cosine_blended_ds.to_netcdf("blending_cosine_of_sza.nc")

print("\nPlotting")
F, ax = plt.subplots(1, 6, figsize=(14, 4))
ax[0].imshow(G18_cropped.cld_height_acha, origin="lower", interpolation="none")
ax[0].set_title("G18")
ax[1].imshow(G16_cropped.cld_height_acha, origin="lower", interpolation="none")
ax[1].set_title("G16")
ax[2].imshow(
    hard_cutoff_blended_ds.cld_height_acha,
    origin="lower",
    interpolation="none",
)
ax[2].set_title("Hard Cutoff")
ax[3].imshow(
    unweighted_blended_ds.cld_height_acha,
    origin="lower",
    interpolation="none",
)
ax[3].set_title("Unweighted")
ax[4].imshow(manual_blended_ds.cld_height_acha, origin="lower", interpolation="none")
ax[4].set_title("Manual")
ax[5].imshow(cosine_blended_ds.cld_height_acha, origin="lower", interpolation="none")
ax[5].set_title("Cos of SZA")
F.savefig("blending_example_fig.png", bbox_inches="tight")
