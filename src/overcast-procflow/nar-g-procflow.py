#!/usr/bin/env python3
"""
Recursive YAML-based OVERCAST Preprocessing workflow using GeoIPS.

Usage:
    python3 yaml_preprocessing_workflow.py
"""

import logging
from typing import Any

import xarray as xr
import yaml
from rich.logging import RichHandler

import geoips


def configure_logger(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger."""
    logging.basicConfig(
        level=level,
        format="%(name)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    return logging.getLogger("yaml_preprocessing")


def read_yaml_file(filepath: str) -> dict[str, Any]:
    """Read and parse a YAML file into a dictionary."""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Step-specific helpers for Reader, Algorithm, and Resampler
# ----------------------------------------------------------------------
def run_reader_step(step_cfg: dict[str, Any], logger: logging.Logger) -> xr.Dataset:
    """
    Execute a GeoIPS reader step.

    Example step_cfg:
      {
        "name": "clavrx_hdf4",
        "arguments": {
          "vars": [...],
          "file": "/path/to/file"
        }
      }
    """
    reader_name = step_cfg["name"]
    arguments = step_cfg.get("arguments", {})
    vars_to_read = arguments.get("vars", [])
    file_path = arguments.get("file")

    if not file_path:
        raise ValueError("Reader step requires a 'file' path.")

    logger.info(f"[Reader] Using '{reader_name}' on file: {file_path}")
    reader_plugin = geoips.interfaces.readers.get_plugin(reader_name)
    data_dict = reader_plugin([file_path], chans=vars_to_read)
    return data_dict["DATA"]


def run_algorithm_step(
    dataset: xr.Dataset,
    step_cfg: dict[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """
    Execute a GeoIPS algorithm step on the given dataset.

    Example step_cfg:
      {
        "name": "sanitize_clavrx",
        "description": "Sanitize data"
      }
    """
    alg_name = step_cfg["name"]
    logger.info(f"[Algorithm] Running '{alg_name}'")
    alg_plugin = geoips.interfaces.algorithms.get_plugin(alg_name)
    return alg_plugin(dataset)


def run_resampler_step(
    dataset: xr.Dataset,
    step_cfg: dict[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """
    Execute a GeoIPS resampler (interpolator) step on the given dataset.

    Example step_cfg:
      {
        "name": "overcast_preprocessing_resample",
        "description": "Resample to OVERCAST grid"
      }
    """
    resampler_name = step_cfg["name"]
    logger.info(f"[Resampler] Using '{resampler_name}'")
    resampler_plugin = geoips.interfaces.interpolators.get_plugin(resampler_name)
    return resampler_plugin(dataset)


# ----------------------------------------------------------------------
# The Single Recursive Function
# ----------------------------------------------------------------------
def run_yaml_workflow(
    workflow_cfg: dict[str, Any],
    processed_datasets: dict[str, xr.Dataset],
    logger: logging.Logger,
    files_dict: dict[str, str],
    default_tag: str | None = None,
) -> dict[str, xr.Dataset]:
    """
    Recursively process the given workflow YAML config.

    workflow_cfg is expected to have the form:
      {
        "steps": [
          {"reader": {...}},
          {"algorithm": {...}},
          {"resampler": {...}},
          {"merge": {...}},
          {"workflow": {...}}
        ]
      }

    If "workflow" references another YAML file, that file is read, parsed,
    and this function is called recursively on it.

    Parameters
    ----------
    workflow_cfg : Dict[str, Any]
        YAML dictionary describing the steps.
    processed_datasets : Dict[str, xr.Dataset]
        Dictionary to store and retrieve intermediate and final datasets.
    logger : logging.Logger
        Logger for progress messages.
    files_dict : Dict[str, str]
        Mapping from placeholders to actual file paths (e.g. "GOES-16" -> "/path/to/file.hdf").
    default_tag : Optional[str], optional
        If a step doesn't specify a distinct tag, use this default (if any).

    Returns
    -------
    Dict[str, xr.Dataset]
        Updated processed_datasets after performing all steps in workflow_cfg.
    """
    steps = workflow_cfg.get("steps", [])

    for step in steps:
        # Identify which type of step it is
        # 1) Reader, 2) Algorithm, 3) Resampler, 4) Merge, 5) Workflow
        if "reader" in step:
            # -------------
            # READER STEP
            # -------------
            reader_cfg = step["reader"]
            # Possibly override the "file" argument using "filename" from the step
            args = reader_cfg.setdefault("arguments", {})

            # If there's a separate "filename" field referencing top-level input:
            step_filename = args.get("file")
            if not step_filename:
                # Or it might be given like: "filename": "${files.GOES-16}"
                # Adjust if necessary. (Your YAML structure may vary.)
                pass

            # If we see an expression like ${files.GOES-16}, do substitution
            if (
                step_filename
                and step_filename.startswith("${files.")
                and step_filename.endswith("}")
            ):
                key = step_filename.replace("${files.", "").replace("}", "")
                if key not in files_dict:
                    raise ValueError(
                        f"No file mapping found for key '{key}' in inputs.files.",
                    )
                args["file"] = files_dict[key]

            # Actually read the data
            new_dataset = run_reader_step(reader_cfg, logger)

            # Tag for storing in processed_datasets
            dataset_tag = step.get("tag", default_tag or "untagged_reader")
            processed_datasets[dataset_tag] = new_dataset

        elif "algorithm" in step:
            # -------------
            # ALGORITHM STEP
            # -------------
            alg_cfg = step["algorithm"]

            # Determine which dataset to apply it to
            input_tag = step.get("input_tag", default_tag or None)
            if not input_tag or input_tag not in processed_datasets:
                # fallback to last dataset if not found or specified
                if processed_datasets:
                    ds_in = list(processed_datasets.values())[-1]
                else:
                    raise RuntimeError("No dataset available for algorithm step.")
            else:
                ds_in = processed_datasets[input_tag]

            ds_out = run_algorithm_step(ds_in, alg_cfg, logger)

            # Tag for storing result
            output_tag = step.get("tag", default_tag or "untagged_algorithm")
            processed_datasets[output_tag] = ds_out

        elif "resampler" in step:
            # -------------
            # RESAMPLER STEP
            # -------------
            res_cfg = step["resampler"]

            # Determine which dataset to apply it to
            input_tag = step.get("input_tag", default_tag or None)
            if not input_tag or input_tag not in processed_datasets:
                if processed_datasets:
                    ds_in = list(processed_datasets.values())[-1]
                else:
                    raise RuntimeError("No dataset available for resampler step.")
            else:
                ds_in = processed_datasets[input_tag]

            ds_out = run_resampler_step(ds_in, res_cfg, logger)

            # Tag for storing result
            output_tag = step.get("tag", default_tag or "untagged_resampler")
            processed_datasets[output_tag] = ds_out

        elif "merge" in step:
            # -------------
            # MERGE STEP
            # -------------
            merge_cfg = step["merge"]
            merge_name = merge_cfg.get("name", "")
            order = merge_cfg.get("arguments", {}).get("order", [])

            logger.info(f"[Merge] Step '{merge_name}' in order: {order}")
            ds_list: list[xr.Dataset] = []
            for tag in order:
                if tag not in processed_datasets:
                    logger.warning(f"[Merge] Tag '{tag}' not found. Skipping.")
                else:
                    ds_list.append(processed_datasets[tag])

            if not ds_list:
                raise ValueError(
                    "No valid datasets to merge. Check configuration or tags.",
                )

            merged = xr.concat(ds_list, dim="concat_dim")
            output_tag = step.get("tag", "merged")
            processed_datasets[output_tag] = merged

        elif "workflow" in step:
            # -------------
            # WORKFLOW STEP (Recursive)
            # -------------
            wf_cfg = step["workflow"]
            # Possibly the sub-workflow is an *inline* set of steps,
            # or it references a file containing the sub-workflow definition.
            if "file" in wf_cfg:
                # If there's a subworkflow file, read and parse it, then recurse
                subworkflow_file = wf_cfg["file"]
                logger.info(f"[Workflow] Reading sub-workflow file: {subworkflow_file}")
                sub_cfg = read_yaml_file(subworkflow_file)
                sub_tag = step.get("tag", default_tag or "untagged_workflow")
                # Recurse
                run_yaml_workflow(
                    sub_cfg,
                    processed_datasets,
                    logger,
                    files_dict,
                    default_tag=sub_tag,
                )
            else:
                # If the sub-workflow is inline, it might have "steps" directly
                # Example:
                #   "workflow": {
                #       "steps": [ {"reader": ...}, {"algorithm": ...} ]
                #   }
                logger.info("[Workflow] Running inline sub-workflow steps")
                run_yaml_workflow(
                    wf_cfg,
                    processed_datasets,
                    logger,
                    files_dict,
                    default_tag=step.get("tag", default_tag),
                )

        else:
            logger.warning(f"[Workflow] Unrecognized step: {step}")

    return processed_datasets


def main():
    logger = configure_logger(level="INFO")

    # Example: We have a top-level "preprocessing.yaml" with 'inputs' and 'spec'
    # plus a separate "read_sanitize_parallax_correct_resample.yaml".
    main_workflow_file = "preprocessing.yaml"
    top_cfg = read_yaml_file(main_workflow_file)

    # Extract top-level "files" mapping for data references:
    inputs_dict = top_cfg.get("inputs", {})
    files_dict = inputs_dict.get("files", {})

    # The top-level workflow steps are often under "spec" -> "steps",
    # but let's just pass the entire top_cfg to run_yaml_workflow.
    # We'll store our results in a dict:
    processed_datasets: dict[str, xr.Dataset] = {}

    logger.info("Starting YAML-based OVERCAST Preprocessing Workflow (Recursive).")
    run_yaml_workflow(
        workflow_cfg=top_cfg.get("spec", {}),
        processed_datasets=processed_datasets,
        logger=logger,
        files_dict=files_dict,
    )

    # If there's a final dataset (tagged "final"), output it to NetCDF
    if "final" in processed_datasets:
        logger.info("[Output] Writing 'final' dataset to final_output.nc")
        final_ds = processed_datasets["final"]
        output_plugin = geoips.interfaces.output_formatters.get_plugin("netcdf_xarray")
        output_plugin(final_ds, "final_output", ["final_output.nc"], clobber=True)
    else:
        logger.info("No 'final' dataset found. Completed without NetCDF output.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
