#!/usr/bin/env python3
"""
Custom OVERCAST Preprocessing Procflow (demo).

This procflow reads, sanitizes, parallax-corrects, and resamples CLAVR-x
data using GeoIPS. The processed data are then resampled to a common grid
and blended into both 2D and 3D outputs.

Usage:
    python3 preprocessing.py --workflow <path to workflow yaml>
"""

import argparse
import logging
from collections.abc import Mapping
from typing import Any

import xarray as xr
import yaml
from dask.base import tokenize
from rich.logging import RichHandler

import geoips

FORMAT = "%(name)s - %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
log = logging.getLogger("main")


def parse_args_with_argparse() -> argparse.Namespace:
    """Parse command-line arguments using argparse.

    Returns
    -------
    argparse.Namespace
        Parsed arguments (e.g. with .workflow attribute).
    """
    parser = argparse.ArgumentParser(description="Demo Overcast Preprocessing Procflow")
    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        help="Path to the workflow YAML file",
    )
    return parser.parse_args()


def read_workflow_file(workflow_file: str) -> dict[str, Any]:
    """
    Read the specified workflow file (e.g. YAML), and return its content.

    Parameters
    ----------
    workflow_file : str
        Path to the workflow file.

    Returns
    -------
    Dict[str, Any]
        Parsed workflow dictionary.
    """
    log.debug(f"Reading workflow file: {workflow_file}")
    with open(workflow_file, encoding="utf-8") as f:
        workflow = yaml.safe_load(f)
    return workflow


def flatten_maplike(m, parent_key=""):
    """Flatten nested map-like into a single-level dict.

    Keys are joined by dots.

    Example
    -------
    >>> flatten_maplike({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    >>> {"a": 1, "b.c": 2, "b.d.e": 3}
    """
    flattened = {}
    for key, value in m.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, Mapping):
            flattened.update(flatten_maplike(value, new_key))
        else:
            flattened[new_key] = value
    return flattened


def applyFunctionToNestedValues(f, maplike):
    new_map = {}
    for key, value in maplike.items():
        if isinstance(value, Mapping):
            new_map[key] = applyFunctionToNestedValues(f, maplike=value)
        if isinstance(value, list):
            new_map[key] = [f(x) for x in value]
        else:
            new_map[key] = f(value)
    return new_map


def expand_workflow(step, inputs):
    kind = next(iter(step.keys()))
    settings = step[kind]
    arguments = settings.get("arguments", None)
    auto_sub = lambda x: (
        x.replace("${", "%(").replace("}", ")s") % inputs  # for arbitrary . notation
        if isinstance(x, str)
        else x
    )
    if arguments:
        arguments = applyFunctionToNestedValues(
            auto_sub,
            arguments,
        )
    tag = settings.get("tag", None)
    input_xobj_tag = settings.get("input-xobj", None)
    pass_datatree = settings.get("pass datatree", False)
    return kind, settings["name"], arguments, tag, pass_datatree, input_xobj_tag


def expand_inputs(inputs, workflow_args):
    inputs = flatten_maplike(inputs)
    if workflow_args:
        for key in inputs:
            if key in workflow_args:
                inputs[key] = workflow_args[key]
    return inputs


def concatenate_merge(datatree, order):
    new = [datatree[item].to_dataset() for item in order]
    print([type(x) for x in new])
    return new


def run_step(kind, name, xobj, arguments):
    if kind == "workflow":
        sub_workflow = read_workflow_file(workflow_file=f"{name}.yaml")
        return process_workflow(sub_workflow, workflow_args=arguments)
    elif kind == "reader":
        arguments["vars"] = arguments.get("vars", None)
        arguments["metadata_only"] = arguments.get("metadata_only", False)
        try:
            files = [arguments["file"]]
        except KeyError:
            files = arguments["files"]
        return geoips.interfaces.readers.get_plugin(name)(
            files,
            chans=arguments["vars"],
            metadata_only=arguments["metadata_only"],
        )["DATA"]
    elif kind == "algorithm":
        if name in ["uv_octopy", "octopy-clavrx", "neo-octopy-clavrx"]:
            return geoips.interfaces.algorithms.get_plugin(name)(
                {"DATA": xobj, "METADATA": xobj},
                **arguments,
            )
        else:
            return geoips.interfaces.algorithms.get_plugin(name)(xobj, **arguments)
    elif kind == "interpolator":
        return geoips.interfaces.interpolators.get_plugin(name)(xobj, **arguments)
    elif kind == "merge":
        return concatenate_merge(xobj, arguments["order"])
    elif kind == "output formatter":
        filename = arguments["filename"]
        del arguments["filename"]
        return geoips.interfaces.output_formatters.get_plugin(name)(
            xobj,
            "",
            [filename],
            **arguments,
        )
    else:
        raise NotImplementedError(f"Kind: {kind}")


def process_workflow(workflow: dict[str, Any], workflow_args: dict[str, Any]) -> None:
    """
    Execute the steps defined in the workflow dictionary.

    This is a placeholder that you can expand to call readers, algorithms,
    resamplers, merges, or other steps. The exact logic depends on your
    pipeline and how the workflow is structured.

    Parameters
    ----------
    workflow : Dict[str, Any]
        Parsed workflow definition from a YAML (or other format).
    """
    log.debug("process_workflow: Start")
    state = xr.DataTree(name="state")
    steps = workflow["spec"]["steps"]
    inputs = expand_inputs(workflow["inputs"], workflow_args)

    xobj = None
    state["last_returned_xobj"] = None
    state.attrs.update(hashes={"last": None})
    for step in steps:
        kind, name, arguments, tag, pass_datatree, input_xobj_tag = expand_workflow(
            step,
            inputs,
        )
        if not arguments:
            arguments = {}

        log.info(
            "Running [bold green]%s[/bold green] [bold cyan]%s[/bold cyan] "
            "with arguments [red]%s[/red]",
            kind,
            name,
            arguments,
        )
        if pass_datatree and input_xobj_tag:
            log.error(
                "Pass data tree and input cannot both be set. Defaulting to input xobj.",
            )
        if input_xobj_tag:
            try:
                xobj = getattr(state, input_xobj_tag)
            except AttributeError:
                xobj = getattr(state.attr, input_xobj_tag)
        elif not pass_datatree:
            xobj = state.last_returned_xobj
        else:
            xobj = state
        log.debug(f"Data passed is of type {type(xobj)}")
        result = run_step(kind, name, xobj, arguments)
        if tag:
            log.debug("Tagging workflow with tag [bold cyan]'%s'[/bold cyan]", tag)
            try:
                state[tag] = result
            except TypeError:
                log.debug("Tagging as attr becuase is not xobj")
                state.attrs[tag] = result
        state.last_returned_xobj = result
        if not result:
            log.warning("Step provided None output")
        step_hash = tokenize(result)
        if "output-hash" in step[kind]:
            log.info(f"Hash of step output: {step_hash}")
            if "tag" in step[kind]["output-hash"]:
                hash_tag = step[kind]["output-hash"]["tag"]
                state.attrs["hashes"].update(**{hash_tag: step_hash})
                logging.debug(f"Storing hash under key {hash_tag}")
            if "match" in step[kind]["output-hash"]:
                hash_match_key = step[kind]["output-hash"]["match"]
                if hash_match_key != "any":
                    try:
                        hash_match = state.attrs["hashes"][hash_match_key]
                    except KeyError:
                        logging.critical(f"Could not find hash {hash_match_key}.")
                        print(state.attrs["hashes"])
                        raise
                    if hash_match != step_hash:
                        log.error(
                            f"Output hash does not match hash with key '{hash_match_key}'.",
                        )
                    else:
                        log.info(
                            f"Output hash matches hash with key '{hash_match_key}'.",
                        )

    log.debug("process_workflow: End")
    return state.last_returned_xobj


if __name__ == "__main__":
    log.debug("NOT a REAL Procflow Started.")
    log.debug("Parsing CLI Args")
    args = parse_args_with_argparse()

    log.debug("Reading workflow file")
    workflow = read_workflow_file(args.workflow)

    log.debug("Processing workflow")
    process_workflow(workflow, workflow_args=None)

    log.info("Done! ☁️☁☁️️")
