"""
File to run the accelerated alignment algorithm.
"""
import argparse
import json
import os
import re
from pathlib import Path

import ml_collections
import pandas as pd
import starfile
import yaml
import torch
import torch.multiprocessing as mp


def _resolve_config_path(config_arg: str) -> str:
    if os.path.isfile(config_arg):
        return config_arg
    candidate = os.path.join("configs", config_arg)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Config not found: '{config_arg}'. Tried '{config_arg}' and '{candidate}'."
    )


def _resolve_resource_path(raw_path: str, config_path: str) -> str:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return str(path)

    config_dir = Path(config_path).resolve().parent
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / path,
        config_dir / path,
        project_root / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(raw_path)


def _resolve_lookup_table_paths(config, config_path: str) -> None:
    for key in ("cs_path", "jl_zeros_path"):
        if key not in config:
            continue
        config[key] = _resolve_resource_path(config[key], config_path=config_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the accelerated alignment algorithm.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--align", action="store_true", help="Run the alignment workflow.")
    parser.add_argument("--example", action="store_true", help="Run the orientation search benchmark example.")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="",
        help="Optional path to write benchmark metrics as JSON (only for --example).",
    )
    # RELION External job contract
    parser.add_argument("--o", type=str, default="", help="RELION output directory.")
    parser.add_argument("--in_parts", type=str, default="", help="RELION input particles STAR.")
    parser.add_argument("--in_3dref", type=str, default="", help="RELION input 3D reference map.")
    parser.add_argument("--in_mask", type=str, default="", help="RELION optional input 3D mask.")
    parser.add_argument("--j", type=int, default=0, help="RELION threads/cpu-workers setting.")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids override.")
    parser.add_argument("--gpus", type=int, default=0, help="GPU count override.")
    return parser


def _resolve_example_default_config(config_arg: str) -> str:
    if config_arg != "config.yaml":
        return config_arg
    for candidate in ("config_example.yaml", os.path.join("configs", "config_example.yaml")):
        if os.path.isfile(candidate):
            return candidate
    return config_arg


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{value}'.")


def _parse_int_list(raw) -> list:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    text = text.strip("[]")
    values = [t.strip() for t in text.split(",") if t.strip()]
    return [int(v) for v in values]


def _parse_unknown_flags(unknown_args):
    extras = {}
    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if not token.startswith("--"):
            idx += 1
            continue

        key = token[2:].replace("-", "_")
        value = True
        if idx + 1 < len(unknown_args) and not unknown_args[idx + 1].startswith("--"):
            value = unknown_args[idx + 1]
            idx += 1
        extras[key] = value
        idx += 1
    return extras


def _coerce_value_like(raw_value, current_value):
    if isinstance(current_value, bool):
        return _parse_bool(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, list):
        if isinstance(raw_value, list):
            values = raw_value
        else:
            text = str(raw_value).strip().strip("[]")
            values = [v.strip() for v in text.split(",") if v.strip()]

        if not current_value:
            return values
        first = current_value[0]
        if isinstance(first, bool):
            return [_parse_bool(v) for v in values]
        if isinstance(first, int) and not isinstance(first, bool):
            return [int(v) for v in values]
        if isinstance(first, float):
            return [float(v) for v in values]
        return values
    return raw_value


def _resolve_relion_templates(path_3dref: str):
    input_path = Path(path_3dref)
    if not input_path.is_file():
        raise FileNotFoundError(f"--in_3dref file does not exist: {path_3dref}")

    path_str = str(input_path)
    has_half1 = re.search(r"half1", path_str, flags=re.IGNORECASE) is not None
    has_half2 = re.search(r"half2", path_str, flags=re.IGNORECASE) is not None

    if has_half1 and has_half2:
        raise ValueError("--in_3dref path contains both 'half1' and 'half2'; cannot infer pair.")

    if has_half1:
        half1 = path_str
        half2 = re.sub(r"half1", "half2", path_str, count=1, flags=re.IGNORECASE)
        if not Path(half2).is_file():
            raise FileNotFoundError(
                f"--in_3dref contains 'half1' but matching half2 map is missing: {half2}"
            )
        return [half1, half2]

    if has_half2:
        half2 = path_str
        half1 = re.sub(r"half2", "half1", path_str, count=1, flags=re.IGNORECASE)
        if not Path(half1).is_file():
            raise FileNotFoundError(
                f"--in_3dref contains 'half2' but matching half1 map is missing: {half1}"
            )
        return [half1, half2]

    # Single map mode: use same map for both halves
    return [path_str, path_str]


def _resolve_gpu_ids(args, extras, config):
    if args.gpu_ids:
        return _parse_int_list(args.gpu_ids)

    extra_gpu_ids = extras.get("gpu_ids", extras.get("gpu_id", ""))
    if extra_gpu_ids:
        return _parse_int_list(extra_gpu_ids)

    explicit_gpu_count = args.gpus
    if explicit_gpu_count <= 0:
        for key in ("gpus", "gpu", "num_gpus", "ngpu"):
            if key in extras:
                explicit_gpu_count = int(extras[key])
                break
    if explicit_gpu_count > 0:
        return list(range(explicit_gpu_count))

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if ids:
            return list(range(len(ids)))

    for env_key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"):
        env_value = os.environ.get(env_key, "").strip()
        if not env_value:
            continue
        match = re.search(r"(\d+)", env_value)
        if match:
            gpu_count = int(match.group(1))
            if gpu_count > 0:
                return list(range(gpu_count))

    if "gpu_ids" in config:
        return [int(x) for x in list(config.gpu_ids)]
    return []


def _write_relion_output_nodes(output_dir: Path, output_particle_star: str):
    output_nodes_path = output_dir / "RELION_OUTPUT_NODES.star"
    df_nodes = pd.DataFrame(
        {
            "rlnPipeLineNodeName": [output_particle_star],
            "rlnPipeLineNodeType": [3],
        }
    )
    starfile.write({"output_nodes": df_nodes}, output_nodes_path, overwrite=True)


def _clear_relion_exit_markers(output_dir: Path):
    for name in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        marker = output_dir / name
        if marker.exists():
            marker.unlink()


def _touch_marker(output_dir: Path, marker_name: str):
    (output_dir / marker_name).touch()


def _run_relion_mode(args, unknown_args, config):
    if not args.o:
        raise ValueError("RELION mode requires --o")
    if not args.in_parts:
        raise ValueError("RELION mode requires --in_parts")
    if not args.in_3dref:
        raise ValueError("RELION mode requires --in_3dref")

    extras = _parse_unknown_flags(unknown_args)
    output_dir = Path(args.o)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_relion_exit_markers(output_dir)

    abort_now = output_dir / "RELION_JOB_ABORT_NOW"
    if abort_now.exists():
        abort_now.unlink()
        _touch_marker(output_dir, "RELION_JOB_EXIT_ABORTED")
        return

    try:
        from utils.setup_utils import assert_inputs
        from utils.run_utils import run_align

        config.relion_external_mode = True
        config.particle_paths_from_star = True
        config.random_half_split = True
        config.relion_version = "relion5"
        config.relion_output_stem = "matcha_particles"
        config.path_output = str(output_dir)
        config.run_data_path = args.in_parts
        config.path_templates = _resolve_relion_templates(args.in_3dref)
        config.subset_IDs = [1, 2]

        if args.in_mask:
            config.path_template_mask = args.in_mask
        else:
            config.path_template_mask = ""
            print("[WARN] No mask provided via --in_mask; continuing without template mask.")

        if args.j > 0:
            config.cpu_reader_workers = int(args.j)
        elif "j" in extras:
            config.cpu_reader_workers = int(extras["j"])

        config.gpu_ids = _resolve_gpu_ids(args=args, extras=extras, config=config)
        if not config.gpu_ids:
            raise ValueError("Could not resolve any GPU ids from flags/env/config.")

        reserved = {
            "o",
            "in_parts",
            "in_3dref",
            "in_mask",
            "j",
            "align",
            "example",
            "config",
            "metrics_out",
            "gpu_ids",
            "gpus",
            "gpu",
            "num_gpus",
            "ngpu",
        }
        for key, raw_value in extras.items():
            if key in reserved or key not in config:
                continue
            try:
                config[key] = _coerce_value_like(raw_value, config[key])
            except Exception as exc:
                print(f"[WARN] Ignoring RELION extra override --{key}: {exc}")

        assert_inputs(config)
        success = run_align(config=config)
        if not success:
            raise RuntimeError("Alignment run finished but STAR join failed.")

        output_particle_star = f"{config.path_output}.star"
        _write_relion_output_nodes(output_dir=output_dir, output_particle_star=output_particle_star)

        if abort_now.exists():
            abort_now.unlink()
            _touch_marker(output_dir, "RELION_JOB_EXIT_ABORTED")
            return

        _touch_marker(output_dir, "RELION_JOB_EXIT_SUCCESS")
    except Exception:
        _touch_marker(output_dir, "RELION_JOB_EXIT_FAILURE")
        raise


def main(argv=None):
    # We don't use autodiff in this project
    with torch.no_grad():
        parser = _build_parser()
        args, unknown_args = parser.parse_known_args(argv)

        if args.metrics_out and not args.example:
            parser.error("--metrics_out is only valid with --example")

        relion_mode = bool(args.o or args.in_parts or args.in_3dref or args.in_mask)
        if not relion_mode and not args.align and not args.example:
            parser.error("Specify one mode: --align, --example, or RELION flags (--o/--in_parts/--in_3dref).")
        if relion_mode and args.example:
            parser.error("--example cannot be combined with RELION flags.")

        # Get config file
        config_arg = _resolve_example_default_config(args.config) if args.example else args.config
        config_path = _resolve_config_path(config_arg)
        with open(config_path) as cf_file:
            config_yaml = yaml.safe_load(cf_file.read())
        config = ml_collections.ConfigDict(config_yaml, type_safe=True)
        _resolve_lookup_table_paths(config=config, config_path=config_path)

        if relion_mode:
            _run_relion_mode(args=args, unknown_args=unknown_args, config=config)
            return

        if args.example:
            from example import main as run_example

            metrics = run_example(config)
            if args.metrics_out:
                with open(args.metrics_out, "w", encoding="utf-8") as mf:
                    json.dump(metrics or {}, mf, indent=2)
            return

        if args.align:
            from utils.setup_utils import assert_inputs
            from utils.run_utils import run_align

            assert_inputs(config)

            # Run alignment
            run_align(config=config)
            return


def cli() -> None:
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    with torch.inference_mode():
        main()


if __name__ == "__main__":
    cli()
