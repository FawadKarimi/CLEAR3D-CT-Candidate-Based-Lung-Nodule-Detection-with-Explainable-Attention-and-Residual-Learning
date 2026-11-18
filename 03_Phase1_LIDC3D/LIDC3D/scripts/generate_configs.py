"""Generate named ablations from a validation-frozen reference configuration."""
import argparse
import copy
from pathlib import Path
import yaml


def generate(resolved_config, out):
    cfg = yaml.safe_load(Path(resolved_config).read_text())
    if cfg["candidates"]["threshold"] is None:
        raise ValueError("Freeze the LoG threshold on validation first")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    specs = {
        "H16": {},
        "C16": {"model": {"kind": "cnn"}},
        "C32": {"model": {"kind": "cnn", "width": 32}},
        "S1": {"model": {"scales": [2]}},
        "P0": {"model": {"positional_encoding": False}},
        "FC": {"model": {"fusion": "concat"}},
        "FG": {"model": {"fusion": "gated"}},
        "PS32": {"model": {"patch_size": 32}},
        "PS64": {"model": {"patch_size": 64}},
        "UW": {"training": {"class_weights": [1, 1]}},
        "W200": {"preprocess": {"window_hu": [-1000, 200]}},
        "MD0": {"preprocess": {"mask_dilation_mm": 0}},
        "CAP200": {"candidates": {"cap": 200}},
        "CAP800": {"candidates": {"cap": 800}},
    }
    for name, changes in specs.items():
        value = copy.deepcopy(cfg)
        for group, fields in changes.items():
            value[group].update(fields)
        with (output / f"{name}.yaml").open("x") as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--out", required=True)
    generate(**vars(parser.parse_args()))
