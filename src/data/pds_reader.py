"""Read detached-label PDS3 elevation rasters without confusing slope with height.

The repository's ``*_SL_*`` products are slope maps, not DEMs, and are rejected.
Only explicit elevation/height/DEM or radius products with declared length units,
map scale and local Cartesian geometry are accepted. Global planetary products
must be reprojected externally: MAP_SCALE alone is not true surface spacing near
the poles. Returned elevations and spacing are always metres.
"""

import argparse
import math
from pathlib import Path
import re

import numpy as np


def _field(text, key, default=None):
    match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*([^\r\n]+)", text, re.M)
    return default if match is None else match.group(1).split("/*", 1)[0].strip().strip('"')


def _number(value, name):
    try:
        number = float(re.sub(r"\s*<[^>]+>\s*$", "", str(value)).replace("D", "E"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"PDS label needs a numeric {name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"PDS {name} must be finite")
    return number


def _length_multiplier(unit):
    normalized = str(unit).strip(' <>"').upper().replace(" ", "")
    normalized = normalized.split("/", 1)[0]
    if normalized in ("M", "METER", "METERS", "METRE", "METRES"):
        return 1.0
    if normalized in ("KM", "KILOMETER", "KILOMETERS", "KILOMETRE", "KILOMETRES"):
        return 1000.0
    raise ValueError(f"Unsupported or missing PDS length unit: {unit!r}")


def _value_and_unit(value, name):
    match = re.search(r"<([^>]+)>", str(value))
    if match is None:
        raise ValueError(f"PDS {name} must declare its length unit")
    return _number(value, name) * _length_multiplier(match.group(1))


def _metadata(img_path):
    image = Path(img_path).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"PDS image not found: {image}")
    label = next((p for p in (image.with_suffix(".LBL"), image.with_suffix(".lbl"))
                  if p.is_file()), None)
    if label is None:
        raise ValueError(f"A matching .LBL label is required for {image.name}")
    text = label.read_text(encoding="utf-8-sig")
    block = re.search(r"^\s*OBJECT\s*=\s*IMAGE\s*$([\s\S]*?)^\s*END_OBJECT\s*=\s*IMAGE\s*$",
                      text, re.M)
    if block is None:
        raise ValueError("PDS label is missing its IMAGE object")
    image_fields = block.group(1)
    name = _field(image_fields, "NAME", "").upper()
    if "SLOPE" in name or "_SL_" in image.stem.upper():
        raise ValueError("SLOPE products contain angles, not elevations; supply an elevation DEM")
    if not any(word in name for word in ("ELEVATION", "HEIGHT", "DEM", "RADIUS")):
        raise ValueError("PDS IMAGE NAME must explicitly identify an elevation or radius product")
    projection = _field(text, "MAP_PROJECTION_TYPE", "").upper().replace(" ", "_")
    if projection not in ("LOCAL_CARTESIAN", "LOCAL_ENU"):
        raise ValueError(
            "PDS terrain must first be reprojected to a local Cartesian metre grid; "
            "MAP_SCALE is not true ground spacing for geographic/polar imagery. "
            "The preprocessed label must explicitly declare "
            "MAP_PROJECTION_TYPE = LOCAL_CARTESIAN or LOCAL_ENU."
        )
    lines = _number(_field(image_fields, "LINES"), "LINES")
    samples = _number(_field(image_fields, "LINE_SAMPLES"), "LINE_SAMPLES")
    if lines < 1 or samples < 1 or not lines.is_integer() or not samples.is_integer():
        raise ValueError("PDS raster dimensions must be positive integers")
    lines, samples = int(lines), int(samples)
    bits = _number(_field(image_fields, "SAMPLE_BITS"), "SAMPLE_BITS")
    sample_type = _field(image_fields, "SAMPLE_TYPE", "").upper()
    encodings = {"PC_REAL": "<f", "IEEE_REAL": ">f", "LSB_INTEGER": "<i",
                 "MSB_INTEGER": ">i", "LSB_UNSIGNED_INTEGER": "<u",
                 "MSB_UNSIGNED_INTEGER": ">u"}
    if sample_type not in encodings or bits not in (16, 32, 64):
        raise ValueError(f"Unsupported PDS sample format {sample_type}/{bits}")
    if sample_type.endswith("REAL") and bits not in (32, 64):
        raise ValueError("PDS real samples must be 32 or 64 bits")
    dtype = np.dtype(encodings[sample_type] + str(int(bits) // 8))
    for key in ("LINE_PREFIX_BYTES", "LINE_SUFFIX_BYTES"):
        if _number(_field(image_fields, key, "0"), key) != 0:
            raise ValueError(f"PDS {key} is unsupported; expected a contiguous raster")
    pointer = _field(text, "^IMAGE")
    if pointer is not None and Path(pointer).name.lower() != image.name.lower():
        raise ValueError("Only detached PDS labels pointing directly to this IMG are supported")
    expected_size = lines * samples * dtype.itemsize
    if image.stat().st_size != expected_size:
        raise ValueError(f"PDS image size mismatch: expected {expected_size} bytes, got {image.stat().st_size}")
    unit = _field(image_fields, "UNIT", _field(image_fields, "UNITS"))
    scale_text = _field(image_fields, "SCALING_FACTOR", "1")
    if unit is None:
        scale_unit = re.search(r"<([^>]+)>", scale_text)
        unit = scale_unit.group(1) if scale_unit else None
    multiplier = _length_multiplier(unit)
    scaling = _number(scale_text, "SCALING_FACTOR")
    if scaling == 0:
        raise ValueError("PDS SCALING_FACTOR cannot be zero")
    offset_text = _field(image_fields, "OFFSET", "0")
    offset_match = re.search(r"<([^>]+)>", offset_text)
    offset_m = (_value_and_unit(offset_text, "OFFSET") if offset_match
                else _number(offset_text, "OFFSET") * multiplier)
    spacing_m = _value_and_unit(_field(text, "MAP_SCALE"), "MAP_SCALE")
    if spacing_m <= 0:
        raise ValueError("PDS MAP_SCALE must be positive")
    reference_radius_m = 0.0
    if "RADIUS" in name:
        reference_radius_m = _value_and_unit(_field(text, "A_AXIS_RADIUS"), "A_AXIS_RADIUS")
        if reference_radius_m <= 0:
            raise ValueError("Radius products need a positive reference A_AXIS_RADIUS")
    missing = []
    for key in ("MISSING_CONSTANT", "INVALID_CONSTANT", "CORE_NULL", "NULL",
                "LOW_REPR_SATURATION", "HIGH_REPR_SATURATION"):
        value = _field(image_fields, key)
        if value is not None:
            radix = re.fullmatch(r"16#([0-9A-Fa-f]+)#", value)
            if radix:
                little = dtype.byteorder == "<" or (dtype.byteorder == "=" and np.little_endian)
                byteorder = "little" if little else "big"
                data = int(radix.group(1), 16).to_bytes(dtype.itemsize, byteorder)
                missing.append(np.frombuffer(data, dtype=dtype)[0])
            else:
                # Compare in source precision: a label's decimal float sentinel
                # need not have the same float64 value as its stored float32.
                missing.append(np.asarray(_number(value, key), dtype=dtype).item())
    return image, dtype, missing, {
        "source": str(image), "label": str(label), "product_name": name,
        "lines": lines, "samples": samples, "sample_spacing_m": spacing_m,
        "source_units": unit, "units": "m", "scale_to_m": scaling * multiplier,
        "offset_m": offset_m - reference_radius_m,
        "reference_radius_m": reference_radius_m,
        "projection": projection,
    }


def _extract(img_path, size, start=None, rng=None):
    image, dtype, missing, metadata = _metadata(img_path)
    if isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)):
        rows = cols = int(size)
    else:
        try:
            rows, cols = size
        except (TypeError, ValueError) as exc:
            raise ValueError("patch size must be an integer or (rows, cols)") from exc
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (rows, cols)):
        raise ValueError("patch dimensions must be integers")
    if not (1 <= rows <= metadata["lines"] and 1 <= cols <= metadata["samples"]):
        raise ValueError("requested patch does not fit the PDS raster")
    if start is None:
        rng = np.random.default_rng() if rng is None else rng
        sr = int(rng.integers(0, metadata["lines"] - rows + 1))
        sc = int(rng.integers(0, metadata["samples"] - cols + 1))
    else:
        sr, sc = start
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in start):
            raise ValueError("patch start must contain integers")
        if not (0 <= sr <= metadata["lines"] - rows and 0 <= sc <= metadata["samples"] - cols):
            raise ValueError("requested patch extends beyond the PDS raster")
    mapped = np.memmap(image, dtype=dtype, mode="r", shape=(metadata["lines"], metadata["samples"]))
    try:
        patch = np.array(mapped[sr:sr + rows, sc:sc + cols], dtype=np.float64)
    finally:
        mapped._mmap.close()
    if not np.isfinite(patch).all() or any(np.any(patch == float(v)) for v in missing):
        raise ValueError("requested PDS patch contains missing or nonfinite samples")
    patch = patch * metadata["scale_to_m"] + metadata["offset_m"]
    if not np.isfinite(patch).all():
        raise ValueError("scaled PDS elevations are nonfinite")
    metadata.update({"row": int(sr), "col": int(sc), "patch_rows": int(rows), "patch_cols": int(cols)})
    return patch, metadata


def read_elevation_patch(img_path, size, rng=None):
    """Return a random (elevations_in_metres, metadata) patch from a validated DEM.

    The label must explicitly declare LOCAL_CARTESIAN or LOCAL_ENU after external
    reprojection; geographic and unspecified projections are rejected. Metadata
    contains ``sample_spacing_m`` from MAP_SCALE. Preserve that spacing or
    explicitly resample both axes; changing spacing alone distorts the terrain.
    """
    return _extract(img_path, size, rng=rng)


def extract_lunar_patch(img_path, output_csv, start_row=1000, start_col=5000, patch_size=500):
    """Compatibility export; errors propagate instead of reporting false success."""
    patch, metadata = _extract(img_path, patch_size, start=(start_row, start_col))
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, patch, fmt="%.6f", delimiter=" ")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a validated PDS3 elevation patch in metres")
    parser.add_argument("img", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--size", type=int, default=500)
    args = parser.parse_args()
    try:
        result = extract_lunar_patch(args.img, args.output, args.row, args.col, args.size)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}; source spacing {result['sample_spacing_m']} m/sample")
