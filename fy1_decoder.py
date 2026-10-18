#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FY-1C/D AVHRR binary decoder based on the uploaded NSMC format document.

Supports:
  * HRPT AVHRR 1B  : 28400-byte records; first two records are headers.
  * HRPT AVHRR 1A.5: 44360-byte records; first record is the header.
  * GDPT AVHRR 1A.5:  9744-byte records; first record is the header.

Dependencies:
  pip install numpy pillow

Examples:
  python fy1_decoder.py input.dat
  python fy1_decoder.py input.dat --format hrpt1b --endian big
  python fy1_decoder.py input.dat --format hrpt1a5 --layout channel
  python fy1_decoder.py input.dat --format gdpt1a5 --out-dir decoded

The source document does not unambiguously state the 1A.5 sample memory
ordering. --layout pixel assumes per-pixel channel interleave; --layout channel
assumes all samples for channel 1, then channel 2, etc. Compare previews if
necessary. HRPT 1B ordering is explicitly described as pixel-major in the PDF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("缺少 numpy，请先运行：python -m pip install numpy", file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError:
    Image = None

RECORD_SIZES = {
    "hrpt1b": 28400,
    "hrpt1a5": 44360,
    "gdpt1a5": 9744,
}
FORMAT_LABELS = {
    "hrpt1b": "FY-1C/D HRPT AVHRR 1B",
    "hrpt1a5": "FY-1C/D HRPT AVHRR 1A.5",
    "gdpt1a5": "FY-1C/D GDPT AVHRR 1A.5",
}


def dtype(endian: str, code: str) -> np.dtype:
    return np.dtype(endian + code)


def read_i16(buf: bytes, offset: int, endian: str, signed: bool = True) -> int:
    code = "h" if signed else "H"
    return int(struct.unpack_from(endian + code, buf, offset)[0])


def read_i32(buf: bytes, offset: int, endian: str, signed: bool = True) -> int:
    code = "i" if signed else "I"
    return int(struct.unpack_from(endian + code, buf, offset)[0])


def read_f32s(buf: bytes, offset: int, count: int, endian: str) -> np.ndarray:
    return np.frombuffer(buf, dtype=dtype(endian, "f4"), count=count, offset=offset).astype(np.float32, copy=True)


def read_i16s(buf: bytes, offset: int, count: int, endian: str, signed: bool = True) -> np.ndarray:
    code = "i2" if signed else "u2"
    return np.frombuffer(buf, dtype=dtype(endian, code), count=count, offset=offset).astype(np.int32 if signed else np.uint16, copy=True)


def detect_format(path: Path) -> str:
    size = path.stat().st_size
    candidates: List[str] = []
    for fmt, rec_size in RECORD_SIZES.items():
        n = size // rec_size
        min_records = 3 if fmt == "hrpt1b" else 2
        if size % rec_size == 0 and n >= min_records:
            candidates.append(fmt)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        divisibility = ", ".join(f"{name}={size % rec}" for name, rec in RECORD_SIZES.items())
        raise ValueError(
            f"无法根据文件长度自动识别格式。文件大小={size} 字节，记录长度余数：{divisibility}。\n"
            "请确认文件完整，或使用 --format hrpt1b / hrpt1a5 / gdpt1a5 指定格式。"
        )
    # Prefer a record format whose first/header record carries the familiar FY-1C/D IDs.
    matches: List[str] = []
    with path.open("rb") as f:
        for fmt in candidates:
            f.seek(RECORD_SIZES[fmt] if fmt == "hrpt1b" else 0)
            head = f.read(8)
            if len(head) >= 2:
                for order in (">", "<"):
                    if struct.unpack_from(order + "H", head, 0)[0] in (113, 114):
                        matches.append(fmt)
                        break
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"文件长度同时符合多种记录格式：{', '.join(candidates)}。请手动指定 --format。"
    )


def detect_endian(path: Path, fmt: str) -> str:
    """Infer the numeric byte order. FY-1B uses single-byte IDs in its header,
    so for 1B use scanline word structure rather than interpreting ID as uint16."""
    rec_size = RECORD_SIZES[fmt]
    if fmt == "hrpt1b":
        # The top two bits of every packed AVHRR word are padding and are zero.
        # On real FY-1D L1B files this is a much stronger test than the mixed
        # one-byte satellite/type fields in the second record.
        with path.open("rb") as f:
            f.seek(2 * rec_size)
            scan_record = f.read(rec_size)
        if len(scan_record) == rec_size:
            packed = scan_record[1000:28308]
            scores: Dict[str, float] = {}
            for order in (">", "<"):
                words = np.frombuffer(packed, dtype=np.dtype(order + "u4"))
                scores[order] = float(np.mean((words >> 30) == 0))
            if abs(scores[">"] - scores["<"]) > 0.03:
                return max(scores, key=scores.get)
        # Scanline prefix is line number, year, day-of-year, and milliseconds.
        with path.open("rb") as f:
            f.seek(2 * rec_size)
            prefix = f.read(10)
        for order in (">", "<"):
            if len(prefix) == 10:
                line_no, year, doy, millis = struct.unpack(order + "HHHI", prefix)
                if 1 <= line_no <= 10000 and 0 <= year <= 99 and 1 <= doy <= 366 and millis < 86400000:
                    return order
        return ">"

    with path.open("rb") as f:
        f.seek(rec_size if fmt == "hrpt1b" else 0)
        header_record = f.read(rec_size)
    if len(header_record) >= 2:
        be = struct.unpack_from(">H", header_record, 0)[0]
        le = struct.unpack_from("<H", header_record, 0)[0]
        if be in (113, 114) and le not in (113, 114):
            return ">"
        if le in (113, 114) and be not in (113, 114):
            return "<"
    return ">"


def decode_1b_samples(record: bytes, endian: str) -> np.ndarray:
    packed = np.frombuffer(record, dtype=dtype(endian, "u4"), count=(28308 - 1000) // 4, offset=1000)
    a = (packed >> 20) & 0x3FF
    b = (packed >> 10) & 0x3FF
    c = packed & 0x3FF
    values = np.empty(packed.size * 3, dtype=np.uint16)
    values[0::3] = a.astype(np.uint16)
    values[1::3] = b.astype(np.uint16)
    values[2::3] = c.astype(np.uint16)
    # 2048 pixels * 10 channels = 20480 values; the final packed word contains
    # only CH9 and CH10 according to the document.
    return values[:2048 * 10].reshape(2048, 10)


def reshape_hrpt_samples(record: bytes, endian: str, layout: str) -> np.ndarray:
    # Word 1601 through 22080 inclusive; the remaining 200 bytes are spare.
    values = np.frombuffer(record, dtype=dtype(endian, "u2"), count=20480, offset=3200).astype(np.uint16, copy=True)
    if layout == "pixel":
        return values.reshape(2048, 10)
    return values.reshape(10, 2048).T.copy()


def parse_line_1b(record: bytes, endian: str) -> Dict[str, Any]:
    line_number = read_i16(record, 0, endian, signed=False)
    year = read_i16(record, 2, endian)
    day = read_i16(record, 4, endian)
    milliseconds = read_i32(record, 6, endian, signed=False)
    quality_bytes = [record[10], record[11]]

    # The documented (2,10) coefficient array is stored as per-channel pairs:
    # slope(CH1), intercept(CH1), slope(CH2), intercept(CH2), ...
    coeffs = np.frombuffer(record, dtype=dtype(endian, "i4"), count=20, offset=16).astype(np.float64)
    slope = coeffs[0::2] / (2.0 ** 30)
    intercept = coeffs[1::2] / (2.0 ** 22)
    # These fields are stored as signed 16-bit values scaled by 128 degrees.
    sun_zenith = read_i16s(record, 96, 51, endian).astype(np.float64) / 128.0
    sat_zenith = read_i16s(record, 198, 51, endian).astype(np.float64) / 128.0
    rel_azimuth = read_i16s(record, 300, 51, endian).astype(np.float64) / 128.0
    geo = read_i16s(record, 402, 102, endian).astype(np.float64).reshape(51, 2) / 128.0
    # R*2(2,51): each tie point contains latitude followed by longitude.
    latitude = geo[:, 0]
    longitude = geo[:, 1]
    return {
        "line_number": line_number, "year": year, "day_of_year": day,
        "milliseconds_of_day": milliseconds,
        "quality": read_i16(record, 10, endian, signed=False),
        "quality_byte1": quality_bytes[0], "quality_byte2": quality_bytes[1],
        "slope": slope, "intercept": intercept,
        "sun_zenith": sun_zenith, "sat_zenith": sat_zenith,
        "relative_azimuth": rel_azimuth,
        "latitude": latitude, "longitude": longitude,
    }


def parse_line_hrpt1a5(record: bytes, endian: str, layout: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    line_number = read_i16(record, 0, endian, signed=False)
    year = read_i16(record, 2, endian)
    milliseconds = read_i32(record, 4, endian, signed=False)
    day = read_i16(record, 8, endian)
    quality = read_i16(record, 12, endian, signed=False)

    coeffs = read_f32s(record, 16, 20, endian).reshape(10, 2)
    slope = coeffs[:, 0].astype(np.float64)
    intercept = coeffs[:, 1].astype(np.float64)
    sun_zenith = read_f32s(record, 96, 51, endian).astype(np.float64)
    geo = read_f32s(record, 300, 102, endian).astype(np.float64).reshape(51, 2)
    latitude, longitude = geo[:, 0], geo[:, 1]
    sat_zenith = read_f32s(record, 1096, 51, endian).astype(np.float64)
    rel_azimuth = read_f32s(record, 1300, 51, endian).astype(np.float64)
    samples = reshape_hrpt_samples(record, endian, layout)
    meta = {
        "line_number": line_number, "year": year, "day_of_year": day,
        "milliseconds_of_day": milliseconds, "quality": quality,
        "slope": slope, "intercept": intercept,
        "sun_zenith": sun_zenith, "sat_zenith": sat_zenith,
        "relative_azimuth": rel_azimuth,
        "latitude": latitude, "longitude": longitude,
    }
    return samples, meta


def parse_line_gdpt1a5(record: bytes, endian: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    line_number = read_i16(record, 0, endian, signed=False)
    year = read_i16(record, 2, endian)
    milliseconds = read_i32(record, 4, endian, signed=False)
    day = read_i16(record, 8, endian)
    quality = read_i16(record, 12, endian, signed=False)

    coeffs = read_f32s(record, 16, 8, endian).reshape(4, 2)
    slope = coeffs[:, 0].astype(np.float64)
    intercept = coeffs[:, 1].astype(np.float64)
    sun_zenith = read_f32s(record, 48, 51, endian).astype(np.float64)
    geo = read_f32s(record, 252, 102, endian).astype(np.float64).reshape(51, 2)
    latitude, longitude = geo[:, 0], geo[:, 1]
    sat_zenith = read_f32s(record, 836, 51, endian).astype(np.float64)
    rel_azimuth = read_f32s(record, 1040, 51, endian).astype(np.float64)
    values = np.frombuffer(record, dtype=dtype(endian, "u2"), count=4072, offset=1400).astype(np.uint16, copy=True)
    samples = values.reshape(1018, 4)
    meta = {
        "line_number": line_number, "year": year, "day_of_year": day,
        "milliseconds_of_day": milliseconds, "quality": quality,
        "slope": slope, "intercept": intercept,
        "sun_zenith": sun_zenith, "sat_zenith": sat_zenith,
        "relative_azimuth": rel_azimuth,
        "latitude": latitude, "longitude": longitude,
    }
    return samples, meta


def parse_header(record: bytes, fmt: str, endian: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "format": FORMAT_LABELS[fmt],
        "record_length_bytes": RECORD_SIZES[fmt],
        "byte_order": "big-endian" if endian == ">" else "little-endian",
        "header_record_bytes": len(record),
    }
    if fmt == "hrpt1b":
        # In the 1B data-header record the first two fields are single bytes:
        # satellite ID (113/114), then data type. Subsequent numeric fields are
        # big/little-endian multi-byte values according to the scanline layout.
        sat_id = record[0] if len(record) >= 1 else -1
        result["satellite_id"] = sat_id
        result["data_type"] = record[1] if len(record) >= 2 else None
        result["satellite"] = {113: "FY-1C", 114: "FY-1D"}.get(sat_id, "unknown/format-specific")
    elif len(record) >= 2:
        sat_id = read_i16(record, 0, endian, signed=False)
        result["satellite_id_raw_u16"] = sat_id
        result["satellite"] = {113: "FY-1C", 114: "FY-1D"}.get(sat_id, "unknown/format-specific")
    # The 1A.5 header uses 2-byte word positions in the PDF table. The 1B data
    # header's offset/count notation is inconsistent in the source, so do not
    # attach possibly wrong decoded time fields to that format.
    if fmt in ("hrpt1a5", "gdpt1a5") and len(record) >= 10:
        result["year_start_raw"] = read_i16(record, 2, endian)
        result["milliseconds_start_raw"] = read_i32(record, 4, endian, signed=False)
        result["day_of_year_start_raw"] = read_i16(record, 8, endian, signed=False)
    elif fmt == "hrpt1b" and len(record) >= 20:
        result["year_start"] = read_i16(record, 2, endian, signed=False)
        result["day_of_year_start"] = read_i16(record, 4, endian, signed=False)
        result["milliseconds_start"] = read_i32(record, 6, endian, signed=False)
        result["scanline_count_header"] = read_i16(record, 10, endian, signed=False)
        result["year_end"] = read_i16(record, 12, endian, signed=False)
        result["day_of_year_end"] = read_i16(record, 14, endian, signed=False)
        result["milliseconds_end"] = read_i32(record, 16, endian, signed=False)
    return result


def percentile_preview(channel_image: np.ndarray) -> np.ndarray:
    values = channel_image.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(values[finite], [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values[finite]))
        hi = float(np.nanmax(values[finite]))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) * (255.0 / (hi - lo))
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def export_outputs(
    out_dir: Path, fmt: str, endian: str, input_path: Path,
    raw: np.ndarray, line_meta: List[Dict[str, Any]], header_info: Dict[str, Any],
    make_png: bool,
    make_calibrated: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = int(raw.shape[2])
    samples = int(raw.shape[1])

    # Build compact arrays for downstream analysis in NumPy/Satpy-adjacent tools.
    def stack_field(key: str, dtype_: Any = None) -> np.ndarray:
        vals = [m[key] for m in line_meta]
        arr = np.asarray(vals, dtype=dtype_)
        return arr

    npz_fields: Dict[str, Any] = {
        "raw": raw,
        "line_number": stack_field("line_number", np.int32),
        "year": stack_field("year", np.int32),
        "day_of_year": stack_field("day_of_year", np.int32),
        "milliseconds_of_day": stack_field("milliseconds_of_day", np.uint32),
        "quality": stack_field("quality", np.uint16),
        "slope": stack_field("slope", np.float64),
        "intercept": stack_field("intercept", np.float64),
        "sun_zenith_deg": stack_field("sun_zenith", np.float64),
        "sat_zenith_deg": stack_field("sat_zenith", np.float64),
        "relative_azimuth_deg": stack_field("relative_azimuth", np.float64),
        "latitude_tiepoints_deg": stack_field("latitude", np.float64),
        "longitude_tiepoints_deg": stack_field("longitude", np.float64),
    }
    np.savez_compressed(out_dir / "fy1_decoded.npz", **npz_fields)
    np.save(out_dir / "raw_counts.npy", raw)

    meta_path = out_dir / "metadata.json"
    combined = dict(header_info)
    combined.update({
        "input_file": str(input_path.resolve()),
        "format_id": fmt,
        "scanline_count": int(raw.shape[0]),
        "samples_per_scanline": samples,
        "channel_count": channels,
        "raw_array_shape": list(raw.shape),
        "raw_array_order": "[scanline, sample, channel]",
        "channels_numbered_from": 1,
        "sample_type": "unsigned integer counts; no radiometric unit implied",
        "note": "Latitude/longitude are 51 tie points per line, not a full pixel geolocation grid.",
    })
    meta_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    line_csv_fields = ["record_index", "line_number", "year", "day_of_year", "milliseconds_of_day", "quality"]
    if fmt == "hrpt1b":
        line_csv_fields += ["quality_byte1", "quality_byte2"]
    with (out_dir / "scanlines.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=line_csv_fields)
        writer.writeheader()
        for i, m in enumerate(line_meta):
            writer.writerow({k: (i if k == "record_index" else m.get(k, "")) for k in line_csv_fields})

    with (out_dir / "geolocation_tiepoints.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["record_index", "line_number", "tiepoint_index", "latitude_deg", "longitude_deg",
                  "sun_zenith_deg", "sat_zenith_deg", "relative_azimuth_deg"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for line_idx, m in enumerate(line_meta):
            for p in range(51):
                writer.writerow({
                    "record_index": line_idx,
                    "line_number": m["line_number"],
                    "tiepoint_index": p,
                    "latitude_deg": float(m["latitude"][p]),
                    "longitude_deg": float(m["longitude"][p]),
                    "sun_zenith_deg": float(m["sun_zenith"][p]),
                    "sat_zenith_deg": float(m["sat_zenith"][p]),
                    "relative_azimuth_deg": float(m["relative_azimuth"][p]),
                })

    if make_png:
        if Image is None:
            print("警告：未安装 Pillow，跳过 PNG 预览。安装命令：python -m pip install pillow", file=sys.stderr)
        else:
            for ch in range(channels):
                preview = percentile_preview(raw[:, :, ch])
                Image.fromarray(preview, mode="L").save(out_dir / f"channel_{ch + 1:02d}.png")

    # Optional: applying coefficients is not the same as validating the full
    # radiometric calibration procedure; keep this large operation opt-in.
    if make_calibrated:
        try:
            slopes = npz_fields["slope"]
            intercepts = npz_fields["intercept"]
            calibrated = raw.astype(np.float32) * slopes[:, None, :].astype(np.float32) + intercepts[:, None, :].astype(np.float32)
            np.savez_compressed(out_dir / "calibrated_values.npz", values=calibrated)
        except Exception as exc:
            print(f"警告：未能生成按定标系数计算的数值：{exc}", file=sys.stderr)


def decode_file(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    fmt = args.format if args.format != "auto" else detect_format(input_path)
    rec_size = RECORD_SIZES[fmt]
    file_size = input_path.stat().st_size
    if file_size % rec_size:
        raise ValueError(
            f"文件大小 {file_size} 不是 {fmt} 记录长度 {rec_size} 的整数倍。"
            "文件可能不完整，或格式选择不正确。"
        )
    record_count = file_size // rec_size
    min_records = 3 if fmt == "hrpt1b" else 2
    if record_count < min_records:
        raise ValueError(f"{fmt} 至少需要 {min_records} 条记录，当前只有 {record_count} 条。")

    endian = {"big": ">", "little": "<"}.get(args.endian)
    if endian is None:
        endian = detect_endian(input_path, fmt)
    labels = {">": "big-endian", "<": "little-endian"}
    print(f"格式：{FORMAT_LABELS[fmt]}")
    print(f"字节序：{labels[endian]}")
    print(f"文件大小：{file_size:,} 字节；记录数：{record_count:,}")

    with input_path.open("rb") as f:
        first_record = f.read(rec_size)
        if fmt == "hrpt1b":
            tbm_header = first_record
            data_header = f.read(rec_size)
            if len(data_header) != rec_size:
                raise ValueError("1B 文件缺少第二条数据头记录。")
            header_info = parse_header(data_header, fmt, endian)
            header_info["tbm_header_bytes"] = len(tbm_header)
            header_info["data_header_bytes"] = len(data_header)
            scan_count = record_count - 2
        else:
            header_info = parse_header(first_record, fmt, endian)
            scan_count = record_count - 1

        if fmt == "hrpt1b" and "scanline_count_header" in header_info:
            declared = int(header_info["scanline_count_header"])
            header_info["scanline_count_header_matches_records"] = (declared == scan_count)
            header_info["scanline_count_header_difference"] = declared - scan_count
            if declared != scan_count:
                print(
                    f"注意：数据头声明扫描线数为 {declared}，但文件中实际有 {scan_count} 条扫描线记录；保留全部实际记录。",
                    file=sys.stderr,
                )
        if scan_count <= 0:
            raise ValueError("没有找到扫描线数据记录。")
        if fmt == "gdpt1a5":
            samples_per_line, channel_count = 1018, 4
        else:
            samples_per_line, channel_count = 2048, 10
        raw = np.empty((scan_count, samples_per_line, channel_count), dtype=np.uint16)
        all_meta: List[Dict[str, Any]] = []
        for idx in range(scan_count):
            record = f.read(rec_size)
            if len(record) != rec_size:
                raise ValueError(f"第 {idx + 1} 条扫描线记录不完整，解码中止。")
            try:
                if fmt == "hrpt1b":
                    meta = parse_line_1b(record, endian)
                    samples = decode_1b_samples(record, endian)
                elif fmt == "hrpt1a5":
                    samples, meta = parse_line_hrpt1a5(record, endian, args.layout)
                else:
                    samples, meta = parse_line_gdpt1a5(record, endian)
            except (ValueError, struct.error, IndexError) as exc:
                raise ValueError(f"解码第 {idx + 1} 条扫描线时失败：{exc}") from exc
            raw[idx] = samples
            all_meta.append(meta)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else input_path.with_name(input_path.stem + "_decoded")
    export_outputs(out_dir, fmt, endian, input_path, raw, all_meta, header_info, not args.no_png, args.calibrated)

    print("\n解码完成：")
    print(f"  扫描线：{raw.shape[0]}")
    print(f"  每线像元：{raw.shape[1]}")
    print(f"  通道数：{raw.shape[2]}")
    print(f"  原始数组：{raw.shape}（扫描线, 像元, 通道）")
    print(f"  输出目录：{out_dir.resolve()}")
    print("  文件：fy1_decoded.npz、raw_counts.npy、scanlines.csv、geolocation_tiepoints.csv、metadata.json")
    if not args.no_png:
        print("  预览：channel_01.png ...")
    if fmt == "hrpt1a5":
        print("提示：如果 HRPT 1A.5 预览出现明显错位，可尝试 --layout channel。")
    if args.calibrated:
        print("提示：calibrated_values.npz 是按斜率/截距直接计算的结果；物理单位及完整定标流程仍需结合产品说明确认。")
    else:
        print("提示：默认只导出原始计数值；需要额外计算斜率/截距时使用 --calibrated。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解码风云一号 FY-1C/D AVHRR 1B、HRPT 1A.5 和 GDPT 1A.5 二进制数据。"
    )
    parser.add_argument("input", help="输入二进制数据文件路径")
    parser.add_argument("-o", "--out-dir", help="输出目录；默认在输入文件旁创建 <文件名>_decoded")
    parser.add_argument("--format", choices=["auto", "hrpt1b", "hrpt1a5", "gdpt1a5"], default="auto",
                        help="数据格式，默认按文件长度自动识别")
    parser.add_argument("--endian", choices=["auto", "big", "little"], default="auto",
                        help="字节序，默认根据头标识/1B 保留位自动判断，无法判断时按大端处理")
    parser.add_argument("--layout", choices=["pixel", "channel"], default="pixel",
                        help="HRPT 1A.5 的样本排列；pixel=每像元交错 10 通道，channel=按通道分别连续存储")
    parser.add_argument("--no-png", action="store_true", help="不生成 PNG 灰度预览")
    parser.add_argument("--calibrated", action="store_true", help="额外导出按斜率/截距计算的 calibrated_values.npz（较耗时，数值单位需另行确认）")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        decode_file(args)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
