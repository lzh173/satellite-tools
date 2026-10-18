#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fy2_svissr.py -- 风云二号（02 批）静止气象卫星 S-VISSR 2.0 展宽数据解码器
==========================================================================

依据文档：国家卫星气象中心（NSMC）《风云二号（02 批）静止气象卫星数据格式》
（fy2_02_1b_format.pdf，即 FY-2C/D/E/F/G/H S-VISSR2.0 一级数据格式说明）

------------------------------------------------------------------ 一帧的结构
（表 4.5 / 表 4.6，一帧 = 一条扫描线，含 1 行红外 + 4 行可见光）

    同步码 10000 bit  |  12 个信息段（354848 bit，即 44356 字节）

    段号 段名       ID(bit) 数据(bit)      CRC(16) 填充(2048)  合计(bit)
     1   文件段DOC   16    18328 (2291B)      16     2048      20408
     2   IR1 高 8 位  16    18328 (2291x8)     16     2048      20408
     3   IR2 高 8 位  16    18328 (2291x8)     16     2048      20408
     4   IR3 高 8 位  16    18328 (2291x8)     16     2048      20408
     5   VIS1        12    54984 (9164x6)     16     2048      57060
     6   VIS2        12    54984 (9164x6)     16     2048      57060
     7   VIS3        12    54984 (9164x6)     16     2048      57060
     8   VIS4        12    54984 (9164x6)     16     2048      57060
     9   IR1 低 2 位  16     4582 (2291x2)     16     2048       6662
    10   IR2 低 2 位  16     4582 (2291x2)     16     2048       6662
    11   IR3 低 2 位  16     4582 (2291x2)     16     2048       6662
    12   IR4        16    22910 (2291x10)    16     2048      24990
                                                        ----------------
                                                        合计 354848 bit

各段识别码（表 4.17 / 4.18）：DOC=0000h, IR1H=1111h, IR2H=2222h, IR3H=4444h,
VIS1=6DBh, VIS2=B6Dh, VIS3=DB6h, VIS4=FFFh, IR1L=8888h, IR2L=9999h,
IR3L=AAAAh, IR4=BBBBh。

------------------------------------------------------------------ 编码链路
（3.1.3 节）发端：有效信息 →(1) 字节取反（同步码之后，每隔 8 bit 取反一次，
即对偶数序号字节取反）→(2) 与 15 位 PN 码逐位异或（含同步码）→ DPSK 调制。
收端顺序相反。解码器可自动识别输入是"已解扰的净帧"还是"含同步码的原始码流"。

图像数据约定：
  * 红外 IR1~IR3   10 bit 量化 = 高 8 位段 + 低 2 位段拼接，2291 像元/行，5 km
  * 红外 IR4       10 bit，2291 像元/行，5 km
  * 可见光 VIS1~4  6 bit，9164 像元/行，1.25 km；每次扫描得 4 条可见光扫描线

命令行用法见 `python fy2_svissr.py -h`。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================== 1. 格式常量
FRAME_BITS = 354848            # 一帧有效信息 354848 bit（表 4.5）
FRAME_BYTES = FRAME_BITS // 8  # = 44356 字节
SYNC_BITS = 10000              # 同步码 10000 bit（= 1250 字节）
SYNC_BYTES = SYNC_BITS // 8
FILL_BITS = 2048               # 每段尾部填充码 2048 bit
CRC_BITS = 16

IR_COLS = 2291                 # 红外：每行像元数
VIS_COLS = 9164                # 可见光：每行像元数
IR_CHANNELS = 4
VIS_CHANNELS = 4

# 段表：(名字, ID 位数, ID 值, 数据位数, 类型, 通道号)
SEGMENTS: List[Tuple[str, int, int, int, str, int]] = [
    ("DOC",  16, 0x0000, 18328, "doc",   -1),
    ("IR1H", 16, 0x1111, 18328, "ir_hi",  0),
    ("IR2H", 16, 0x2222, 18328, "ir_hi",  1),
    ("IR3H", 16, 0x4444, 18328, "ir_hi",  2),
    ("VIS1", 12, 0x6DB,  54984, "vis",    0),
    ("VIS2", 12, 0xB6D,  54984, "vis",    1),
    ("VIS3", 12, 0xDB6,  54984, "vis",    2),
    ("VIS4", 12, 0xFFF,  54984, "vis",    3),
    ("IR1L", 16, 0x8888,  4582, "ir_lo",  0),
    ("IR2L", 16, 0x9999,  4582, "ir_lo",  1),
    ("IR3L", 16, 0xAAAA,  4582, "ir_lo",  2),
    ("IR4",  16, 0xBBBB, 22910, "ir_hi",  3),
]

# 计算每段的比特偏移（相对一帧起点）
SEG_OFFSETS: List[int] = []
_o = 0
for _s in SEGMENTS:
    SEG_OFFSETS.append(_o)
    _o += _s[1] + _s[3] + CRC_BITS + FILL_BITS
assert _o == FRAME_BITS, f"段长度合计 {_o} != {FRAME_BITS}"

SAT_NAMES = {0x23: "FY-2C", 0x24: "FY-2D", 0x25: "FY-2E",
             0x26: "FY-2F", 0x27: "FY-2G", 0x28: "FY-2H"}

SCAN_MODES = {0x00: "全圆盘扫描", 0xF0: "机动扫描", 0xFF: "单线扫描"}

# 扫描状态字（表 4.8 第 2 字，b8..b1）常见值
SCAN_STATUS_DESC = {
    0x33: "正扫：自北向南逐行（正常步进）",
    0xCC: "回扫：自南向北快速步进",
    0x00: "无动作",
}


def scan_status_desc(v: int) -> str:
    """按表 4.8 解释扫描状态字。"""
    if v in SCAN_STATUS_DESC:
        return SCAN_STATUS_DESC[v]
    parts = []
    if v & 0b00000011 == 0b11:
        parts.append("正向扫描(北→南)")
    if v & 0b00001100 == 0b00001100:
        parts.append("反向扫描(南→北)")
    if v & 0b00110000 == 0b00110000:
        parts.append("正常步进")
    if v & 0b11000000 == 0b11000000:
        parts.append("快速步进(10 步/周)")
    return "、".join(parts) if parts else "无动作"


def is_backward_scan(scan_status: int) -> bool:
    """回扫判据（与 SatDump 一致）：扫描状态字低 2 位不为 11，即不是"自北向南正向扫描"。"""
    return (scan_status & 0b11) == 0

CH_NAMES = {0: "IR1", 1: "IR2", 2: "IR3", 3: "IR4"}
CH_LABELS = {
    "IR1": "IR1 长波红外 10.3-11.3 µm",
    "IR2": "IR2 红外分裂窗 11.5-12.5 µm",
    "IR3": "IR3 水汽 6.3-7.6 µm",
    "IR4": "IR4 中波红外 3.5-4.0 µm",
    "VIS": "VIS 可见光 0.55-0.90 µm",
}

# 填充 6bit/2bit/10bit 数据用的权重
W6 = np.array([32, 16, 8, 4, 2, 1], np.uint8)
W2 = np.array([2, 1], np.uint8)
W10 = np.array([512, 256, 128, 64, 32, 16, 8, 4, 2, 1], np.uint16)


# ============================================================== 2. 位/字段工具
def bcd_to_int(b) -> Optional[int]:
    """BCD*n -> int；任一半字节 >9 视为无效（返回 None）。"""
    v = 0
    for x in bytes(b):
        hi, lo = x >> 4, x & 0xF
        if hi > 9 or lo > 9:
            return None
        v = v * 100 + hi * 10 + lo
    return v


def bcd_bytes(n: int) -> bytes:
    """int -> 定长 BCD（调试/合成用）。"""
    s = f"{n:0{len(str(n))}d}"
    return bytes(int(s[i:i + 2], 16) for i in range(0, len(s), 2))


def int_n(b: bytes) -> int:
    """I*n：n 字节补码整数。"""
    return int.from_bytes(b, "big", signed=True)


def uint_n(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=False)


def real_nm(b: bytes, m: int) -> float:
    """R*n.m：首位为符号的补码整数 × 10^-m（3.1.4 节）。"""
    return int_n(b) * (10.0 ** (-m))


def u12(b: bytes) -> int:
    """12 bit 无符号数：首字节低 4 位 + 次字节（表 4.8 第 11~12 字等）。"""
    return ((b[0] & 0x0F) << 8) | b[1]


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """G(X)=X^16+X^12+X^5+1（CRC-16/CCITT，MSB first，无最终取反）。"""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """把 0/1 数组补齐到整字节后打包。"""
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    return bytes(np.packbits(bits).tolist())


def bits_to_int(bits: np.ndarray) -> int:
    """0/1 数组 -> 整数（MSB first），支持非 8 整数倍长度（如 12 bit ID）。"""
    v = 0
    for b in bits.tolist():
        v = (v << 1) | int(b)
    return v


def pn_sequence(nbits: int, seed: int = 0b011001110011111) -> np.ndarray:
    """15 位串行移位寄存器 PN 码发生器（图 4.1）：反馈 = 第 14 位 XOR 第 15 位。"""
    reg = seed & 0x7FFF
    out = np.empty(nbits, np.uint8)
    for i in range(nbits):
        fb = ((reg >> 13) & 1) ^ ((reg >> 14) & 1)
        out[i] = fb
        reg = ((reg << 1) | fb) & 0x7FFF
    return out


def pn_descramble(frame_bits: np.ndarray) -> np.ndarray:
    """PN 解扰（对整帧逐位异或，与发端对称）。"""
    return frame_bits ^ pn_sequence(len(frame_bits))


def invert_even_bytes(buf: bytes) -> bytes:
    """字节取反：对偶数序号字节取反（3.1.3 节第 1 步）。"""
    arr = np.frombuffer(buf, np.uint8).copy()
    arr[0::2] ^= 0xFF
    return arr.tobytes()


# ============================================================== 3. 字段表
# 表 4.8  卫星和 CDAS 工作状态块（126 字节，块内 1 基偏移）
DOC_FIELDS: List[Tuple[int, int, str, str, str]] = [
    # (块内1基位置, 字节数, 类型, 字段名, 中文说明)
    (1,   1, "u8",   "scan_mode",           "扫描方式"),
    (2,   1, "u8",   "scan_status",         "扫描辐射计扫描状态"),
    (3,   1, "u8",   "frame_flag",          "帧标志"),
    (4,   1, "u8",   "image_flag",          "图像标志"),
    (5,   2, "bcd",  "image_start_line",    "图像开始扫描行号"),
    (7,   2, "bcd",  "image_end_line",      "图像结束扫描行号"),
    (9,   2, "bcd",  "image_line_count",    "图像有效行计数"),
    (11,  2, "u12",  "west_horizon",        "西地平点"),
    (13,  2, "u12",  "east_horizon",        "东地平点"),
    (15,  1, "u8",   "dpl_lock",            "DPL 锁定标记"),
    (16,  2, "u12",  "raw_bit_errors",      "原始图像比特误码计数"),
    (18,  2, "bcd",  "year",                "年"),
    (20,  1, "bcd",  "month",               "月"),
    (21,  1, "bcd",  "day",                 "日"),
    (22,  1, "bcd",  "hour",                "时"),
    (23,  1, "bcd",  "minute",              "分"),
    (24,  1, "bcd",  "second",              "秒"),
    (25,  1, "bcd",  "millisec",            "1/100 秒"),
    (26,  2, "u16",  "cal_table_version",   "更新定标表的累积计数"),
    (28,  2, "u16",  "manam_version",       "更新 MANAM 表的累积计数"),
    (30,  1, "u8",   "image_source",        "展宽数据来源"),
    (31,  1, "u8",   "potentiometer1",      "电位器 1"),
    (32,  1, "u8",   "potentiometer2",      "电位器 2"),
    (33, 33, "bytes", "spare_1",            "备用"),
    (65,  1, "u8",   "reserved_65",         "保留"),
    (66,  2, "u12",  "visr_line_count",     "VISSR 仪器扫描行计数"),
    (68,  1, "u8",   "sensor_select",       "传感器选择参数"),
    (69,  1, "u8",   "vis_order",           "可见光传感器排列顺序"),
    (70,  3, "u24",  "beta_count",          "β 角计数"),
    (73,  3, "u24",  "spin_period_count",   "自旋周期计数"),
    (76,  3, "bytes", "spare_2",            "备用"),
    (79,  3, "u24",  "raw_clock_count",     "原始图像时钟频率计数"),
    (82,  6, "bytes", "spare_3",            "备用"),
    (88,  1, "u8",   "resample_mode",       "重采样方式"),
    (89,  1, "u8",   "image_reference",     "星上图像基准"),
    (90,  1, "u8",   "satellite_id",        "卫星标志"),
    (91,  3, "u24",  "sun_pulse_error",     "模拟太阳脉冲角跟踪误差 ΔSp"),
    (94,  3, "u24",  "dpl_error",           "DPL 数字跟踪误差 ΔN"),
    (97,  2, "bytes", "spare_4",            "备用"),
    (99,  1, "u8",   "pos_update_flag",     "定位信息更新标志"),
    (100, 7, "bcd",  "pos_update_time",     "定位信息更新时间 YYYYMMDDHHmmss"),
    (107, 2, "u16",  "ias_cycle_count",     "地面站独立循环计数"),
    (109, 2, "bytes", "spare_5",            "备用"),
    (111, 2, "u16",  "n_value",             "N 值"),
    (113, 14, "bytes", "spare_6",           "备用"),
]

# 表 4.9  常数块（64 字节）
CONST_FIELDS: List[Tuple[int, int, str, str]] = [
    (1,  4, "i32",    "earth_equatorial_radius_m"),
    (5,  4, "i32",    "satellite_nominal_height_m"),
    (9,  4, "i32",    "ir_step_angle_nrad"),
    (13, 4, "i32",    "ir_sample_angle_nrad"),
    (17, 4, "i32",    "subsat_lat_mdeg"),
    (21, 4, "i32",    "subsat_lon_mdeg"),
    (25, 4, "i32",    "ir1_nadir_row"),
    (29, 4, "i32",    "ir1_nadir_col"),
    (33, 4, "r7",     "pi"),
    (37, 4, "r2",     "vis_row_offset"),
    (41, 4, "r2",     "vis_col_offset"),
    (45, 4, "r2",     "ir2_row_offset"),
    (49, 4, "r2",     "ir2_col_offset"),
    (53, 4, "r2",     "ir3_row_offset"),
    (57, 4, "r2",     "ir3_col_offset"),
    (61, 4, "r6",     "earth_flattening_inverse"),
]

# 文件段内各子交换块的 0 基偏移（表 4.7，相对"文件段数据块"= 不含 2 字节段识别码）
# 文件段布局：ID 2B | 卫星/CDAS 126B | 常数 64B | 子交换 4B | 网格 100B |
#             轨道姿态 128B | MANAM 410B | 定标1 256B | 定标2 1024B | 备用 179B
DOC_SAT = (0, 126)          # 卫星和 CDAS 工作状态块 126 B
DOC_CONST = (126, 190)      # 常数块 64 B
DOC_SUBSWAP = (190, 194)    # 子交换标志 4 B
DOC_GRID = (194, 294)       # 简化网格块 100 B/组
DOC_ORBIT = (294, 422)      # 轨道和姿态数据块 128 B/组
DOC_MANAM = (422, 832)      # 卫星周工作日程表 410 B/组
DOC_CAL1 = (832, 1088)      # 定标数据块 1  256 B/组
DOC_CAL2 = (1088, 2112)     # 定标数据块 2  1024 B/组
DOC_SPARE = (2112, 2291)    # 备用块 179 B
assert DOC_SPARE[1] == 2291


# ============================================================== 4. 帧解析
@dataclass
class Frame:
    index: int
    doc: Dict[str, object]
    const: Dict[str, Optional[float]]
    groups: Dict[str, bytes]
    sub_group: int
    sub_repeat: int
    ir: Optional[np.ndarray] = None   # (4, 2291) uint16
    vis: Optional[np.ndarray] = None  # (4, 9164) uint8
    id_ok: int = 0
    crc_ok: int = 0
    row: Optional[int] = None         # 绝对扫描行号（1 基）

    @property
    def time(self) -> Optional[_dt.datetime]:
        d = self.doc
        try:
            return _dt.datetime(int(d["year"]), int(d["month"]), int(d["day"]),
                                int(d["hour"]), int(d["minute"]), int(d["second"]),
                                int(d["millisec"]) * 10000)
        except (TypeError, ValueError):
            return None


def _decode_doc_block(blk: bytes) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for pos, n, typ, name, _cn in DOC_FIELDS:
        raw = blk[pos - 1:pos - 1 + n]
        if len(raw) != n:
            out[name] = None
            continue
        if typ == "bcd":
            out[name] = bcd_to_int(raw)
        elif typ in ("u8", "u16", "u24"):
            out[name] = uint_n(raw)
        elif typ == "u12":
            out[name] = u12(raw)
        else:
            out[name] = None
    out["raw"] = blk
    return out


def _decode_const_block(blk: bytes) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for pos, n, typ, name in CONST_FIELDS:
        raw = blk[pos - 1:pos - 1 + n]
        if len(raw) != n:
            out[name] = None
        elif typ == "i32":
            out[name] = int_n(raw)
        elif typ == "r7":
            out[name] = real_nm(raw, 7)
        elif typ == "r6":
            out[name] = real_nm(raw, 6)
        elif typ == "r2":
            out[name] = real_nm(raw, 2)
        else:
            out[name] = None
    return out


def parse_frame(frame_bytes: bytes, index: int = 0,
                with_images: bool = True, check_crc: bool = False) -> Frame:
    """解析一帧（354848 bit = 44356 字节，不含同步码）。"""
    fb = np.unpackbits(np.frombuffer(frame_bytes, np.uint8))

    id_ok = 0
    crc_ok = 0
    ir = np.zeros((IR_CHANNELS, IR_COLS), np.uint16) if with_images else None
    vis = np.zeros((VIS_CHANNELS, VIS_COLS), np.uint8) if with_images else None

    for (name, idbits, idval, dbits, kind, ch), off in zip(SEGMENTS, SEG_OFFSETS):
        ident = bits_to_int(fb[off:off + idbits])
        # ID 允许少量误码：汉明距离 <= 2 视为命中
        if bin(ident ^ idval).count("1") <= 2:
            id_ok += 1
        data = fb[off + idbits: off + idbits + dbits]

        if check_crc:
            crc_rx = int.from_bytes(
                bits_to_bytes(fb[off + idbits + dbits: off + idbits + dbits + CRC_BITS]), "big")
            if crc16_ccitt(bits_to_bytes(fb[off: off + idbits + dbits])) == crc_rx:
                crc_ok += 1

        if kind == "doc" or not with_images:
            continue                       # 文件段由 parse_file_segment 按字节解析
        if kind == "ir_hi":
            if ch == 3:                       # IR4：完整的 2291×10 bit
                ir[3] = (data.reshape(IR_COLS, 10) @ W10).astype(np.uint16)
            else:                             # IR1~IR3：先取高 8 位，左移 2 位
                ir[ch] = np.packbits(data).astype(np.uint16) << 2
        elif kind == "ir_lo":
            ir[ch] |= (data.reshape(IR_COLS, 2) @ W2).astype(np.uint16)
        elif kind == "vis":
            vis[ch] = data.reshape(VIS_COLS, 6) @ W6

    doc, const, groups, sg, sr = parse_file_segment(frame_bytes)
    return Frame(index=index, doc=doc, const=const, groups=groups,
                 sub_group=sg, sub_repeat=sr, ir=ir, vis=vis,
                 id_ok=id_ok, crc_ok=crc_ok)


# ============================================================== 5. 码流识别
def detect_layout(path: str, max_probe: int = 8) -> Dict[str, object]:
    """
    自动识别输入码流结构，返回 dict：
        stride_bytes  每帧字节数（44356 = 已去同步码；45606 = 含 10000 bit 同步码）
        skip_bytes    帧内跳过字节数（同步码 1250 字节）
        transform     变换：none / pn / invert / pn+invert
        score         命中的段 ID 数（最大 12）
    判定依据：12 个段识别码在正确偏移上必须基本全部命中。
    """
    size = os.path.getsize(path)
    best = {"stride_bytes": FRAME_BYTES, "skip_bytes": 0, "transform": "none", "score": -1}
    with open(path, "rb") as f:
        head = f.read(min(size, 45606 * 2 + 16))

    cands = [(FRAME_BYTES, 0), (FRAME_BYTES + SYNC_BYTES, SYNC_BYTES)]
    for stride, skip in cands:
        if len(head) < stride:
            continue
        for transform in ("none", "pn", "invert", "pn+invert"):
            raw = head[skip:skip + stride]
            if "invert" in transform:
                raw = invert_even_bytes(raw)
            if "pn" in transform:
                b = np.unpackbits(np.frombuffer(raw, np.uint8))
                raw = bits_to_bytes(pn_descramble(b))
            fb = np.unpackbits(np.frombuffer(raw, np.uint8))
            score = 0
            for (name, idbits, idval, dbits, kind, ch), off in zip(SEGMENTS, SEG_OFFSETS):
                ident = bits_to_int(fb[off:off + idbits])
                if bin(ident ^ idval).count("1") <= 2:
                    score += 1
            if score > best["score"]:
                best = {"stride_bytes": stride, "skip_bytes": skip,
                        "transform": transform, "score": score}
    return best


def apply_transform(buf: bytes, transform: str) -> bytes:
    if "invert" in transform:
        buf = invert_even_bytes(buf)
    if "pn" in transform:
        b = np.unpackbits(np.frombuffer(buf, np.uint8))
        buf = bits_to_bytes(pn_descramble(b))
    return buf


def _clean_lut(lut: np.ndarray, decreasing: bool) -> np.ndarray:
    """定标表清洗：中值滤波去掉孤立误码点，再按物理单调性约束。"""
    if lut.size < 5:
        return lut
    pad = np.pad(lut.astype(np.float64), 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 5)
    med = np.median(win, axis=-1)
    return (np.minimum.accumulate(med) if decreasing
            else np.maximum.accumulate(med)).astype(np.float32)


# ============================================================== 6. 标定
@dataclass
class Calibration:
    """定标查找表（表 4.15 / 4.16）。"""
    vis: Dict[int, np.ndarray] = field(default_factory=dict)   # 64 级 -> 反照率
    ir: Dict[int, np.ndarray] = field(default_factory=dict)    # 1024 级 -> 温度 K
    info: Dict[str, object] = field(default_factory=dict)
    source: Optional[str] = None

    @classmethod
    def from_groups(cls, best: Dict[str, Dict[int, bytes]], src: str) -> "Calibration":
        cal = cls(source=src)
        table = best.get(src, {})
        if len(table) < 25:
            return cal
        blob = b"".join(table[g] for g in range(25))
        stride = 256 if src == "cal1" else 1024
        n_ir = 256 if src == "cal1" else 1024
        if len(blob) < stride * 25:
            return cal
        cal.info = {
            "flag": blob[0:4].hex(),
            "generated": bcd_to_int(blob[4:10]),
            "sensor": blob[10],
        }
        for i in range(4):
            base = 256 + i * 256
            vals = np.array([real_nm(blob[base + j * 4: base + j * 4 + 4], 6)
                             for j in range(64)], np.float32)
            cal.vis[i] = _clean_lut(vals, decreasing=False)
        for i in range(4):
            base = 1280 + i * (n_ir * 4)
            vals = np.array([real_nm(blob[base + j * 4: base + j * 4 + 4], 3)
                             for j in range(n_ir)], np.float32)
            cal.ir[i] = _clean_lut(vals, decreasing=True)
        return cal

    def ir_bt(self, ch: int, counts: np.ndarray) -> np.ndarray:
        lut = self.ir.get(ch)
        if lut is None:
            return counts.astype(np.float32)
        if lut.size == 1024:
            idx = np.clip(counts.astype(np.int32), 0, 1023)
        else:  # 8 bit 简略表，等间隔对应 0,4,8,...,1020
            idx = np.clip((counts.astype(np.int32)) // 4, 0, lut.size - 1)
        return lut[idx]

    def vis_albedo(self, ch: int, counts: np.ndarray) -> np.ndarray:
        lut = self.vis.get(ch)
        if lut is None:
            return counts.astype(np.float32)
        return lut[np.clip(counts.astype(np.int32), 0, lut.size - 1)]


def _vote_bytes(copies: List[bytes]) -> bytes:
    """对同一子交换块的多次重复做逐比特多数表决（纠随机误码）。"""
    if len(copies) == 1:
        return copies[0]
    n = len(copies[0])
    arr = np.frombuffer(b"".join(c[:n] for c in copies), np.uint8).reshape(len(copies), n)
    bits = np.unpackbits(arr, axis=1)
    maj = (bits.sum(axis=0).astype(np.int32) * 2 >= len(copies)).astype(np.uint8)
    return np.packbits(maj).tobytes()


def parse_grid(blob: bytes) -> np.ndarray:
    """
    表 4.11 简化网格块：2500 字节 = 25×25 个网格点的 (行号, 列号)，各 I*2。
    网格范围 60°N~60°S、45°E~165°E，间隔 5°。返回 (25, 25, 2) float，无效为 NaN。
    """
    if len(blob) < 2500:
        return np.full((25, 25, 2), np.nan, np.float32)
    a = np.frombuffer(blob[:2500], ">i2").astype(np.float32).reshape(25, 25, 2)
    a[(a < 0) | (a > 2500)] = np.nan       # 明显被误码破坏的点
    return a


# ============================================================== 7. 数据集
class SVISSRFile:
    """S-VISSR 2.0 码流读取器。"""

    def __init__(self, path: str, layout: Optional[Dict[str, object]] = None,
                 check_crc: bool = False):
        self.path = path
        self.size = os.path.getsize(path)
        self.layout = layout or detect_layout(path)
        self.stride = int(self.layout["stride_bytes"])
        self.skip = int(self.layout["skip_bytes"])
        self.transform = str(self.layout["transform"])
        self.n_frames = (self.size - self.skip) // self.stride
        self.check_crc = check_crc
        self._mm = None

    def __enter__(self):
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self

    def __exit__(self, *exc):
        self._mm = None

    def frame_bytes(self, i: int) -> bytes:
        a = self.skip + i * self.stride
        return self._mm[a:a + self.stride].tobytes()

    def frame(self, i: int, with_images: bool = True) -> Frame:
        buf = self.frame_bytes(i)
        if self.transform != "none":
            buf = apply_transform(buf, self.transform)
        buf = buf[:FRAME_BYTES]
        return parse_frame(buf, i, with_images, self.check_crc)

    def __iter__(self):
        for i in range(self.n_frames):
            yield self.frame(i)


# ============================================================== 8. 图像组装
@dataclass
class Product:
    meta: Dict[str, object]
    ir: Optional[np.ndarray] = None        # (4, rows, 2291) uint16
    vis: Optional[np.ndarray] = None       # (4, rows, 9164) uint8  (逐传感器)
    line_rows: Optional[np.ndarray] = None  # 每条记录对应的绝对行号
    grid: Optional[np.ndarray] = None      # (25,25,2) 简化网格 (行号, 列号)
    frames_used: int = 0
    duplicates: int = 0
    missing: int = 0
    pass_index: int = 1


def _majority_const(items: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """常数块逐帧受误码影响，取所有帧中出现次数最多的值（众数）。"""
    from collections import Counter
    out: Dict[str, Optional[float]] = {}
    for _pos, _n, _typ, name in CONST_FIELDS:
        c: Counter = Counter()
        for d in items:
            v = d.get(name)
            if v is not None:
                c[v] += 1
        out[name] = c.most_common(1)[0][0] if c else None
    return out


def parse_file_segment(frame_bytes: bytes) -> Tuple[Dict[str, object],
                                                    Dict[str, Optional[float]],
                                                    Dict[str, bytes], int, int]:
    """
    只解析文件段（DOC）。DOC 段长 20408 bit = 2551 字节、8 对齐，因此可以直接按字节
    切片，不需要任何位展开 —— 扫描一遍大文件时这是最省时省内存的路径。
    """
    blk = frame_bytes[2:2293]                      # 跳过 2 字节段识别码
    doc = _decode_doc_block(blk[DOC_SAT[0]:DOC_SAT[1]])
    const = _decode_const_block(blk[DOC_CONST[0]:DOC_CONST[1]])
    groups = {
        "grid": blk[DOC_GRID[0]:DOC_GRID[1]],
        "orbit": blk[DOC_ORBIT[0]:DOC_ORBIT[1]],
        "manam": blk[DOC_MANAM[0]:DOC_MANAM[1]],
        "cal1": blk[DOC_CAL1[0]:DOC_CAL1[1]],
        "cal2": blk[DOC_CAL2[0]:DOC_CAL2[1]],
    }
    return doc, const, groups, blk[191], blk[193]   # 组号 / 重复计数


def _doc_summary(frame_bytes: bytes) -> Tuple[int, ...]:
    """
    供扫描阶段使用的极简 DOC 摘要（纯下标运算，不建 dict）：
    scan_mode, scan_status, frame_flag, image_flag, start_line, end_line,
    line_count, visr, satellite_id, beta, spin, 时间戳(BCD->int)
    """
    blk = frame_bytes[2:2293]
    ts = bcd_to_int(blk[17:25])                    # YYYYMMDDhhmmsscc
    return (blk[0], blk[1], blk[2], blk[3],
            bcd_to_int(blk[4:6]) or 0,
            bcd_to_int(blk[6:8]) or 0,
            bcd_to_int(blk[8:10]) or 0,
            u12(blk[65:67]),
            blk[89],
            int.from_bytes(blk[69:72], "big"),
            int.from_bytes(blk[72:75], "big"),
            -1 if ts is None else ts)


def _fmt_time(ts: int) -> Optional[str]:
    """2026091810592273 -> '2026-09-18 10:59:22.73'。"""
    if ts is None or ts < 0:
        return None
    s = str(ts).zfill(16)
    try:
        return (f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"
                f".{s[14:16]}")
    except IndexError:
        return None


def _ts_valid(ts: int) -> bool:
    """BCD 时间戳是否落在合理范围（个别帧的年份会被误码打成 4026 之类）。"""
    if ts is None or ts < 0:
        return False
    y, mo, d = ts // 10 ** 12, (ts // 10 ** 10) % 100, (ts // 10 ** 8) % 100
    return 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31


def scan_docs(sf: "SVISSRFile", n: int) -> Dict[str, np.ndarray]:
    """扫描整个码流的文件段，返回按帧索引的紧凑 ndarray（内存 ~n×50 字节）。"""
    keys = ("scan_mode", "scan_status", "frame_flag", "image_flag", "start_line",
            "end_line", "line_count", "visr", "sat_id", "beta", "spin")
    cols = {k: np.zeros(n, np.int32) for k in keys}
    ts = np.full(n, -1, np.int64)
    for i in range(n):
        s = _doc_summary(sf.frame_bytes(i)[:2293])
        for k, v in zip(keys, s[:11]):
            cols[k][i] = v
        ts[i] = s[11]
    cols["time"] = ts
    return cols


def segment_passes(docs: Dict[str, np.ndarray], n: int,
                   gap_frames: int = 200) -> List[Dict[str, object]]:
    """
    把码流切成若干"幅"（每次全圆盘观测一幅；SatDump 里叫一次 scan）。

    做法分两步，对误码都鲁棒：
      1) 只拿"强帧"（图像标志=FF 且行计数有效）划界：两次强帧相隔超过 `gap_frames`
         帧、或行计数连续 2 次回落到 ≤10（计数从 1 重新开始）就切新的一幅；
      2) 把夹在某幅首末强帧之间、图像标志=FF 但行计数被误码破坏的"弱帧"补回该幅，
         避免因为这些帧丢行；回扫帧归属于它前面那一幅。
    """
    strong = [i for i in range(n)
              if int(docs["image_flag"][i]) == 0xFF
              and 1 <= int(docs["line_count"][i]) <= 2500]
    groups: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    last_cnt = 0
    prev_i: Optional[int] = None
    small_run = 0
    for i in strong:
        cnt = int(docs["line_count"][i])
        gap = (i - prev_i) if prev_i is not None else 0
        small_run = small_run + 1 if cnt <= 10 else 0
        restart = (cur is None) or (gap > gap_frames) or (small_run >= 2 and last_cnt > 20)
        if restart:
            if cur is not None:
                groups.append(cur)
            cur = {"frames": [], "rollback": []}
            last_cnt = 0
            small_run = 1 if cnt <= 10 else 0
        cur["frames"].append(i)
        last_cnt = cnt
        prev_i = i
    if cur is not None:
        groups.append(cur)
    if not groups:
        return []

    # 第 2 步：把一首一尾之间的所有帧都归入该幅（卫星在一幅中间不会真的空闲，
    # 中间出现的图像标志/行计数异常都是误码），这样不会因为个别误码白丢一行数据
    for g in groups:
        lo, hi = g["frames"][0], g["frames"][-1]
        have = set(g["frames"])
        g["frames"] = [i for i in range(lo, hi + 1)]

    # 回扫帧：归到它前面最近那一幅
    firsts = [g["frames"][0] for g in groups]
    for i in range(n):
        if not is_backward_scan(int(docs["scan_status"][i])):
            continue
        k = 0
        for j, f in enumerate(firsts):
            if i >= f:
                k = j
        groups[k]["rollback"].append(i)

    return [g for g in groups if g["frames"]]


def _pass_start_row(docs: Dict[str, np.ndarray], frames: List[int]) -> Tuple[int, int]:
    """一幅图的起止行号：取该幅内出现次数最多的值（个别帧受误码影响会给出离谱值）。"""
    from collections import Counter
    c0 = Counter(int(docs["start_line"][i]) for i in frames
                 if 1 <= int(docs["start_line"][i]) <= 2500)
    c1 = Counter(int(docs["end_line"][i]) for i in frames
                 if 1 <= int(docs["end_line"][i]) <= 2500)
    start = c0.most_common(1)[0][0] if c0 else 1
    end = c1.most_common(1)[0][0] if c1 else start + 2290
    if end <= start:
        end = start + 2290
    return start, end


def decode(path: str, outdir: str, *, max_frames: Optional[int] = None,
           check_crc: bool = False, quiet: bool = False,
           save_npz: bool = False, render: bool = True,
           despike: bool = True, fill_missing: bool = True,
           graticule: bool = True, lut_path: Optional[str] = None,
           falsecolor: bool = True, rollback: bool = True,
           fill_from_rollback: bool = False, rollback_tolerance: int = 2,
           max_passes: Optional[int] = None, only_pass: Optional[int] = None):
    """
    解码入口。一个文件里可能有多幅全圆盘图（多次观测），此时按幅分段、逐幅流式输出，
    每幅一个子目录；只有一幅时保持原来的扁平输出结构。

    返回 Product（单幅）或 List[Product]（多幅）。
    """
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    with SVISSRFile(path, check_crc=check_crc) as sf:
        lay = sf.layout
        log(f"[i] 码流识别: 帧长 {sf.stride} 字节 / 跳过 {sf.skip} 字节 / 变换 {sf.transform}"
            f" / ID 命中 {lay['score']}/12")
        n = sf.n_frames if max_frames is None else min(max_frames, sf.n_frames)
        log(f"[i] 文件 {sf.size/1e6:.1f} MB -> {n} 帧（每条扫描线一帧）")

        # --- 轻量扫描：只解文件段（按字节切片，不展开位流）
        docs = scan_docs(sf, n)
        passes = segment_passes(docs, n)
        log(f"[i] 扫描完成：识别出 {len(passes)} 幅图像")
        for k, ps in enumerate(passes):
            s, e = _pass_start_row(docs, ps["frames"])
            t0 = _fmt_time(int(docs["time"][ps["frames"][0]]))
            log(f"      第 {k+1} 幅: {len(ps['frames'])} 帧（行 {s}~{e}），"
                f"回扫帧 {len(ps['rollback'])}，起报 {t0}")

        if only_pass:
            sel = [k for k in range(len(passes)) if k + 1 == only_pass]
        elif max_passes:
            sel = list(range(min(max_passes, len(passes))))
        else:
            sel = list(range(len(passes)))

        # 只解析一次 CLUT
        lut = None
        if falsecolor:
            if lut_path is None:
                for cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "lut.png"),
                             "lut.png"):
                    if os.path.exists(cand):
                        lut_path = cand
                        break
            if lut_path and os.path.exists(lut_path):
                lut = load_clut(lut_path)
            if lut is None:
                log("[-] 未找到 lut.png，跳过假彩色（可用 --lut 指定）")

        prods: List[Product] = []
        for k in sel:
            ps = passes[k]
            if len(sel) == 1 and len(passes) == 1:
                sub = outdir
            else:
                t0 = _fmt_time(int(docs["time"][ps["frames"][0]])) or ""
                stamp = t0.replace("-", "").replace(":", "").replace(" ", "_")[:15]
                sub = os.path.join(outdir, f"pass{k+1:02d}_{stamp}")
            log(f"[i] === 第 {k+1} 幅 -> {sub}")
            prods.append(_decode_pass(
                sf, docs, ps, sub, path=path, log=log, k=k, total=len(passes),
                check_crc=check_crc, save_npz=save_npz, render=render,
                despike=despike, fill_missing=fill_missing, graticule=graticule,
                lut=lut, lut_path=lut_path, falsecolor=falsecolor, rollback=rollback,
                fill_from_rollback=fill_from_rollback,
                rollback_tolerance=rollback_tolerance,
                keep_arrays=(len(sel) == 1)))
            gc.collect()

        return prods[0] if len(prods) == 1 else prods


def _decode_pass(sf: "SVISSRFile", docs: Dict[str, np.ndarray],
                 ps: Dict[str, object], outdir: str, *, path: str, log=print,
                 k: int = 0, total: int = 1, check_crc: bool = False,
                 save_npz: bool = False, render: bool = True, despike: bool = True,
                 fill_missing: bool = True, graticule: bool = True,
                 lut: Optional[np.ndarray] = None, lut_path: Optional[str] = None,
                 falsecolor: bool = True,
                 rollback: bool = True, fill_from_rollback: bool = False,
                 rollback_tolerance: int = 2,
                 keep_arrays: bool = True) -> Product:
    """解码一幅（一次全圆盘观测）：拼图 -> 补齐 -> 定标 -> 出图，然后释放内存。"""
    frames: List[int] = list(ps["frames"])
    rb_frames: List[int] = list(ps["rollback"])
    start_row, end_row = _pass_start_row(docs, frames)
    n_rows = end_row - start_row + 1
    log(f"[i] 图像行范围 {start_row}~{end_row}（{n_rows} 行）")

    # --- 行号映射：优先用"图像有效行计数"，误码时按顺序补齐
    rows: Dict[int, Optional[int]] = {}
    last_row: Optional[int] = None
    for i in frames:
        cand = None
        cnt = int(docs["line_count"][i])
        if int(docs["image_flag"][i]) == 0xFF and 1 <= cnt <= 2500:
            cand = start_row + cnt - 1
            if not (start_row <= cand <= end_row):
                cand = None
            elif last_row is not None and (cand <= last_row or cand - last_row > 4):
                cand = last_row + 1 if last_row < end_row else None
        if cand is None and last_row is not None and last_row < end_row:
            cand = last_row + 1
        if cand is None or cand > end_row:
            cand = None
        else:
            last_row = cand
        rows[i] = cand

    ir = np.zeros((IR_CHANNELS, n_rows, IR_COLS), np.uint16)
    vis = np.zeros((VIS_CHANNELS, n_rows, VIS_COLS), np.uint8)
    filled = np.zeros(n_rows, bool)

    # 子交换块：同一组在连续 8 条扫描线里重复发送 8 次，按比特多数表决纠错
    sub_copies: Dict[str, Dict[int, List[bytes]]] = {
        "cal1": {}, "cal2": {}, "grid": {}, "orbit": {}, "manam": {}}
    consts: List[Dict[str, Optional[float]]] = []
    used = 0
    dups = 0
    rb_entries: List[Dict[str, object]] = []

    total_jobs = len(frames) + len(rb_frames)
    done = 0
    for i in frames:
        fr = sf.frame(i, with_images=True)
        done += 1
        consts.append(fr.const)
        for key in ("cal1", "cal2", "grid", "orbit", "manam"):
            g = fr.sub_group
            blk = fr.groups.get(key)
            if blk and 0 <= g < 25:
                lst = sub_copies[key].setdefault(g, [])
                if len(lst) < 8:
                    lst.append(blk)
        r = rows.get(i)
        if r is None:
            continue
        idx = r - start_row
        if not (0 <= idx < n_rows) or filled[idx]:
            dups += 1
            continue
        ir[:, idx, :] = fr.ir
        vis[:, idx, :] = fr.vis
        filled[idx] = True
        used += 1
        if done % 400 == 0:
            log(f"    ... {done}/{total_jobs} 帧  行 {r}")

    # --- 回扫帧（自南向北快速步进）：单独收集，可用来补漏收的行
    for i in rb_frames:
        fr = sf.frame(i, with_images=True)
        done += 1
        rb_entries.append({
            "frame": i, "scan_status": fr.doc.get("scan_status"),
            "frame_flag": fr.doc.get("frame_flag"),
            "image_flag": fr.doc.get("image_flag"),
            "visr": int(docs["visr"][i]) if 1 <= int(docs["visr"][i]) <= 2500 else 0,
            "time": _fmt_time(int(docs["time"][i])),
            "ir": fr.ir, "vis": fr.vis,
        })
        if done % 400 == 0:
            log(f"    ... {done}/{total_jobs} 帧（回扫）")

    # 分组表决 -> 每个子交换组一份最优数据
    sub_best: Dict[str, Dict[int, bytes]] = {
        kk: {g: _vote_bytes(v) for g, v in d.items()} for kk, d in sub_copies.items()}
    del sub_copies

    cal = Calibration.from_groups(sub_best, "cal2")
    if not cal.ir:
        cal = Calibration.from_groups(sub_best, "cal1")
    n_missing = int((~filled).sum())
    log(f"[i] 有效行 {used}，重复 {dups}，缺行 {n_missing}；"
        f"定标表 {'已获取(%s)' % cal.source if cal.ir else '未获取'}")

    # 丢行补空：优先用回扫数据补（--fill-from-rollback），其余用上下行内插
    gap_rows: List[Dict[str, object]] = []
    n_rb_fill = 0
    if n_missing and fill_missing:
        gap_idx = [int(r) for r in np.nonzero(~filled)[0]]

        # 回扫行号 -> 帧（回扫期间 image_line_count 不可用，只能靠 VISSR 仪器行号）
        rb_map: Dict[int, Dict[str, object]] = {}
        if fill_from_rollback:
            for e in rb_entries:
                v = int(e["visr"]) if e["visr"] else 0
                if 1 <= v <= 2500 and v not in rb_map and e["ir"] is not None:
                    rb_map[v] = e

        def _use(r: int, e: Dict[str, object], src: str, offset: int) -> None:
            ir[:, r, :] = e["ir"]
            vis[:, r, :] = e["vis"]
            filled[r] = True
            gap_rows.append({"row": r + start_row, "source": src,
                             "src_line": int(e["visr"]), "offset": offset})

        # 1) 行号完全相同：直接取回扫那一行
        remaining: List[int] = []
        for r in gap_idx:
            absrow = r + start_row
            e = rb_map.get(absrow) if rb_map else None
            if e is not None:
                _use(r, e, "rollback", 0)
            else:
                remaining.append(r)

        # 2) 行号相近：允许用 ±tolerance 内的回扫行顶替（每条回扫行只用一次，
        #    避免同一行被复制给多个缺行）
        if rb_map and rollback_tolerance > 0 and remaining:
            free = sorted(set(rb_map) - {int(g["src_line"]) for g in gap_rows})
            still: List[int] = []
            for r in remaining:
                absrow = r + start_row
                best, bdist = None, None
                for v in free:
                    d = abs(v - absrow)
                    if d <= rollback_tolerance and (bdist is None or d < bdist):
                        best, bdist = v, d
                if best is None:
                    still.append(r)
                else:
                    _use(r, rb_map[best], "rollback_nearest", best - absrow)
                    free.remove(best)
            remaining = still

        # 3) 其余：用上下最近的有效行内插（此时已被回扫补过的行也算有效）
        valid_idx = np.nonzero(filled)[0]
        for r in remaining:
            above = valid_idx[valid_idx < r]
            below = valid_idx[valid_idx > r]
            a = int(above[-1]) if above.size else None
            b = int(below[0]) if below.size else None
            gap_rows.append({"row": r + start_row, "source": "interpolate",
                             "src_line": None, "offset": None})
            for cube in (ir, vis):
                if a is not None and b is not None:
                    cube[:, r, :] = ((cube[:, a, :].astype(np.uint32) +
                                      cube[:, b, :]) // 2).astype(cube.dtype)
                elif a is not None:
                    cube[:, r, :] = cube[:, a, :]
                elif b is not None:
                    cube[:, r, :] = cube[:, b, :]
            filled[r] = True

        n_rb_fill = sum(1 for g in gap_rows if str(g["source"]).startswith("rollback"))
        log(f"[i] 丢行 {n_missing} 行：回扫补 {n_rb_fill} 行、上下行内插 {n_missing - n_rb_fill} 行")
        for g in gap_rows:
            extra = "" if not g["offset"] else f"（Δ{g['offset']:+d} 行顶替）"
            log(f"      行 {g['row']}: {g['source']}{extra}")
    elif n_missing:
        log(f"[i] 丢行 {n_missing} 行，未补（--no-fill）")

    sat_id = next((int(docs["sat_id"][i]) for i in frames
                   if int(docs["sat_id"][i]) not in (0,)), None)
    sat = SAT_NAMES.get(sat_id, f"FY-2?(0x{sat_id:02X})" if sat_id else "FY-2?")
    times = [int(docs["time"][i]) for i in frames if _ts_valid(int(docs["time"][i]))]
    t0 = _fmt_time(times[0]) if times else None
    t1 = _fmt_time(times[-1]) if times else None
    const = _majority_const(consts)
    del consts

    meta = {
        "source": os.path.abspath(path),
        "file_size": sf.size,
        "layout": sf.layout,
        "output_dir": os.path.abspath(outdir),
        "pass": {"index": k + 1, "of": total},
        "frames_total_in_file": sf.n_frames,
        "frames_in_pass": len(frames),
        "frames_used": used,
        "duplicate_frames": dups,
        "missing_rows": n_missing,
        "rollback_filled": n_rb_fill,
        "satellite": sat,
        "satellite_id": sat_id,
        "image_start_line": start_row,
        "image_end_line": end_row,
        "image_rows": n_rows,
        "ir_columns": IR_COLS,
        "vis_columns": VIS_COLS,
        "time_start": t0,
        "time_end": t1,
        "constants": const,
        "scan_mode": SCAN_MODES.get(int(docs["scan_mode"][frames[0]]), None),
        "calibration": {
            "source": cal.source,
            "info": {kk: (int(v) if isinstance(v, (int, np.integer)) else v)
                     for kk, v in cal.info.items()},
            "ir_units": "K",
            "vis_units": "albedo",
            "ir_lut0": [float(x) for x in cal.ir[0][:4]] if 0 in cal.ir else None,
            "vis_lut0": [float(x) for x in cal.vis[0][:4]] if 0 in cal.vis else None,
        },
        "line_times": [
            {"frame": i, "row": rows.get(i),
             "time": _fmt_time(int(docs["time"][i])),
             "visr_line": int(docs["visr"][i]),
             "beta": int(docs["beta"][i]),
             "spin": int(docs["spin"][i]),
             "image_flag": int(docs["image_flag"][i])}
            for i in frames
        ],
        "gap_rows": gap_rows,
        "rollback_filled_rows": n_rb_fill,
        "rollback": {"frames": len(rb_entries),
                     "with_line_number": sum(1 for e in rb_entries if e["visr"])},
        "subblocks": {kk: sorted(v.keys()) for kk, v in sub_best.items()},
    }
    if lut is not None:
        meta["false_color"] = {
            "clut": os.path.abspath(lut_path or ""),
            "axes": "x = 可见光亮度 (255-反照率), y = IR1 亮温",
            "credit": "LUT.png by @eswnl / https://github.com/eswnl/FY-2x_CLUT",
        }

    grid = parse_grid(b"".join(sub_best["grid"][g] for g in range(25)
                               if g in sub_best["grid"]))
    prod = Product(meta=meta, ir=ir, vis=vis, frames_used=used,
                   duplicates=dups, missing=n_missing, grid=grid,
                   pass_index=k + 1)

    os.makedirs(outdir, exist_ok=True)
    _write_metadata(prod, cal, outdir, grid)
    if rollback and rb_entries:
        _write_rollback(rb_entries, outdir, cal, start_row, log,
                        despike=despike, save_npz=save_npz)
    del rb_entries
    if render:
        _render(prod, cal, outdir, log, despike=despike, graticule=graticule,
                clut=lut, falsecolor=falsecolor)
    if save_npz:
        np.savez_compressed(os.path.join(outdir, "fy2_svissr.npz"),
                            ir=ir, vis=vis, rows=np.arange(start_row, end_row + 1))
        log("[+] 原始计数值已存 fy2_svissr.npz")
    if not keep_arrays:          # 多幅时不必把每幅的整副图像都留在内存里
        prod.ir = prod.vis = None
        del ir, vis
    return prod




# ============================================================== 9. 输出
def _percentile_range(a: np.ndarray, lo: float, hi: float,
                      mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """抽样估计百分位（大图不必逐像素统计）。"""
    vals = a[mask] if (mask is not None and mask.any()) else a
    if vals.size > 400_000:
        vals = vals[::max(1, vals.size // 400_000)]
    p_lo, p_hi = np.percentile(np.asarray(vals, np.float32), [lo, hi])
    return float(p_lo), float(p_hi) if p_hi > p_lo else float(p_lo + 1.0)


def _apply_stretch(a: np.ndarray, p_lo: float, p_hi: float,
                   out: Optional[np.ndarray] = None,
                   gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    """用固定黑白点做分块拉伸（不整图转 float32）。"""
    h, w = a.shape[:2]
    if out is None:
        out = np.empty(a.shape, np.uint8)
    rows = max(32, 2_000_000 // max(1, w))
    span = max(p_hi - p_lo, 1e-6)
    for y0 in range(0, h, rows):
        blk = a[y0:y0 + rows].astype(np.float32)
        y = np.clip((blk - p_lo) / span, 0, 1)
        if invert:
            y = 1.0 - y
        if gamma != 1.0:
            y = y ** gamma
        out[y0:y0 + rows] = (y * 255 + 0.5).astype(np.uint8)
    return out


def _stretch(a: np.ndarray, lo: float = 0.5, hi: float = 99.5,
             invert: bool = False, gamma: float = 1.0,
             mask: Optional[np.ndarray] = None,
             out: Optional[np.ndarray] = None) -> np.ndarray:
    """百分位线性拉伸（分块处理，避免大图一次性转 float32 占几百 MB）。"""
    p_lo, p_hi = _percentile_range(a, lo, hi, mask)
    return _apply_stretch(a, p_lo, p_hi, out=out, gamma=gamma, invert=invert)


def despeckle(a: np.ndarray, threshold: float, chunk: Optional[int] = None) -> np.ndarray:
    """冲激噪声抑制：与 4 邻域中位数相差超过 threshold 的像元用中位数替换。

    保持输入 dtype、按行分块处理（不会整图 float32 拷贝），大图也只需 ~1 份输出。
    该码流为业余接收，存在 ~1e-3 量级的符号误码（DPSK 差分译码表现为相邻双比特
    错误），显示前滤波可显著改善观感；用 --no-despike 可关闭。
    """
    if threshold <= 0:
        return a
    h, w = a.shape[:2]
    if chunk is None:
        chunk = max(32, 2_000_000 // max(1, w))
    out = a.copy()
    dt = a.dtype
    is_int = np.issubdtype(dt, np.integer)
    dmax = float(np.iinfo(dt).max) if is_int else 0.0
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        n = y1 - y0
        idx = np.arange(y0, y1)
        cur = a[y0:y1].astype(np.float32)
        up = a[np.maximum(idx - 1, 0)].astype(np.float32)
        dn = a[np.minimum(idx + 1, h - 1)].astype(np.float32)
        lf = cur.copy()
        lf[:, 1:] = cur[:, :-1]
        rt = cur.copy()
        rt[:, :-1] = cur[:, 1:]
        med = np.median(np.stack([up, dn, lf, rt]), axis=0)
        m = np.abs(cur - med) > threshold
        if m.any():
            cur[m] = med[m]
            out[y0:y1] = (np.clip(np.rint(cur), 0, dmax).astype(dt) if is_int
                          else cur.astype(dt))
    return out


def _space_mask(counts: np.ndarray, sat_level: int) -> np.ndarray:
    """估计"地球圆盘"掩膜：近饱和（太空/冷背景）像元判为非地球。"""
    return counts < (sat_level - 12)


# ---------------------------------------------------- 假彩色（CLUT 查表，参照 SatDump）
# LUT.png 由 @eswnl 制作，取自 https://github.com/eswnl/FY-2x_CLUT （经作者许可转为 PNG）。
# 做法与 SatDump SVISSRImageDecoderModule::writeImages 相同：以"可见光亮度"为纵轴
# （255 - 反照率，越亮越靠上）、"红外温度"为横轴（越暖越靠右）查 256x256 彩色表。
CLUT_SIZE = 256


def load_clut(path: str) -> Optional[np.ndarray]:
    """载入 CLUT 并重采样成 256x256x3 uint8（与 SatDump 的 lutImage.resize(256,256) 一致）。"""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if im.size != (CLUT_SIZE, CLUT_SIZE):
            im = im.resize((CLUT_SIZE, CLUT_SIZE), Image.BILINEAR)
        return np.asarray(im)
    except Exception:
        return None


def false_color(ir_counts: np.ndarray, vis: np.ndarray, cal: "Calibration",
                clut: np.ndarray, *, ch_ir: int = 0,
                white_point: Optional[float] = None,
                gamma: float = 0.9, tint_max: float = 205.0,
                despike: bool = True) -> np.ndarray:
    """
    用 CLUT 生成假彩色。

    纵轴 x（LUT 行）：`x = 255 - (反照率/白点)^gamma * 255` —— 越亮越靠上（LUT 顶部为白）。
    横轴 y（LUT 列）：`y = 亮温百分比 * tint_max` —— 越暖越靠右（LUT 右侧偏粉/棕）。

    内存：可见光用整数累加降采样（不整图 float32），查表按行分块，峰值约
    "vis 一份 + 几个 h×w 缓冲"。
    """
    h, w = ir_counts.shape

    # 可见光 9164x2291 → 红外分辨率 2291x2291：4 个传感器 + 横向 4:1 一起整数平均
    if vis.shape[2] == w * 4:
        acc = vis.reshape(vis.shape[0], h, w, 4).sum(axis=(0, 3), dtype=np.uint32)
        vis_ds = acc.astype(np.float32) / (vis.shape[0] * 4)
        del acc
    else:
        vis_ds = vis.mean(axis=0, dtype=np.float32)

    counts_ir = despeckle(ir_counts, 60) if despike else ir_counts
    bt = (cal.ir_bt(ch_ir, counts_ir) if cal.ir else counts_ir.astype(np.float32))
    alb = (cal.vis_albedo(0, np.clip(vis_ds, 0, 63)) if cal.vis else vis_ds)
    del vis_ds
    if not despike:
        counts_ir = ir_counts

    disk = _space_mask(ir_counts, 1023)
    stat_bt = bt[disk] if disk.any() else bt.ravel()
    stat_alb = alb[disk] if disk.any() else alb.ravel()
    if stat_bt.size == 0:
        return np.zeros((h, w, 3), np.uint8)
    if stat_bt.size > 400_000:
        step = max(1, stat_bt.size // 400_000)
        stat_bt, stat_alb = stat_bt[::step], stat_alb[::step]

    b_lo, b_hi = np.percentile(stat_bt, [2, 98])
    if b_hi <= b_lo:
        b_hi = b_lo + 1.0
    if white_point is None:      # 白点：圆盘内 99.5% 反照率，但不低于 0.55
        white_point = float(max(0.55, np.percentile(stat_alb, 99.5) / 100.0))
    wp = max(white_point * 100.0, 1e-6)

    # 分块查表，避免再开 x/y 两张整图 float32
    out = np.empty((h, w, 3), np.uint8)
    rows = max(32, 1_000_000 // max(1, w))
    bt_span = b_hi - b_lo
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        x = 255.0 - np.clip(alb[y0:y1] / wp, 0, 1) ** gamma * 255.0
        y = np.clip((bt[y0:y1] - b_lo) / bt_span, 0, 1) * tint_max
        out[y0:y1] = clut[np.clip(x, 0, 255).astype(np.uint8),
                          np.clip(y, 0, 255).astype(np.uint8)]
    return out


def _save_png(arr: np.ndarray, path: str, compress_level: Optional[int] = None) -> None:
    """存 PNG。大图用较低压缩级别：optimize=True/高级别对这种 2000 万像素的图非常慢。"""
    from PIL import Image
    if compress_level is None:
        compress_level = 3 if arr.size >= 4_000_000 else 6
    Image.fromarray(arr).save(path, compress_level=compress_level)


def _colorize(gray: np.ndarray, chunk: int = 512) -> np.ndarray:
    """IR 伪彩色：冷（高云顶）为白/蓝，暖（地表洋面）为暗棕（常见 IR 增强配色）。

    分块查色，避免 (h,w,3) float32 的整图中间量。
    """
    stops = np.array([
        [0.00, 46, 37, 41],
        [0.25, 98, 78, 70],
        [0.44, 152, 124, 108],
        [0.60, 198, 183, 170],
        [0.76, 140, 186, 218],
        [0.89, 222, 239, 248],
        [1.00, 255, 255, 255],
    ], np.float32)
    h, w = gray.shape[:2]
    out = np.empty((h, w, 3), np.uint8)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        x = gray[y0:y1].astype(np.float32) / 255.0
        for k in range(3):
            out[y0:y1, :, k] = np.interp(x, stops[:, 0], stops[:, 1 + k]).astype(np.uint8)
    return out


def _draw_graticule(rgb: np.ndarray, grid: Optional[np.ndarray], start_row: int,
                    step: int = 5, min_lum: float = 45.0) -> np.ndarray:
    """
    用简化网格块（表 4.11）在图像上画 5°×5° 经纬网。
    线色随背景亮度自适应：亮背景用深线、暗背景用浅线，太暗（太空/夜面）就不画。
    """
    if grid is None or not np.isfinite(grid).any():
        return rgb
    from PIL import Image, ImageDraw
    lum = rgb.astype(np.float32).mean(axis=2)
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img, "RGBA")
    h, w = rgb.shape[:2]

    def px(node):
        row, col = node
        if not (np.isfinite(row) and np.isfinite(col)):
            return None
        x = int(round(col - 1))
        y = int(round(row - start_row))
        return (x, y) if (0 <= x < w and 0 <= y < h) else None

    def draw_line(pts):
        for p1, p2 in zip(pts, pts[1:]):
            n = max(2, int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))) // 2 + 2)
            xs = np.linspace(p1[0], p2[0], n).astype(np.int32)
            ys = np.linspace(p1[1], p2[1], n).astype(np.int32)
            lv = float(lum[ys, xs].mean())
            if lv < min_lum:
                continue
            col = (12, 12, 24, 70) if lv > 145 else (255, 255, 235, 78)
            dr.line([p1, p2], fill=col, width=1)

    for i in range(grid.shape[0]):          # 纬线
        draw_line([p for p in (px(grid[i, j]) for j in range(grid.shape[1])) if p])
    for j in range(grid.shape[1]):          # 经线
        draw_line([p for p in (px(grid[i, j]) for i in range(grid.shape[0])) if p])
    try:                                     # 星下点标记
        c = px(grid[12, 12])
        if c and lum[c[1], c[0]] > min_lum:
            dr.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], outline=(255, 70, 70, 230))
    except IndexError:
        pass
    return np.asarray(img)


def _write_rollback(entries: List[Dict[str, object]], outdir: str, cal: Calibration,
                    start_row: int, log=print, despike: bool = True,
                    save_npz: bool = False) -> None:
    """
    单独导出回扫（自南向北快速步进，scan_status=0xCC）部分的数据。

    回扫期间扫描镜按 10 步/周快速回程，每条扫描线与正扫同号行内容一致但采样稀疏
    （行间约 10 行）、采集时间更晚，且星上帧标志为"无效"，因此默认解码器（以及
    SatDump）会把它丢掉。这里单独存一份，供事后做时间差/配准等分析。
    """
    d = os.path.join(outdir, "rollback")
    os.makedirs(d, exist_ok=True)

    with_data = [e for e in entries if e["visr"] and e["ir"] is not None]
    ordered = sorted(with_data, key=lambda e: int(e["visr"]))
    log(f"[i] 回扫帧 {len(entries)}（含有效行号 {len(ordered)}） -> {d}")

    # --- 清单
    with open(os.path.join(d, "rollback.csv"), "w", encoding="utf-8") as f:
        f.write("frame,time,visr_line,scan_status,scan_status_desc,frame_flag,"
                "image_flag,ir1_mean,ir1_min,ir1_max,vis1_max\n")
        for e in entries:
            ir1 = e["ir"][0] if e["ir"] is not None else None
            v1 = e["vis"][0] if e["vis"] is not None else None
            f.write(f'{e["frame"]},{e["time"] or ""},{e["visr"]},'
                    f'0x{int(e["scan_status"] or 0):02X},"{scan_status_desc(int(e["scan_status"] or 0))}",'
                    f'{e["frame_flag"]},{e["image_flag"]},'
                    f'{"" if ir1 is None else round(float(ir1.mean()), 1)},'
                    f'{"" if ir1 is None else int(ir1.min())},'
                    f'{"" if ir1 is None else int(ir1.max())},'
                    f'{"" if v1 is None else int(v1.max())}\n')

    if not ordered:
        log("[-] 回扫部分没有带行号的数据，仅输出清单")
        return

    ir_stack = np.stack([e["ir"] for e in ordered])     # (N, 4, 2291)
    vis_stack = np.concatenate([e["vis"] for e in ordered], axis=0)  # (4N, 9164)
    visr = np.array([int(e["visr"]) for e in ordered], np.int32)

    # --- 按行号稠密堆叠（北→南），每帧一行；红外与主产品一样用亮温拉伸
    for c, name in CH_NAMES.items():
        counts = ir_stack[:, c]
        if despike:
            counts = despeckle(counts, 60)
        data = cal.ir_bt(c, counts) if cal.ir else counts
        img = _stretch(data, 2, 98, invert=True, mask=_space_mask(ir_stack[:, c], 1023))
        _save_png(img, os.path.join(d, f"fy2_rollback_{name.lower()}.png"))
    log(f"[+] rollback/fy2_rollback_ir1..4.png  ({len(ordered)}x2291)")

    vb = despeckle(vis_stack, 6) if despike else vis_stack
    _save_png(_stretch(vb, 0.5, 99.5, gamma=0.8),
              os.path.join(d, "fy2_rollback_vis.png"))
    log(f"[+] rollback/fy2_rollback_vis.png  ({vb.shape[0]}x{vb.shape[1]})")

    # --- 按真实行号摆放（体现 ~10 行/帧的稀疏采样几何）
    h = 2291
    canvas = np.zeros((h, IR_COLS), np.uint16)
    filled = np.zeros(h, bool)
    for e in ordered:
        r = int(e["visr"]) - start_row
        if 0 <= r < h:
            canvas[r] = e["ir"][0]
            filled[r] = True
    mask = filled[:, None] & (canvas < 1011)
    img = _stretch(canvas, 2, 98, invert=True, mask=mask)
    img[~filled] = 0                     # 未采样的行留黑，直观看出 10 行间隔
    _save_png(img, os.path.join(d, "fy2_rollback_positions.png"))
    log(f"[+] rollback/fy2_rollback_positions.png  ({int(filled.sum())} 行采样 / 2291 行，间隔可见)")

    if save_npz:
        np.savez_compressed(os.path.join(d, "rollback.npz"),
                            ir=ir_stack, vis=vis_stack, visr=visr)
        log("[+] rollback/rollback.npz")


def _render(prod: Product, cal: Calibration, outdir: str, log=print,
            despike: bool = True, graticule: bool = True,
            clut: Optional[np.ndarray] = None, falsecolor: bool = True) -> None:
    ir, vis = prod.ir, prod.vis
    rows = ir.shape[1]

    # ---- 红外：定标表 -> 亮温，冲激滤波 + 圆盘内百分位拉伸，冷（云顶）为白
    # 只缓存拉伸后的 8 bit 灰度图（5 MB/通道），不缓存 float32 亮温
    gray: Dict[int, np.ndarray] = {}
    for c, name in CH_NAMES.items():
        counts = despeckle(ir[c], 60) if despike else ir[c]
        if cal.ir:
            bt = cal.ir_bt(c, counts)
            bt = despeckle(bt, 15) if despike else bt
            img = _stretch(bt, 2, 98, invert=True, mask=_space_mask(ir[c], 1023))
            del bt
            unit = "BT(K)"
        else:
            img = _stretch(counts, 2, 98, invert=True, mask=_space_mask(ir[c], 1023))
            unit = "count"
        gray[c] = img
        _save_png(img, os.path.join(outdir, f"fy2_{name.lower()}.png"))
        col = _colorize(img)
        if graticule and c == 0:
            col = _draw_graticule(col, prod.grid, int(prod.meta["image_start_line"]))
        _save_png(col, os.path.join(outdir, f"fy2_{name.lower()}_color.png"))
        del col
        log(f"[+] fy2_{name.lower()}.png / _color.png  ({unit}, {img.shape[0]}x{img.shape[1]})")

    # ---- 可见光：4 个传感器各得一条扫描线，按行交织成 9164x(4N)
    # 黑白点取一次、逐传感器直接拉进目标缓冲，省掉一份中间整图
    vis_band = np.empty((rows * 4, VIS_COLS), np.uint8)
    p_lo, p_hi = _percentile_range(vis[:, ::4, ::4], 0.5, 99.5)
    for s in range(4):
        d = despeckle(vis[s], 6) if despike else vis[s]
        _apply_stretch(d, p_lo, p_hi, out=vis_band[s::4], gamma=0.8)
        del d
    _save_png(vis_band, os.path.join(outdir, "fy2_vis.png"))
    log(f"[+] fy2_vis.png  ({vis_band.shape[0]}x{vis_band.shape[1]})")

    # ---- 合成图 1：可见光 + IR1（sandwich）
    g_vis = _stretch(vis_band[::4, ::4][:rows, :IR_COLS], 0.5, 99.5, gamma=0.7)
    del vis_band
    g_ir1 = gray[0]
    rgb = np.dstack([np.maximum(g_vis, (g_ir1 * 0.9).astype(np.uint8)), g_ir1, g_ir1])
    del g_vis
    if graticule:
        rgb = _draw_graticule(rgb, prod.grid, int(prod.meta["image_start_line"]))
    _save_png(rgb, os.path.join(outdir, "fy2_rgb_sandwich.png"))
    del rgb
    log("[+] fy2_rgb_sandwich.png  (R=VIS, G=B=IR1)")

    # ---- 合成图 2：IR3/IR1/IR2 伪彩（云相态、水汽分析）
    rgb2 = np.dstack([gray[2], g_ir1, gray[1]])
    _save_png(rgb2, os.path.join(outdir, "fy2_rgb_ir.png"))
    del rgb2
    log("[+] fy2_rgb_ir.png  (R=IR3, G=IR1, B=IR2)")

    # ---- 假彩色（CLUT）：VIS × IR1 查表，参照 SatDump / @eswnl 的 FY-2x CLUT
    if falsecolor:
        if clut is None:
            log("[-] 未找到 lut.png，跳过假彩色（可用 --lut 指定）")
        else:
            fc = false_color(ir[0], vis, cal, clut, despike=despike)
            if graticule:
                fc = _draw_graticule(fc, prod.grid, int(prod.meta["image_start_line"]))
            _save_png(fc, os.path.join(outdir, "fy2_falsecolor.png"))
            del fc
            log("[+] fy2_falsecolor.png  (CLUT: x=可见光亮度, y=IR1 亮温)")


def _write_metadata(prod: Product, cal: Calibration, outdir: str,
                    grid: Optional[np.ndarray] = None) -> None:
    meta = dict(prod.meta)
    if cal.ir:
        meta["calibration"]["ir_lut"] = {CH_NAMES[k]: [float(x) for x in v]
                                         for k, v in cal.ir.items()}
    if cal.vis:
        meta["calibration"]["vis_lut"] = {f"VIS{k+1}": [float(x) for x in v]
                                          for k, v in cal.vis.items()}
    lines = meta.pop("line_times", [])
    gaps = meta.get("gap_rows") or []
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if gaps:
        with open(os.path.join(outdir, "gapfill.csv"), "w", encoding="utf-8") as f:
            f.write("row,source,src_line,offset\n")
            for g in gaps:
                f.write(f'{g["row"]},{g["source"]},'
                        f'{"" if g["src_line"] is None else g["src_line"]},'
                        f'{"" if g["offset"] is None else g["offset"]}\n')
    if grid is not None:
        with open(os.path.join(outdir, "grid.csv"), "w", encoding="utf-8") as f:
            f.write("lat,lon,row,col\n")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    row, col = grid[i, j]
                    f.write(f"{60 - 5 * i},{45 + 5 * j},"
                            f"{'' if not np.isfinite(row) else int(row)},"
                            f"{'' if not np.isfinite(col) else int(col)}\n")
    with open(os.path.join(outdir, "line_times.csv"), "w", encoding="utf-8") as f:
        f.write("frame,row,time,visr_line,beta_count,spin_period,image_flag\n")
        for r in lines:
            f.write(",".join("" if r[k] is None else str(r[k])
                             for k in ("frame", "row", "time", "visr_line",
                                       "beta", "spin", "image_flag")) + "\n")


# ============================================================== 10. CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="风云二号（02 批）S-VISSR 2.0 展宽数据解码器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python fy2_svissr.py fengyun_svissr.svissr -o out\n"
               "  python fy2_svissr.py raw.bin --info\n")
    ap.add_argument("input", help="输入文件（.svissr / 原始 S-VISSR 码流）")
    ap.add_argument("-o", "--outdir", default="out", help="输出目录（默认 out）")
    ap.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧")
    ap.add_argument("--info", action="store_true", help="只打印元数据，不生成图像")
    ap.add_argument("--check-crc", action="store_true", help="逐段校验 CRC-16")
    ap.add_argument("--npz", action="store_true", help="额外保存原始计数值 .npz")
    ap.add_argument("--no-despike", action="store_true",
                    help="关闭冲激噪声滤波（默认开启，抑制误码造成的孤立噪点）")
    ap.add_argument("--no-fill", action="store_true", help="丢行不内插补空")
    ap.add_argument("--no-graticule", action="store_true",
                    help="不在合成图上叠加简易经纬网")
    ap.add_argument("--lut", metavar="PATH", default=None,
                    help="CLUT 彩色表（默认用脚本同目录的 lut.png）")
    ap.add_argument("--no-falsecolor", action="store_true",
                    help="不生成 CLUT 假彩色图")
    ap.add_argument("--no-rollback", action="store_true",
                    help="不单独导出回扫（自南向北快速步进）部分数据")
    ap.add_argument("--fill-from-rollback", action="store_true",
                    help="用回扫数据补齐主图缺失的行（回扫是地面站漏收行的补充采样）")
    ap.add_argument("--rollback-tolerance", type=int, default=2, metavar="N",
                    help="补齐时允许用行号相差 ≤N 的回扫行顶替（默认 2；0=只认同号行）")
    ap.add_argument("--max-passes", type=int, default=None, metavar="N",
                    help="一个文件里有多幅图时，只解前 N 幅")
    ap.add_argument("--only-pass", type=int, default=None, metavar="N",
                    help="只解第 N 幅（1 基）")
    ap.add_argument("--list-passes", action="store_true", help="列出文件里所有幅，不解码")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.exists(a.input):
        print(f"输入文件不存在: {a.input}", file=sys.stderr)
        return 2

    if a.info or a.list_passes:
        with SVISSRFile(a.input, check_crc=a.check_crc) as sf:
            print(json.dumps(sf.layout, ensure_ascii=False, indent=2))
            print(f"帧数: {sf.n_frames}")
            docs = scan_docs(sf, sf.n_frames)
            passes = segment_passes(docs, sf.n_frames)
            print(f"识别出 {len(passes)} 幅图像:")
            for k, ps in enumerate(passes):
                s, e = _pass_start_row(docs, ps["frames"])
                print(f"  第 {k+1} 幅: 帧 {ps['frames'][0]}~{ps['frames'][-1]}，"
                      f"{len(ps['frames'])} 行（{s}~{e}），回扫帧 {len(ps['rollback'])}，"
                      f"起报 {_fmt_time(int(docs['time'][ps['frames'][0]]))}")
            if a.info:
                fr = sf.frame(0, with_images=False)
                print(json.dumps({k2: v for k2, v in fr.doc.items() if k2 != "raw"},
                                 ensure_ascii=False, indent=2, default=str))
                print(json.dumps(fr.const, ensure_ascii=False, indent=2))
        return 0

    res = decode(a.input, a.outdir, max_frames=a.max_frames,
                 check_crc=a.check_crc, quiet=a.quiet, save_npz=a.npz,
                 despike=not a.no_despike, fill_missing=not a.no_fill,
                 graticule=not a.no_graticule, lut_path=a.lut,
                 falsecolor=not a.no_falsecolor, rollback=not a.no_rollback,
                 fill_from_rollback=a.fill_from_rollback,
                 rollback_tolerance=a.rollback_tolerance,
                 max_passes=a.max_passes, only_pass=a.only_pass)
    prods = res if isinstance(res, list) else [res]
    print(f"\n===== 解码完成（{len(prods)} 幅） =====")
    for p in prods:
        m = p.meta
        print(f"\n[第 {m['pass']['index']}/{m['pass']['of']} 幅]")
        print(f"卫星        : {m['satellite']}")
        print(f"观测时间    : {m['time_start']} ~ {m['time_end']}")
        print(f"扫描行      : {m['image_start_line']}~{m['image_end_line']} "
              f"（{m['image_rows']} 行，缺 {m['missing_rows']}，"
              f"回扫补 {m.get('rollback_filled', 0)}）")
        print(f"星下点经度  : {(m['constants'] or {}).get('subsat_lon_mdeg')}")
        print(f"输出目录    : {m.get('output_dir', os.path.abspath(a.outdir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
