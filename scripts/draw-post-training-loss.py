"""Draw the original teaching figures for llm-post-training-loss.md with Pillow."""

import argparse
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "llm-post-training-loss"
PAPER = "#fbfaf5"
INK = "#233a31"
MUTED = "#4e655b"
LINE = "#d1dcd3"
GREEN = "#17654f"
ORANGE = "#9e4431"


@lru_cache
def font(path, size):
    return ImageFont.truetype(path, size)


def loss_map(make, save):
    picture, draw, label = make(592)
    label((32, 25), "样本怎么来", 30, GREEN)
    label((330, 25), "训练信号", 30, GREEN)
    label((565, 25), "怎么汇总", 30, GREEN)
    rows = [
        ("PPO", "旧策略采样", "奖励与价值估计", "逐 token 裁剪", "加价值与熵项", "#e7f0e8"),
        ("GRPO", "旧策略同题多采", "组内相对得分", "逐 token 裁剪", "先按回答平均", "#f2ecd9"),
        ("DPO", "固定偏好对", "优选与落选", "逐回答求和", "对偏好对平均", "#e6eef2"),
        ("OPD", "学生策略采样", "教师逐步给分布", "逐前缀算 KL", "对 token 平均", "#f3e8e5"),
    ]
    for index, (method, source, signal, first, second, fill) in enumerate(rows):
        top = 76 + 126 * index
        draw.rounded_rectangle((16, top, 784, top + 113), radius=18, fill=fill)
        label((32, top + 10), method, 37)
        label((32, top + 59), source, 31, MUTED)
        label((303, top + 30), "›", 38, MUTED, "mm")
        label((330, top + 38), signal, 29)
        label((548, top + 30), "›", 38, MUTED, "mm")
        label((565, top + 15), first, 30)
        label((565, top + 59), second, 28, MUTED)
    save(picture, "loss-map.png")


def clip_chart(make, save):
    picture, draw, label = make(492)
    label((32, 21), "取负前的单 token 目标", 32)
    label((767, 24), "ε = 0.2", 28, MUTED, "ra")
    for left, sign, color, title in [
        (82, 1, GREEN, "正优势 A = +1"),
        (472, -1, ORANGE, "负优势 A = −1"),
    ]:
        top, bottom, width = 137, 351, 290

        def x(ratio):
            return left + width * (ratio - 0.4)

        def y(value):
            return bottom - (bottom - top) * (value - low) / (high - low)

        low, high = (0.3, 1.5) if sign > 0 else (-1.5, -0.3)
        boundary = 1.2 if sign > 0 else 0.8
        label((left + width / 2, 96), title, 29, color, "mm")
        for value in (0.4, 0.8, 1.2):
            level = sign * value
            py = y(level)
            draw.line((left, py, left + width, py), fill=LINE, width=2)
            label((left - 9, py), f"{level:.1f}".replace("-", "−"), 21, MUTED, "rm")
        draw.line((x(boundary), top, x(boundary), bottom), fill=color, width=2)
        draw.line((left, top, left, bottom, left + width, bottom), fill=MUTED, width=2)
        for ratio in (0.4, 0.8, 1.2, 1.4):
            draw.line((x(ratio), bottom, x(ratio), bottom + 6), fill=MUTED, width=2)
            label((x(ratio), bottom + 25), f"{ratio:.1f}", 22, MUTED, "mm")
        points = []
        for step in range(101):
            ratio = 0.4 + step / 100
            clipped = min(1.2, max(0.8, ratio))
            points.append((x(ratio), y(min(sign * ratio, sign * clipped))))
        draw.line(points, fill=color, width=6)
        px, py = x(boundary), y(sign * boundary)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        label((left + width / 2, 425), "右侧变平" if sign > 0 else "左侧变平", 29, color, "mm")
    label((400, 469), "概率比 ρ = 当前概率 ÷ 旧概率", 27, MUTED, "mm")
    save(picture, "ppo-clip.png")


def opd_chart(make, save):
    picture, draw, label = make(375)
    label((33, 19), "同一前缀的下一 token 概率", 33)
    label((230, 93), "学生", 30, GREEN)
    label((507, 93), "教师", 30, ORANGE)
    for top, name, student, teacher in [(155, "token 1", 0.8, 0.5), (243, "token 2", 0.2, 0.5)]:
        label((35, top + 23), name, 29, INK, "lm")
        for left, value, color in [(230, student, GREEN), (507, teacher, ORANGE)]:
            end = left + 233 * value
            draw.rounded_rectangle((left, top, left + 233, top + 47), radius=12, fill="#e7ebe4")
            draw.rounded_rectangle((left, top, end, top + 47), radius=12, fill=color)
            if value >= 0.4:
                label((left + 13, top + 23), f"{value:.1f}", 29, "white", "lm")
            else:
                label((end + 12, top + 23), f"{value:.1f}", 29, color, "lm")
    draw.line((32, 312, 767, 312), fill=LINE, width=2)
    label((400, 345), "用学生概率加权 → reverse KL ≈ 0.193", 28, INK, "mm")
    save(picture, "opd-distributions.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default="C:/Windows/Fonts/msyh.ttc", help="Path to a font with CJK glyphs")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def make(height):
        picture = Image.new("RGB", (800, height), PAPER)
        draw = ImageDraw.Draw(picture)

        def label(position, content, size, color=INK, anchor="la"):
            draw.text(position, content, font=font(args.font, size), fill=color, anchor=anchor)

        return picture, draw, label

    def save(picture, name):
        picture.save(OUTPUT / name, optimize=True)

    for draw_figure in (loss_map, clip_chart, opd_chart):
        draw_figure(make, save)


if __name__ == "__main__":
    main()
