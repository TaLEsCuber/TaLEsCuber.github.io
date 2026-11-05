"""Analytic teaching figures and checks accompanying a reading of Tahara & Yasui.

Usage:
    python figures-and-checks.py --output /path/to/image-directory --qa /path/to/qa
    python figures-and-checks.py --output ./figures --source-pdf /path/to/paper.pdf

Requires numpy and matplotlib. Source-figure extraction additionally needs
PyMuPDF. The optional --audit-tools directory may contain the Nature Figure
alignment helper. No RCWA, adjoint optimization, CODE V model, or full-lens
electromagnetic simulation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def setup_style():
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]
    chosen = next((p for p in candidates if p.exists()), None)
    family = FontProperties(fname=chosen).get_name() if chosen else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "axes.linewidth": 0.8,
    })


def save(fig, name, output, qa, audit_tools):
    fig.canvas.draw()
    if audit_tools:
        sys.path.insert(0, str(audit_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(qa / f"{name}.alignment.json"),
            overlay_svg=str(qa / f"{name}.alignment.svg"), strict=True,
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=320)
    plt.close(fig)


def analytic_figures(output, qa, audit_tools):
    setup_style()
    blue, orange = "#24788E", "#C67136"
    # Two complementary quantities under the same normal-incidence air model.
    na = np.linspace(0.1, 0.95, 500)
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 86 / 25.4))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.22, top=0.84, wspace=0.36)
    for label, ax in zip("ab", axes):
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-4, 6),
                    textcoords="offset points", ha="right", va="bottom", weight="bold", fontsize=11)
        ax.set_xlim(0.1, 0.95)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_xlabel("局部 NA（正入射、出射为空气）")
        ax.grid(axis="y", color="#E5E9EB", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].plot(na, 1 / na, color=blue, linewidth=2)
    axes[0].set_ylabel("完整相位周期 / 真空波长")
    axes[0].set_ylim(0, 10.5)
    axes[0].set_yticks([0, 2, 4, 6, 8, 10])
    axes[1].plot(na, 940 / (380 * na), color=orange, linewidth=2)
    axes[1].set_ylabel("每个 2π 周期内的晶格间隔数")
    axes[1].set_ylim(0, 26)
    axes[1].set_yticks([0, 5, 10, 15, 20, 25])
    fig.text(0.5, 0.97, "大 NA 将局部相位变化压缩到更短的距离", ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.035, "解析示意：p/λ = 380/940；不代表论文实际相位处方或效率", ha="center", fontsize=8)
    save(fig, "01-phase-gradient", output, qa, audit_tools)

    fig, ax = plt.subplots(figsize=(180 / 25.4, 86 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.975, bottom=0.35, top=0.83)
    r1 = np.linspace(455.6, 848.9, 400)
    r2 = np.linspace(848.9, 1080, 250)
    pitch1, pitch2 = np.deg2rad([0.0552, 0.0276])
    ax.plot(r1, r1 * pitch1 * 1000, color=blue, linewidth=2, label="角向周期 0.0552°")
    ax.plot(r2, r2 * pitch2 * 1000, color=orange, linewidth=2, label="角向周期 0.0276°")
    ax.plot([848.9, 848.9], [848.9 * pitch1 * 1000, 848.9 * pitch2 * 1000],
            color="#838C93", linestyle=":", linewidth=1.1)
    ax.axhline(940, color="#555D65", linestyle="--", linewidth=1, label="空气波长 940 nm")
    ax.axhline(940 / 1.44, color="#979DA1", linestyle="-.", linewidth=1, label="封装内波长约 653 nm")
    ax.set(xlim=(440, 1095), ylim=(350, 990), xlabel="距透镜中心的半径 / μm", ylabel="超胞周向宽度 / nm")
    ax.set_xticks([450, 600, 750, 900, 1050])
    ax.set_yticks([400, 500, 600, 700, 800, 900])
    ax.grid(axis="y", color="#E5E9EB", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", bbox_to_anchor=(0.55, 0.075), frameon=False, ncol=2, fontsize=8)
    fig.text(0.5, 0.97, "角向周期在 848.9 μm 处减半，周向宽度随之减半", ha="center", va="top", fontsize=11)
    fig.text(0.5, 0.035, "按论文给出的舍入参数计算；曲线不是 RCWA 仿真", ha="center", fontsize=8)
    save(fig, "03-angular-width", output, qa, audit_tools)


def extract_source_figures(pdf, output, qa):
    import pymupdf
    # Coordinates are fractions of complete PDF pages, preserving panel labels.
    crops = [
        (2, "02-paper-partition", (0.18, 0.075, 0.80, 0.372), "Fig. 1"),
        (6, "02b-paper-optical-system", (0.51, 0.075, 0.94, 0.295), "Fig. 7"),
        (3, "04-paper-objective", (0.18, 0.078, 0.82, 0.372), "Fig. 2"),
        (4, "05-paper-constraints", (0.20, 0.076, 0.81, 0.382), "Fig. 3"),
        (5, "06-paper-sequential", (0.075, 0.408, 0.925, 0.695), "Fig. 6"),
        (9, "07-paper-boundaries", (0.075, 0.075, 0.925, 0.458), "Fig. 13"),
        (10, "08-paper-offset", (0.505, 0.438, 0.950, 0.750), "Fig. 16"),
        (12, "09-paper-performance", (0.072, 0.080, 0.491, 0.318), "Fig. 20"),
    ]
    doc = pymupdf.open(pdf)
    manifest = {"doi": "10.1051/epjam/2024001", "license": "CC BY 4.0",
                "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "figures": []}
    for page_number, name, fractions, figure in crops:
        page = doc[page_number - 1]
        width, height = page.rect.width, page.rect.height
        a, b, c, d = fractions
        clip = pymupdf.Rect(a * width, b * height, c * width, d * height)
        page.get_pixmap(matrix=pymupdf.Matrix(2.8, 2.8), clip=clip, alpha=False).save(output / f"{name}.png")
        manifest["figures"].append({"file": f"{name}.png", "page": page_number,
                                      "original_figure": figure, "crop_fraction": list(fractions),
                                      "processing": "Crop and rasterize only; original labels and data unchanged"})
    (qa / "source-figure-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def check_formulas(qa):
    r0, reset = 455.6, 848.9
    p1, p2 = np.deg2rad([0.0552, 0.0276])
    widths = {"inner_nm": r0 * p1 * 1000,
              "reset_inside_nm": reset * p1 * 1000,
              "reset_outside_nm": reset * p2 * 1000,
              "outer_at_1080um_nm": 1080 * p2 * 1000}
    assert math.isclose(widths["reset_inside_nm"] / widths["reset_outside_nm"], 2)
    assert max(widths.values()) < 940
    # Verify the phase-shift optimum using an analytic phasor sum and a grid.
    delta = np.array([-.80, -.72, -.85, -.77]) * np.pi
    weight = np.array([.8, .9, .7, .85])
    z = np.sum(weight * np.exp(1j * delta))
    optimum = (-np.angle(z) / (2 * np.pi)) % 1
    grid = np.linspace(0, 1, 200001)
    q = np.mean(weight[None, :] * np.abs(np.exp(1j * (delta[None, :] + 2 * np.pi * grid[:, None])) - 1) ** 2, axis=1)
    grid_optimum = float(grid[int(np.argmin(q))])
    assert abs(optimum - grid_optimum) < 5.1e-6
    # Exact hyperbolic-phase ring boundaries versus an integer phase cycle.
    wavelength, focal = .940, 100.0
    orders = np.arange(1, 41)
    radii = np.sqrt(2 * focal * orders * wavelength + (orders * wavelength) ** 2)
    phase = -2 * np.pi / wavelength * (np.sqrt(focal * focal + radii * radii) - focal)
    residual = float(np.max(np.abs(phase + 2 * np.pi * orders)))
    assert residual < 1e-10
    result = {"scope": "Analytic geometry and formula checks only; no reproduction of unpublished layout or electromagnetic efficiency",
              "paper": {"doi": "10.1051/epjam/2024001", "wavelength_nm": 940,
                        "fov_deg": 131, "f_number": 1.6, "center_boundary_um": 455.6,
                        "reset_radius_um": 848.9, "angle_samples": 10},
              "circumferential_widths": widths,
              "air_thin_lens_na_from_f_number_illustrative_only": 1 / np.sqrt(1 + 4 * 1.6 ** 2),
              "phase_shift_toy": {"analytic_optimum_fraction": float(optimum), "grid_optimum_fraction": grid_optimum,
                                  "absolute_error_fraction": abs(float(optimum) - grid_optimum)},
              "hyperbolic_ring_max_phase_residual_rad": residual,
              "all_checks_passed": True}
    (qa / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--audit-tools", type=Path)
    args = parser.parse_args()
    args.qa = args.qa or args.output
    args.output.mkdir(parents=True, exist_ok=True)
    args.qa.mkdir(parents=True, exist_ok=True)
    analytic_figures(args.output, args.qa, args.audit_tools)
    if args.source_pdf:
        extract_source_figures(args.source_pdf, args.output, args.qa)
    check_formulas(args.qa)


if __name__ == "__main__":
    main()
