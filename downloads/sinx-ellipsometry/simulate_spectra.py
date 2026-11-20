"""Reproduce the illustrative Ψ and Δ spectra for an unpatterned SiNx film.

Requires NumPy and Matplotlib. A CJK font is optional.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


INCIDENCE_DEG = 65.0
DESIGN_WAVELENGTH_NM = 369.5


def fused_silica_index(wavelength_um):
    """Malitson Sellmeier equation at 20 °C (doi:10.1364/JOSA.55.001205)."""
    wavelength_sq = wavelength_um**2
    sellmeier_terms = (
        (0.6961663, 0.0684043),
        (0.4079426, 0.1162414),
        (0.8974794, 9.896161),
    )
    index_sq = 1 + sum(
        strength * wavelength_sq / (wavelength_sq - resonance_um**2)
        for strength, resonance_um in sellmeier_terms
    )
    return np.sqrt(index_sq)


def film_reflection_coefficients(wavelength_nm, thickness_nm, incidence_deg):
    """Return complex electric-field coefficients (r_p, r_s), with e^(-iωt)."""
    wavelength_um = wavelength_nm * 1e-3
    air_index = 1.0
    film_index = 1.98 + 0.015 / wavelength_um**2  # Illustrative Cauchy model, k = 0.
    substrate_index = fused_silica_index(wavelength_um)
    transverse_index = air_index * np.sin(np.deg2rad(incidence_deg))
    air_cos, film_cos, substrate_cos = (
        np.sqrt(1 - (transverse_index / index) ** 2)
        for index in (air_index, film_index, substrate_index)
    )

    def interface(index_i, index_j, cos_i, cos_j):
        p = (index_j * cos_i - index_i * cos_j) / (index_j * cos_i + index_i * cos_j)
        s = (index_i * cos_i - index_j * cos_j) / (index_i * cos_i + index_j * cos_j)
        return p, s

    p01, s01 = interface(air_index, film_index, air_cos, film_cos)
    p12, s12 = interface(film_index, substrate_index, film_cos, substrate_cos)
    round_trip = np.exp(4j * np.pi * film_index * film_cos * thickness_nm / wavelength_nm)

    def total(front, buried):
        return (front + buried * round_trip) / (1 + front * buried * round_trip)

    return total(p01, p12), total(s01, s12)


def make_figure():
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei")
    chinese_font = next((font for font in chinese_fonts if font in installed_fonts), None)
    fallback_fonts = ["DejaVu Sans", "sans-serif"]
    plt.rcParams.update({
        "font.family": [chinese_font, *fallback_fonts] if chinese_font else fallback_fonts,
        "font.size": 12,
        "axes.edgecolor": "#bac6d0",
        "axes.labelcolor": "#223249",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#455b71",
        "ytick.color": "#455b71",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    demo_wavelength_nm = np.linspace(350, 900, 2201)
    fig, (psi_axis, delta_axis) = plt.subplots(2, 1, sharex=True, figsize=(9.3, 5.9))
    fig.subplots_adjust(left=0.115, right=0.971, bottom=0.157, top=0.767, hspace=0.22)

    if chinese_font:
        title = "膜厚改变椭偏光谱"
        subtitle = "模拟  ·  65° 入射  ·  空气 / SiNₓ / 熔融石英"
        xlabel, delta_label = "真空波长（nm）", "Δ（°，模 360°）"
        note = "SiNₓ 膜采用示例 Cauchy 色散，k = 0。石英采用 Malitson 色散，模型忽略背面返回光。"
    else:
        title = "Film thickness changes the ellipsometric spectra"
        subtitle = "Simulation  ·  65° incidence  ·  Air / SiNₓ / fused silica"
        xlabel, delta_label = "Vacuum wavelength (nm)", "Δ (°, mod 360°)"
        note = "Illustrative Cauchy SiNₓ (k = 0) and Malitson silica. Rear reflection excluded."

    fig.text(0.115, 0.939, title, fontsize=22, fontweight="bold", color="#16283c", va="top")
    fig.text(0.115, 0.862, subtitle, fontsize=12, color="#455b71", va="top")
    fig.text(0.115, 0.037, note, fontsize=10, color="#455b71", va="bottom")

    series = ((800, "#176985", "-"), (790, "#ad461c", (0, (5, 2.5))))
    for thickness_nm, color, line_style in series:
        p, s = film_reflection_coefficients(demo_wavelength_nm, thickness_nm, INCIDENCE_DEG)
        ratio = p / s
        psi_deg = np.rad2deg(np.arctan(np.abs(ratio)))
        delta_deg = np.mod(np.angle(ratio, deg=True), 360)
        crossings = np.flatnonzero(np.abs(np.diff(delta_deg)) > 180) + 1
        delta_deg[crossings] = np.nan  # Break the drawing at the 360° boundary.
        style = {"color": color, "linestyle": line_style, "linewidth": 2.25}
        psi_axis.plot(demo_wavelength_nm, psi_deg, label=f"{thickness_nm} nm", **style)
        delta_axis.plot(demo_wavelength_nm, delta_deg, **style)

    marker_style = {"color": "#7b8794", "linestyle": (0, (1.2, 2.4)), "linewidth": 1.45}
    for axis in (psi_axis, delta_axis):
        axis.axvline(DESIGN_WAVELENGTH_NM, zorder=1.8, **marker_style)
        axis.grid(axis="y", color="#e1e8ef", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(length=3.5, width=0.8, pad=5, labelsize=11)
    psi_axis.tick_params(axis="x", length=0)
    psi_axis.set_ylabel("Ψ (°)", fontsize=12)
    psi_axis.set_ylim(4, 16)
    psi_axis.set_yticks((5, 10, 15))
    delta_axis.set_ylabel(delta_label, fontsize=12)
    delta_axis.set_ylim(-10, 370)
    delta_axis.set_yticks((0, 90, 180, 270, 360))
    delta_axis.set(xlim=(350, 900), xlabel=xlabel)
    delta_axis.set_xticks((350, 450, 550, 650, 750, 850, 900))

    handles, labels = psi_axis.get_legend_handles_labels()
    handles.append(Line2D([], [], **marker_style))
    labels.append(f"λ = {DESIGN_WAVELENGTH_NM} nm")
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.777),
               ncols=3, frameon=False, fontsize=11, handlelength=2.2, columnspacing=1.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("03-spectra.png"))
    parser.add_argument("--vector", action="append", type=Path, default=[],
                        help="optional vector export, e.g. spectra.pdf or spectra.svg")
    parser.add_argument("--alignment-tools", type=Path, help="optional figure audit script directory")
    args = parser.parse_args()
    fig = make_figure()
    if args.alignment_tools:
        sys.path.insert(0, str(args.alignment_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        report = require_matplotlib_panel_alignment(fig, panel_ids=["psi", "delta"])
        print(f"Panel alignment: {report['verdict']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    for vector_path in args.vector:
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(vector_path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
