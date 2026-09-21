#!/usr/bin/env python3
"""Reproduce the figures and numerical checks in the Rabi-oscillation article."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


COLORS = {
    "blue": "#2864dc",
    "orange": "#ed8b23",
    "green": "#27946f",
    "red": "#ce3c55",
    "purple": "#7c57c2",
    "gray": "#617083",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )


def rabi_probability(t: np.ndarray, omega: float, delta: float) -> np.ndarray:
    omega_r = np.hypot(omega, delta)
    if omega_r == 0:
        return np.zeros_like(t, dtype=float)
    return (omega / omega_r) ** 2 * np.sin(omega_r * t / 2) ** 2


def figure_population_and_detuning(output_dir: Path) -> None:
    theta = np.linspace(0, 4 * np.pi, 1200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    ratios = [0.0, 0.5, 1.0, 2.0]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    for ratio, color in zip(ratios, colors):
        axes[0].plot(
            theta / np.pi,
            rabi_probability(theta, 1.0, ratio),
            lw=2.1,
            color=color,
            label=rf"$\Delta/\Omega={ratio:g}$",
        )
    axes[0].set(xlabel=r"Pulse area $\Omega t/\pi$", ylabel=r"Excited-state probability $P_e$")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(ncol=2, loc="upper right")
    axes[0].set_title("Time-domain Rabi oscillations")

    detuning = np.linspace(-6, 6, 1800)
    pulse_time = np.pi
    response = np.array([rabi_probability(np.array([pulse_time]), 1.0, d)[0] for d in detuning])
    axes[1].plot(detuning, response, lw=2.2, color=COLORS["purple"])
    axes[1].axvline(0, color=COLORS["gray"], lw=1, ls="--")
    axes[1].fill_between(detuning, 0, response, color=COLORS["purple"], alpha=0.10)
    axes[1].set(xlabel=r"Detuning $\Delta/\Omega$", ylabel=r"$P_e(t_\pi)$")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_title(r"Detuning response of a resonant $\pi$ pulse")
    fig.tight_layout()
    fig.savefig(output_dir / "population-and-detuning.png", bbox_inches="tight")
    plt.close(fig)


def bloch_trajectory(delta_ratio: float, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    omega = 1.0
    delta = delta_ratio * omega
    omega_r = np.hypot(omega, delta)
    t = np.linspace(0, 2 * np.pi / omega_r, points)
    u = -(omega * delta / omega_r**2) * (1 - np.cos(omega_r * t))
    v = (omega / omega_r) * np.sin(omega_r * t)
    w = -(delta**2 + omega**2 * np.cos(omega_r * t)) / omega_r**2
    return np.vstack((u, v, w)), np.array([omega, 0.0, delta]) / omega_r


def draw_bloch_panel(ax: plt.Axes, delta_ratio: float, title: str) -> None:
    azimuth = np.linspace(0, 2 * np.pi, 60)
    polar = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color="#bcc5d3", alpha=0.18, lw=0.45)
    ring = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(ring), np.sin(ring), 0 * ring, color="#9aa6b6", lw=0.8, alpha=0.55)
    ax.plot([0, 0], [0, 0], [-1.08, 1.08], color="#8b97a8", lw=0.8)

    trajectory, axis = bloch_trajectory(delta_ratio)
    ax.plot(*trajectory, color=COLORS["blue"], lw=3.0)
    ax.scatter(*trajectory[:, 0], color=COLORS["blue"], s=24)
    ax.scatter([0], [0], [-1], color=COLORS["red"], s=34, label=r"initial $|g\rangle$")
    ax.quiver(0, 0, 0, *(1.15 * axis), color=COLORS["orange"], lw=2.4, arrow_length_ratio=0.12)
    ax.text(*(1.28 * axis), r"$\mathbf{b}$", color=COLORS["orange"], fontsize=12)
    ax.text(0, 0, 1.17, r"$|e\rangle$", ha="center")
    ax.text(0, 0, -1.22, r"$|g\rangle$", ha="center")
    ax.set_title(title, pad=8)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
    ax.set_xlabel(r"$u$", labelpad=-6)
    ax.set_ylabel(r"$v$", labelpad=-6)
    ax.set_zlabel(r"$w$", labelpad=-6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=23, azim=38)


def figure_bloch(output_dir: Path) -> None:
    fig = plt.figure(figsize=(9.5, 4.3))
    draw_bloch_panel(fig.add_subplot(121, projection="3d"), 0.0, r"On resonance: $\Delta=0$")
    draw_bloch_panel(fig.add_subplot(122, projection="3d"), 1.0, r"Off resonance: $\Delta=\Omega$")
    fig.suptitle("Bloch-vector precession around the effective field", y=0.98, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "bloch-trajectories.png", bbox_inches="tight")
    plt.close(fig)


def damped_bloch_rhs(_: float, state: np.ndarray, omega: float, gamma1: float, gamma2: float) -> np.ndarray:
    u, v, w = state
    return np.array([-gamma2 * u, -omega * w - gamma2 * v, omega * v - gamma1 * (w + 1)])


def damped_analytic(t: np.ndarray, omega: float, gamma1: float, gamma2: float) -> np.ndarray:
    alpha = (gamma1 + gamma2) / 2
    omega_d = np.sqrt(omega**2 - (gamma1 - gamma2) ** 2 / 4)
    pe_ss = omega**2 / (2 * (omega**2 + gamma1 * gamma2))
    return pe_ss * (1 - np.exp(-alpha * t) * (np.cos(omega_d * t) + alpha / omega_d * np.sin(omega_d * t)))


def lab_frame_rhs(tau: float, state: np.ndarray, carrier_ratio: float) -> np.ndarray:
    ce, cg = state
    h = np.array(
        [[carrier_ratio / 2, np.cos(carrier_ratio * tau)], [np.cos(carrier_ratio * tau), -carrier_ratio / 2]],
        dtype=complex,
    )
    return -1j * h @ np.array([ce, cg])


def integrate_lab_frame(carrier_ratio: float, tau: np.ndarray) -> np.ndarray:
    solution = solve_ivp(
        lab_frame_rhs,
        (tau[0], tau[-1]),
        np.array([0.0 + 0.0j, 1.0 + 0.0j]),
        t_eval=tau,
        args=(carrier_ratio,),
        rtol=2e-10,
        atol=2e-12,
        max_step=2 * np.pi / carrier_ratio / 50,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.abs(solution.y[0]) ** 2


def figure_damping_and_rwa(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    omega, gamma1, gamma2 = 1.0, 0.12, 0.08
    t = np.linspace(0, 8 * np.pi, 1600)
    numeric = solve_ivp(
        damped_bloch_rhs,
        (t[0], t[-1]),
        np.array([0.0, 0.0, -1.0]),
        t_eval=t,
        args=(omega, gamma1, gamma2),
        rtol=2e-10,
        atol=2e-12,
    )
    pe_numeric = (numeric.y[2] + 1) / 2
    pe_analytic = damped_analytic(t, omega, gamma1, gamma2)
    axes[0].plot(t / (2 * np.pi), rabi_probability(t, omega, 0), color="#aeb7c4", lw=1.4, label="No dissipation")
    axes[0].plot(t / (2 * np.pi), pe_analytic, color=COLORS["red"], lw=2.3, label="Analytic damped solution")
    axes[0].plot(t[::35] / (2 * np.pi), pe_numeric[::35], "o", ms=3, color=COLORS["blue"], label="Bloch-equation integration")
    axes[0].set(xlabel=r"Time $t/T_R$", ylabel=r"Excited-state probability $P_e$")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_title("Damped Rabi oscillations")
    axes[0].legend(loc="upper right", fontsize=8.8)

    tau = np.linspace(0, 4.5 * np.pi, 3000)
    axes[1].plot(tau / (2 * np.pi), np.sin(tau / 2) ** 2, color="#232d3b", lw=2.1, ls="--", label="RWA")
    for ratio, color in [(20.0, COLORS["green"]), (4.0, COLORS["orange"])]:
        exact = integrate_lab_frame(ratio, tau)
        axes[1].plot(tau / (2 * np.pi), exact, color=color, lw=1.7, label=rf"Exact, $\omega_0/\Omega={ratio:g}$")
    axes[1].set(xlabel=r"Time $t/T_R$", ylabel=r"Excited-state probability $P_e$")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_title("Laboratory-frame dynamics versus RWA")
    axes[1].legend(loc="upper right", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(output_dir / "damping-and-rwa.png", bbox_inches="tight")
    plt.close(fig)

    assert np.max(np.abs(pe_numeric - pe_analytic)) < 2e-8
    weak_error = np.max(np.abs(integrate_lab_frame(20.0, tau) - np.sin(tau / 2) ** 2))
    assert weak_error < 0.04


def check_rwa_formula() -> None:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    initial = np.array([0.0, 1.0], dtype=complex)
    omega, delta = 1.37, -0.64
    times = np.linspace(0, 11, 600)

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        return -0.5j * (delta * sigma_z + omega * sigma_x) @ state

    numeric = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times, rtol=2e-11, atol=2e-13)
    error = np.max(np.abs(np.abs(numeric.y[0]) ** 2 - rabi_probability(times, omega, delta)))
    if error >= 2e-9:
        raise AssertionError(f"Rabi formula check failed: max error={error:.3e}")
    print(f"RWA formula check: max |numerical - analytic| = {error:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("rabi-figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    check_rwa_formula()
    figure_population_and_detuning(args.output_dir)
    figure_bloch(args.output_dir)
    figure_damping_and_rwa(args.output_dir)
    print(f"Wrote figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
