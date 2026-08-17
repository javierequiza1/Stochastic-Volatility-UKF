"""Biyecciones theta <-> theta_u y jacobiano del cambio de variable.

Compartido integramente entre L=2 y L=3: la parametrizacion no restringida
(alpha, psi, omega, zeta) es identica en ambas arquitecturas, tal como se
describe en la Seccion "Transformacion a un espacio no restringido" del
anexo.
"""
from __future__ import annotations

import numpy as np

PARAM_NAMES = ["alpha", "phi", "sigma_eta", "rho"]
PARAM_NAMES_U = ["alpha", "psi", "omega", "zeta"]


def theta_to_unconstrained(theta):
    """theta = (alpha, phi, sigma_eta, rho) -> theta_u = (alpha, psi, omega, zeta)."""
    alpha, phi, sigma_eta, rho = theta
    return np.array([
        alpha,
        np.arctanh(np.clip(phi, -0.9999, 0.9999)),
        np.log(max(sigma_eta, 1e-8)),
        np.arctanh(np.clip(rho, -0.9999, 0.9999)),
    ])


def unconstrained_to_theta(theta_u):
    """theta_u = (alpha, psi, omega, zeta) -> theta = (alpha, phi, sigma_eta, rho)."""
    alpha, psi, omega, zeta = theta_u
    return np.array([alpha, np.tanh(psi), np.exp(omega), np.tanh(zeta)])


def log_jacobian(theta_u):
    """log|det J_g(theta_u)| = log(1 - phi^2) + omega + log(1 - rho^2)."""
    _, psi, omega, zeta = np.asarray(theta_u, dtype=float)
    phi = np.tanh(psi)
    rho = np.tanh(zeta)
    return float(
        np.log(max(1.0 - phi ** 2, 1e-30))
        + omega
        + np.log(max(1.0 - rho ** 2, 1e-30))
    )
