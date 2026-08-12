from dolfin import *


def compute_mass(u, dx):
    return assemble(u * dx)


def compute_source_total(f_expr, dx):
    return assemble(f_expr * dx)


def compute_inner_flux(u, kappa_value, c_int_value, alpha_value, ds_inner):
    return assemble(kappa_value * (c_int_value - alpha_value * u) * ds_inner)


def compute_outer_flux(u, Df_value, n, ds_outer):
    return assemble(-Df_value * dot(grad(u), n) * ds_outer)


def compute_balance_residual(mass_new, mass_old, dt_value, source_total, flux_in, flux_out):
    return (mass_new - mass_old) / dt_value - (source_total + flux_in - flux_out)


def compute_relative_balance_residual(balance_abs, source_total, flux_in, flux_out):
    scale = max(1.0, abs(source_total) + abs(flux_in) + abs(flux_out))
    return abs(balance_abs) / scale