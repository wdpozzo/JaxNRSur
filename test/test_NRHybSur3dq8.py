import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxnrsur.Harmonics import SpinWeightedSphericalHarmonics
from jaxnrsur.NRHybSur3dq8 import (
    NRHybSur3dq8DataLoader,
    NRHybSur3dq8Model,
    get_T3_phase,
)


@pytest.fixture(scope="module")
def model():
    # Use default modelist for basic test
    return NRHybSur3dq8Model()


@pytest.fixture(scope="module")
def time():
    # Reasonable time array for waveform
    return jnp.linspace(-1000, 100, 1000)


@pytest.fixture(scope="module")
def params():
    # Example parameters: q, chi1z, chi2z
    return jnp.array([0.9, 0.1, 0.1])


def _check_waveform_tuple(h_tuple, shape):
    assert isinstance(h_tuple, tuple) and len(h_tuple) == 2
    hp, hc = h_tuple
    assert hp.shape == shape
    assert hc.shape == shape
    assert jnp.issubdtype(hp.dtype, jnp.floating)
    assert jnp.issubdtype(hc.dtype, jnp.floating)
    assert not jnp.isnan(hp).any()
    assert not jnp.isnan(hc).any()


@pytest.mark.parametrize(
    "waveform_fn, param_shape",
    [
        (lambda m, t, p: m(t, p), (1000,)),  # basic
        (
            lambda m, t, p: eqx.filter_jit(
                eqx.filter_vmap(m.get_waveform_geometric, in_axes=(None, 0, None, None))
            )(t, jnp.repeat(p[None, :], 5, axis=0), 0.0, 0.0),
            (5, 1000),
        ),  # jit+vmap
    ],
)
def test_waveform_variants(model, time, params, waveform_fn, param_shape):
    h_tuple = waveform_fn(model, time, params)
    _check_waveform_tuple(h_tuple, param_shape)


def test_model_initialization(model):
    assert isinstance(model, NRHybSur3dq8Model)
    assert model.n_modes > 0
    assert hasattr(model, "data")
    assert hasattr(model, "harmonics")


def test_inertial_mode_reconstruction(monkeypatch):
    grid = jnp.array([-1200.0, -1000.0, -800.0, -600.0])
    constant = jnp.ones_like(grid)
    phase_basis = 0.05 * (grid + 1000.0)

    def component(basis, predictor):
        return {"n_nodes": 1, "predictors": [predictor], "eim_basis": basis[None]}

    modes = {
        (2, 0): {
            "real": component(constant, lambda _: 0.2),
            "imag": component(constant, lambda _: 0.0),
        },
        (2, 2): {
            "amp": component(constant, lambda fit: 1 + fit[0, 0]),
            "phase": component(phase_basis, lambda fit: fit[0, 2]),
        },
        (3, 3): {
            "real": component(constant, lambda fit: fit[0, 1]),
            "imag": component(constant, lambda fit: 0.5 * fit[0, 1]),
        },
    }
    monkeypatch.setattr(
        "jaxnrsur.NRHybSur3dq8.load_data", lambda *_: {"domain": grid}
    )
    monkeypatch.setattr(
        NRHybSur3dq8DataLoader,
        "read_single_mode",
        lambda self, file, mode: {"mode": mode, **modes[mode]},
    )
    model = NRHybSur3dq8Model(modelist=[(2, 0), (2, 2), (3, 3)])
    params = jnp.array([3.0, 0.4, -0.2])
    theta, phi = 1.1, 0.4

    chi_a = 0.3
    eta = 3.0 / 16.0
    chi_hat = 0.1 + 0.5 * chi_a / (1 - 76 * eta / 113)
    phase_22 = get_T3_phase(params[0], grid) - chi_a * phase_basis
    h_22 = (1 + jnp.log(3.0)) * jnp.exp(1j * phase_22)
    h_33 = chi_hat * (1 + 0.5j) * jnp.exp(1.5j * phase_22)
    expected_modes = {
        (2, 0): 0.2,
        (2, 2): h_22,
        (2, -2): jnp.conj(h_22),
        (3, 3): h_33,
        (3, -3): -jnp.conj(h_33),
    }
    expected = sum(
        strain * SpinWeightedSphericalHarmonics(-2, ell, m)(theta, phi)
        for (ell, m), strain in expected_modes.items()
    )

    for evaluate, times in (
        (model.get_waveform_at_native_indices, jnp.arange(grid.size)),
        (model.get_waveform_geometric_linear, grid),
        (model.get_waveform_geometric_dense, grid),
        (model.get_waveform_geometric, grid),
    ):
        plus, cross = evaluate(times, params, theta, phi)
        # The sparse evaluator rescales a unit-mass-ratio T3 spline, which
        # rounds differently from the direct T3 evaluation in float32.
        assert jnp.allclose(plus - 1j * cross, expected, rtol=1e-4, atol=1e-6), (
            evaluate.__name__
        )


def test_grad_waveform_time(model, time, params):
    # Gradient with respect to time
    def target(time_):
        hp, hc = model(time_, params)
        return jnp.sum(hp) + jnp.sum(hc)

    grad_time = jax.grad(target)
    grad_val = grad_time(time)
    assert grad_val.shape == time.shape
    assert not jnp.isnan(grad_val).any()


def test_grad_waveform_params(model, time, params):
    # Gradient with respect to params
    def target(params_):
        hp, hc = model(time, params_)
        return jnp.sum(hp) + jnp.sum(hc)

    grad_params = jax.grad(target)
    grad_val = grad_params(params)
    assert grad_val.shape == params.shape
    assert not jnp.isnan(grad_val).any()


def test_sparse_linear_matches_native_grid_interpolation(model, params):
    grid = model.data.sur_time
    time = jnp.concatenate(
        (
            jnp.asarray([grid[0] - 1.0, grid[0]]),
            jnp.linspace(grid[0], grid[-1], 256),
            jnp.asarray([grid[-1], grid[-1] + 1.0]),
        )
    )
    native_indices = jnp.arange(grid.size, dtype=jnp.int32)
    native_plus, native_cross = model.get_waveform_at_native_indices(
        native_indices, params
    )
    expected_plus = jnp.interp(
        time, grid, native_plus, left=0.0, right=0.0
    )
    expected_cross = jnp.interp(
        time, grid, native_cross, left=0.0, right=0.0
    )
    actual_plus, actual_cross = model.get_waveform_geometric_linear(
        time, params
    )
    assert jnp.allclose(actual_plus, expected_plus, rtol=2e-11, atol=2e-13)
    assert jnp.allclose(actual_cross, expected_cross, rtol=2e-11, atol=2e-13)


def test_sparse_cubic_matches_dense_cubic(model, params):
    grid = model.data.sur_time
    time = jnp.concatenate(
        (
            jnp.asarray([grid[0] - 1.0, grid[0]]),
            jnp.linspace(-900.123, 90.456, 256),
            jnp.asarray([grid[-1], grid[-1] + 1.0]),
        )
    )
    expected = model.get_waveform_geometric_dense(time, params, 0.7, 0.2)
    actual = model.get_waveform_geometric(time, params, 0.7, 0.2)
    assert jnp.allclose(actual[0], expected[0], rtol=2e-9, atol=2e-11)
    assert jnp.allclose(actual[1], expected[1], rtol=2e-9, atol=2e-11)


def test_sparse_cubic_derivatives_match_dense_and_are_finite(model, params):
    time = jnp.linspace(-900.123, 90.456, 128)

    def dense_objective(values):
        plus, cross = model.get_waveform_geometric_dense(
            time, values, 0.7, 0.2
        )
        return jnp.mean(plus**2 + cross**2)

    def sparse_objective(values):
        plus, cross = model.get_waveform_geometric(time, values, 0.7, 0.2)
        return jnp.mean(plus**2 + cross**2)

    dense_gradient = jax.grad(dense_objective)(params)
    sparse_gradient = jax.grad(sparse_objective)(params)
    sparse_hessian = jax.hessian(sparse_objective)(params)
    assert jnp.allclose(
        sparse_gradient,
        dense_gradient,
        rtol=2e-8,
        atol=2e-10,
    )
    assert jnp.all(jnp.isfinite(sparse_gradient))
    assert jnp.all(jnp.isfinite(sparse_hessian))


def test_sparse_linear_derivatives_are_finite(model, params):
    time = jnp.linspace(-900.123, 90.456, 128)

    def objective(values):
        plus, cross = model.get_waveform_geometric_linear(time, values)
        return jnp.mean(plus**2 + cross**2)

    gradient = jax.grad(objective)(params)
    hessian = jax.hessian(objective)(params)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.isfinite(hessian))


def test_sparse_linear_gradient_matches_native_grid_interpolation(model, params):
    time = jnp.linspace(-900.123, 90.456, 128)

    def reference_objective(values):
        grid = model.data.sur_time
        native_indices = jnp.arange(grid.size, dtype=jnp.int32)
        plus, cross = model.get_waveform_at_native_indices(
            native_indices, values
        )
        interpolated_plus = jnp.interp(time, grid, plus)
        interpolated_cross = jnp.interp(time, grid, cross)
        return jnp.mean(interpolated_plus**2 + interpolated_cross**2)

    def sparse_objective(values):
        plus, cross = model.get_waveform_geometric_linear(time, values)
        return jnp.mean(plus**2 + cross**2)

    reference_gradient = jax.grad(reference_objective)(params)
    sparse_gradient = jax.grad(sparse_objective)(params)
    assert jnp.allclose(
        sparse_gradient,
        reference_gradient,
        rtol=2e-9,
        atol=2e-11,
    )
