import h5py

import jax.numpy as jnp
import jax
from jaxnrsur.DataLoader import load_data, h5Group_to_dict, h5_mode_tuple
from jaxnrsur.Spline import CubicSpline, CubicSplineFactorization
from jaxnrsur.EIMPredictor import EIMpredictor
from jaxnrsur.Harmonics import SpinWeightedSphericalHarmonics
from jaxnrsur import WaveformModel
from jaxtyping import Array, Float, Int
import equinox as eqx


def get_T3_phase(q: float, t: Float[Array, " n"], t_ref: float = 1000.0) -> float:
    """
    Compute the T3 phase correction for the waveform model.

    Args:
        q (float): Mass ratio.
        t (Float[Array, " n"]): Time array.
        t_ref (float, optional): Reference time. Defaults to 1000.0.

    Returns:
        float: T3 phase correction value.
    """
    eta = q / (1 + q) ** 2
    theta_raw = (eta * (t_ref - t) / 5) ** (-1.0 / 8)
    theta_cal = (eta * (t_ref + 1000) / 5) ** (-1.0 / 8)
    return 2.0 / (eta * theta_raw**5) - 2.0 / (eta * theta_cal**5)


class NRHybSur3dq8DataLoader(eqx.Module):
    sur_time: Float[Array, " n_sample"]
    modes: list[dict]

    def __init__(
        self,
        modelist: list[tuple[int, int]] = [
            (2, 2),
            (2, 1),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (4, 4),
            (5, 5),
        ],
    ) -> None:
        """
        Initialize the NRHybSur3dq8DataLoader.

        Args:
            modelist (list[tuple[int, int]], optional): List of mode tuples to load.
        """
        data = load_data(
            "https://zenodo.org/records/3348115/files/NRHybSur3dq8.h5?download=1",
            "NRHybSur3dq8.h5",
        )
        self.sur_time = jnp.array(data["domain"])

        self.modes = []
        for i in range(len(modelist)):
            self.modes.append(self.read_single_mode(data, modelist[i]))

    def read_function(self, node_data: h5py.Group) -> dict:
        """
        Read a function group from the HDF5 node data and construct the EIM predictors.

        Args:
            node_data (h5py.Group): HDF5 group containing node function data.

        Returns:
            dict: Dictionary containing predictors, EIM basis, and metadata.

        Raises:
            ValueError: If required data is missing or incorrectly formatted.
        """
        try:
            result = {}
            if isinstance(node_data["n_nodes"], h5py.Dataset):
                n_nodes = int(node_data["n_nodes"][()])  # type: ignore
                result["n_nodes"] = n_nodes

                predictors = []
                for count in range(n_nodes):
                    try:
                        fit_data = node_data[
                            "node_functions/ITEM_%d/node_function/DICT_fit_data"
                            % (count)
                        ]
                    except ValueError:
                        raise ValueError("GPR Fit info doesn't exist")

                    assert isinstance(
                        fit_data, h5py.Group
                    ), "GPR Fit info is not a group"
                    res = h5Group_to_dict(fit_data)
                    node_predictor = EIMpredictor(res)
                    predictors.append(node_predictor)

                result["predictors"] = predictors
                result["eim_basis"] = jnp.array(node_data["ei_basis"])
                result["name"] = node_data["name"][()].decode("utf-8")  # type: ignore
                return result
            else:
                raise ValueError("n_nodes data doesn't exist")
        except ValueError:
            raise ValueError("n_nodes data doesn't exist")

    @staticmethod
    def make_empty_function(name: str, length: int) -> dict:
        """
        Create an empty function dictionary for a mode component.

        Args:
            name (str): Name of the function ('re' or 'im').
            length (int): Length of the EIM basis.

        Returns:
            dict: Dictionary representing an empty function.
        """
        return {
            "n_nodes": 1,
            "predictors": [lambda x: 1],
            "eim_basis": jnp.zeros((1, length)),
            "name": name,
        }

    def read_single_mode(self, file: h5py.File, mode: tuple[int, int]) -> dict:
        """
        Read a single mode's data from the HDF5 file.

        Args:
            file (h5py.File): HDF5 file object.
            mode (tuple[int, int]): Mode tuple (l, m).

        Returns:
            dict: Dictionary containing mode data.
        """
        result = {}
        data = file["sur_subs/%s/func_subs" % (h5_mode_tuple[mode])]
        assert isinstance(data, h5py.Group), "Mode data is not a group"
        if mode == (2, 2):
            result["phase"] = self.read_function(data["ITEM_0"])  # type: ignore
            result["amp"] = self.read_function(data["ITEM_1"])  # type: ignore
        else:
            if mode[1] != 0:
                result["real"] = self.read_function(data["ITEM_0"])  # type: ignore
                result["imag"] = self.read_function(data["ITEM_1"])  # type: ignore
            else:
                local_function = self.read_function(data["ITEM_0"])  # type: ignore
                if local_function["name"] == "re":
                    result["real"] = local_function
                    result["imag"] = self.make_empty_function(
                        "im", local_function["eim_basis"].shape[1]
                    )
                else:
                    result["imag"] = local_function
                    result["real"] = self.make_empty_function(
                        "re", local_function["eim_basis"].shape[1]
                    )
        result["mode"] = mode
        return result


class NRHybSur3dq8Model(WaveformModel):
    data: NRHybSur3dq8DataLoader
    mode_no22: list[dict]
    harmonics: list[SpinWeightedSphericalHarmonics]
    negative_harmonics: list[SpinWeightedSphericalHarmonics]
    mode_22_index: int
    m_mode: Int[Array, " n_modes-1"]
    negative_mode_prefactor: Int[Array, " n_modes-1"]
    sparse_cubic_enabled: bool
    t3_phase_reference: Float[Array, " n_sample"]
    t3_phase_reference_coefficients: Float[Array, " n_sample"]

    def __init__(
        self,
        modelist: list[tuple[int, int]] = [
            (2, 2),
            (2, 1),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (4, 4),
            (5, 5),
        ],
        precompute_spline_coefficients: bool = True,
    ):
        """
        Initialize NRHybSur3dq8Model.

        The model is described in the paper:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.064045

        Args:
            modelist (list[tuple[int, int]]): List of modes to be used.
            precompute_spline_coefficients (bool): Prepare the exact sparse
                cubic evaluator. Disable this when only the sparse linear
                evaluator is needed to avoid retaining a second basis-sized
                coefficient array.
        """
        self.data = NRHybSur3dq8DataLoader(modelist=modelist)  # type: ignore
        self.harmonics = []
        self.negative_harmonics = []
        negative_mode_prefactor = []
        for mode in modelist:
            if mode != (2, 2):
                self.harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], mode[1])
                )
                self.negative_harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], -mode[1])
                )
            if mode[1] > 0:
                negative_mode_prefactor.append((-1) ** mode[0])
            else:
                negative_mode_prefactor.append(0)

        self.mode_no22 = [
            self.data.modes[i] for i in range(len(self.data.modes)) if i != 0
        ]
        self.mode_22_index = int(
            jnp.where((jnp.array(modelist) == jnp.array([[2, 2]])).all(axis=1))[0][0]
        )
        self.m_mode = jnp.array(
            [modelist[i][1] for i in range(len(modelist)) if i != self.mode_22_index]
        )
        self.negative_mode_prefactor = jnp.array(negative_mode_prefactor)

        self.sparse_cubic_enabled = precompute_spline_coefficients
        if precompute_spline_coefficients:
            # Cubic spline construction is linear in the values on the fixed
            # surrogate grid. Factor each EIM basis row once so waveform calls
            # combine and gather only the two bracketing columns.
            spline_factorization = CubicSplineFactorization(self.data.sur_time)
            for mode in self.data.modes:
                for component_name in ("amp", "phase", "real", "imag"):
                    if component_name not in mode:
                        continue
                    component = mode[component_name]
                    component["spline_coefficients"] = jax.vmap(
                        spline_factorization.solve
                    )(component["eim_basis"])

            reference_q = jnp.asarray(1.0, dtype=self.data.sur_time.dtype)
            self.t3_phase_reference = get_T3_phase(
                reference_q,
                self.data.sur_time,
            )
            self.t3_phase_reference_coefficients = spline_factorization.solve(
                self.t3_phase_reference
            )
        else:
            self.t3_phase_reference = jnp.empty(
                0, dtype=self.data.sur_time.dtype
            )
            self.t3_phase_reference_coefficients = jnp.empty(
                0, dtype=self.data.sur_time.dtype
            )

    def __call__(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Compute the waveform for given time and source parameters.

        Args:
            time (Float[Array, " n_sample"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.
            theta (float, optional): Polar angle. Defaults to 0.0.
            phi (float, optional): Azimuthal angle. Defaults to 0.0.

        Returns:
            tuple: Plus and cross polarizations of the waveform.
        """
        return self.get_waveform_geometric(time, params, theta, phi)

    @property
    def n_modes(self) -> int:
        """
        Get the number of modes in the model.

        Returns:
            int: Number of modes.
        """
        return len(self.data.modes)

    @staticmethod
    def get_fit_params(
        params: Float[Array, " n_dim"],
    ) -> Float[Array, " n_dim"]:
        """
        Map the physical parameters onto the coordinates the surrogate fits were
        trained in: (q, chi1z, chi2z) -> (log(q), chi_hat, chi_a).

        Args:
            params (Float[Array, " n_dim"]): [q, chi1z, chi2z].

        Returns:
            Float[Array, " n_dim"]: [log(q), chi_hat, chi_a].
        """
        q, chi1z, chi2z = params[0], params[1], params[2]
        eta = q / (1 + q) ** 2
        chi_wt_avg = (q * chi1z + chi2z) / (1 + q)
        chi_hat = (chi_wt_avg - 38.0 * eta / 113.0 * (chi1z + chi2z)) / (
            1.0 - 76.0 * eta / 113.0
        )
        chi_a = (chi1z - chi2z) / 2.0
        return jnp.array([jnp.log(q), chi_hat, chi_a])

    @staticmethod
    def get_physical_mass_ratio(
        params: Float[Array, " n_dim"],
    ) -> Float:
        """Return the physical mass ratio used by the analytic T3 phase."""
        return params[0]

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, " n_dim"]
    ) -> Float[Array, " n_sample"]:
        """
        Construct the EIM basis given the source parameters.

        Args:
            eim_dict (dict): EIM dictionary containing predictors and basis.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            Float[Array, " n_sample"]: EIM basis evaluated at parameters.
        """
        result = jnp.zeros((eim_dict["n_nodes"], 1))
        for i in range(eim_dict["n_nodes"]):
            result = result.at[i].set(eim_dict["predictors"][i](params))
        return jnp.dot(eim_dict["eim_basis"].T, result[:, 0])

    @staticmethod
    def get_real_imag(
        mode: dict, params: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Get the real and imaginary parts for a mode given parameters.

        Args:
            mode (dict): Mode dictionary containing 'real' and 'imag' EIM data.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            tuple: Real and imaginary parts as arrays.
        """
        params = params[None]
        real = NRHybSur3dq8Model.get_eim(mode["real"], params)
        imag = NRHybSur3dq8Model.get_eim(mode["imag"], params)
        return real, imag

    @staticmethod
    def get_multi_real_imag(
        modes: list[dict], params: Float[Array, " n_dim"]
    ) -> tuple[list[Float[Array, " n_sample"]], list[Float[Array, " n_sample"]]]:
        """
        Get real and imaginary parts for multiple modes.

        Args:
            modes (list[dict]): List of mode dictionaries.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            tuple: Lists of real and imaginary arrays for each mode.
        """
        return jax.tree_util.tree_map(
            lambda mode: __class__.get_real_imag(mode, params),
            modes,
            is_leaf=lambda x: isinstance(x, dict),
        )

    @staticmethod
    def get_eim_at_native_indices(
        eim_dict: dict,
        params: Float[Array, " n_dim"],
        native_indices: Int[Array, " ..."],
    ) -> Float[Array, " ..."]:
        """Evaluate an EIM expansion at selected native-grid columns only."""
        coefficients = jnp.zeros((eim_dict["n_nodes"], 1))
        for index in range(eim_dict["n_nodes"]):
            coefficients = coefficients.at[index].set(
                eim_dict["predictors"][index](params)
            )
        selected_basis = jnp.take(
            eim_dict["eim_basis"],
            native_indices,
            axis=1,
        )
        return jnp.tensordot(
            coefficients[:, 0],
            selected_basis,
            axes=(0, 0),
        )

    @staticmethod
    def get_eim_coefficients(
        eim_dict: dict,
        params: Float[Array, " n_dim"],
    ) -> Float[Array, " n_nodes"]:
        """Evaluate the parameter-dependent coefficients of an EIM expansion."""
        coefficients = jnp.zeros((eim_dict["n_nodes"], 1))
        for index in range(eim_dict["n_nodes"]):
            coefficients = coefficients.at[index].set(
                eim_dict["predictors"][index](params)
            )
        return coefficients[:, 0]

    def get_spline_brackets(
        self,
        time: Float[Array, " ..."],
    ) -> tuple[
        Int[Array, " ..."],
        Int[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
        Float[Array, " ..."],
    ]:
        """Return native columns and exact natural-cubic interpolation weights."""
        grid = self.data.sur_time
        safe_time = jnp.clip(time, grid[0], grid[-1])
        upper = jnp.clip(jnp.digitize(safe_time, grid), 1, grid.size - 1)
        lower = upper - 1
        interval = grid[upper] - grid[lower]
        distance_left = safe_time - grid[lower]
        distance_right = grid[upper] - safe_time

        value_weight_left = distance_right / interval
        value_weight_right = distance_left / interval
        coefficient_weight_left = (
            distance_right**3 / (6.0 * interval)
            - interval * distance_right / 6.0
        )
        coefficient_weight_right = (
            distance_left**3 / (6.0 * interval)
            - interval * distance_left / 6.0
        )
        return (
            lower,
            upper,
            value_weight_left,
            value_weight_right,
            coefficient_weight_left,
            coefficient_weight_right,
        )

    @staticmethod
    def evaluate_spline_from_brackets(
        values: Float[Array, " n_grid"],
        spline_coefficients: Float[Array, " n_grid"],
        brackets: tuple,
    ) -> Float[Array, " ..."]:
        """Evaluate a represented natural cubic spline at prepared brackets."""
        (
            lower,
            upper,
            value_weight_left,
            value_weight_right,
            coefficient_weight_left,
            coefficient_weight_right,
        ) = brackets
        return (
            value_weight_left * values[lower]
            + value_weight_right * values[upper]
            + coefficient_weight_left * spline_coefficients[lower]
            + coefficient_weight_right * spline_coefficients[upper]
        )

    def get_eim_cubic_at_brackets(
        self,
        eim_dict: dict,
        params: Float[Array, " n_dim"],
        brackets: tuple,
    ) -> Float[Array, " ..."]:
        """Evaluate an EIM expansion through its exact sparse cubic spline."""
        coefficients = self.get_eim_coefficients(eim_dict, params)
        lower, upper = brackets[:2]
        basis_left = jnp.take(eim_dict["eim_basis"], lower, axis=1)
        basis_right = jnp.take(eim_dict["eim_basis"], upper, axis=1)
        spline_left = jnp.take(
            eim_dict["spline_coefficients"], lower, axis=1
        )
        spline_right = jnp.take(
            eim_dict["spline_coefficients"], upper, axis=1
        )
        values_left = jnp.tensordot(coefficients, basis_left, axes=(0, 0))
        values_right = jnp.tensordot(coefficients, basis_right, axes=(0, 0))
        coefficients_left = jnp.tensordot(
            coefficients, spline_left, axes=(0, 0)
        )
        coefficients_right = jnp.tensordot(
            coefficients, spline_right, axes=(0, 0)
        )
        return (
            brackets[2] * values_left
            + brackets[3] * values_right
            + brackets[4] * coefficients_left
            + brackets[5] * coefficients_right
        )

    def get_mode(
        self,
        real: Float[Array, " n_sample"],
        imag: Float[Array, " n_sample"],
        time: Float[Array, " n_time"],
    ) -> Float[Array, " n_sample"]:
        """
        Interpolate real and imaginary mode data to the given time grid.

        Args:
            real (Float[Array, " n_sample"]): Real part of mode.
            imag (Float[Array, " n_sample"]): Imaginary part of mode.
            time (Float[Array, " n_time"]): Time grid.

        Returns:
            Float[Array, " n_sample"]: Complex mode data at requested times.
        """
        return CubicSpline(self.data.sur_time, real)(time) + 1j * CubicSpline(
            self.data.sur_time, imag
        )(time)

    def get_22_mode(
        self,
        time: Float[Array, " n_samples"],
        params: Float[Array, " n_dim"],
    ) -> Float[Array, " n_sample"]:
        """
        Compute the (2,2) mode for the waveform.

        Args:
            time (Float[Array, " n_samples"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            Float[Array, " n_sample"]: Complex (2,2) mode data.
        """
        # 22 mode has weird dict that making a specical function is easier.
        q = self.get_physical_mass_ratio(params)
        # the EIM fits live in (log q, chi_hat, chi_a); the T3 phase wants the raw q
        fit_params = self.get_fit_params(params)[None]
        amp = self.get_eim(self.data.modes[self.mode_22_index]["amp"], fit_params)
        phase = -self.get_eim(self.data.modes[self.mode_22_index]["phase"], fit_params)
        phase = phase + get_T3_phase(q, self.data.sur_time)  # type: ignore
        amp_interp = CubicSpline(self.data.sur_time, amp)(time)
        phase_interp = CubicSpline(self.data.sur_time, phase)(time)
        return amp_interp * jnp.exp(1j * phase_interp)

    def get_waveform_at_native_indices(
        self,
        native_indices: Int[Array, " ..."],
        params: Float[Array, " n_dim"],
        theta: Float = 0.0,
        phi: Float = 0.0,
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        """Return polarizations at selected native surrogate-grid samples."""
        native_indices = jnp.asarray(native_indices, dtype=jnp.int32)
        native_times = self.data.sur_time[native_indices]
        fit_params = self.get_fit_params(params)[None]

        mode_22 = self.data.modes[self.mode_22_index]
        amplitude_22 = self.get_eim_at_native_indices(
            mode_22["amp"], fit_params, native_indices
        )
        phase_22 = -self.get_eim_at_native_indices(
            mode_22["phase"], fit_params, native_indices
        )
        phase_22 += get_T3_phase(
            self.get_physical_mass_ratio(params),
            native_times,
        )
        h_22 = amplitude_22 * jnp.exp(1j * phase_22)

        waveform = jnp.zeros_like(native_times, dtype=jnp.complex64)
        waveform += h_22 * SpinWeightedSphericalHarmonics(-2, 2, 2)(
            theta, phi
        )
        waveform += jnp.conj(h_22) * SpinWeightedSphericalHarmonics(
            -2, 2, -2
        )(theta, phi)

        for index, harmonics in enumerate(self.harmonics):
            mode = self.mode_no22[index]
            real = self.get_eim_at_native_indices(
                mode["real"], fit_params, native_indices
            )
            imag = self.get_eim_at_native_indices(
                mode["imag"], fit_params, native_indices
            )
            complex_mode = real + 1j * imag
            waveform += complex_mode * harmonics(theta, phi)
            waveform += (
                self.negative_mode_prefactor[index]
                * jnp.conj(complex_mode)
                * self.negative_harmonics[index](theta, phi)
            )

        return waveform.real, -waveform.imag

    def get_waveform_geometric_linear(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: Float = 0.0,
        phi: Float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Evaluate the native-grid waveform with sparse linear interpolation.

        Only the two native surrogate samples bracketing each requested time
        are constructed. This avoids reconstructing all modes on the complete
        native grid and preserves the ordinary linear interpolation convention,
        including zero-valued samples outside the surrogate time domain.
        """
        time = jnp.asarray(time)
        surrogate_times = self.data.sur_time
        in_domain = (time >= surrogate_times[0]) & (
            time <= surrogate_times[-1]
        )
        safe_times = jnp.clip(time, surrogate_times[0], surrogate_times[-1])

        upper_indices = jnp.searchsorted(
            surrogate_times,
            safe_times,
            side="right",
        )
        lower_indices = jnp.clip(
            upper_indices - 1,
            0,
            surrogate_times.size - 2,
        )
        upper_indices = lower_indices + 1
        bracket_indices = jnp.stack((lower_indices, upper_indices))

        lower_times = surrogate_times[lower_indices]
        upper_times = surrogate_times[upper_indices]
        fractions = (safe_times - lower_times) / (upper_times - lower_times)
        bracket_plus, bracket_cross = self.get_waveform_at_native_indices(
            bracket_indices,
            params,
            theta=theta,
            phi=phi,
        )
        interpolated_plus = (
            (1.0 - fractions) * bracket_plus[0]
            + fractions * bracket_plus[1]
        )
        interpolated_cross = (
            (1.0 - fractions) * bracket_cross[0]
            + fractions * bracket_cross[1]
        )
        return (
            jnp.where(in_domain, interpolated_plus, 0.0),
            jnp.where(in_domain, interpolated_cross, 0.0),
        )

    def get_waveform_geometric_dense(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: Float = 0.0,
        phi: Float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Compute the waveform by reconstructing every native-grid sample.

        This is the original dense reference implementation. It remains
        available for validation; normal calls use the mathematically
        equivalent sparse cubic evaluator in :meth:`get_waveform_geometric`.

        Args:
            time (Float[Array, " n_sample"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.
            theta (Float, optional): Polar angle. Defaults to 0.0.
            phi (Float, optional): Azimuthal angle. Defaults to 0.0.

        Returns:
            tuple: Plus and cross polarizations of the waveform.
        """
        coeff = jnp.stack(
            jnp.array(
                self.get_multi_real_imag(self.mode_no22, self.get_fit_params(params))
            )
        )
        modes = eqx.filter_vmap(self.get_mode, in_axes=(0, 0, None))(
            coeff[:, 0], coeff[:, 1], time
        )

        waveform = jnp.zeros_like(time, dtype=jnp.complex64)

        h22 = self.get_22_mode(time, params)
        waveform += h22 * SpinWeightedSphericalHarmonics(-2, 2, 2)(theta, phi)
        waveform += jnp.conj(h22) * SpinWeightedSphericalHarmonics(-2, 2, -2)(
            theta, phi
        )

        for i, harmonics in enumerate(self.harmonics):
            waveform += modes[i] * harmonics(theta, phi)
            waveform += (
                self.negative_mode_prefactor[i]
                * jnp.conj(modes[i])
                * self.negative_harmonics[i](theta, phi)
            )

        # Mask to ensure output is zero outside the model's time range
        mask = (time >= self.data.sur_time[0]) * (time <= self.data.sur_time[-1])
        hp = jnp.where(mask, waveform.real, 0.0)
        hc = jnp.where(mask, -waveform.imag, 0.0)
        return hp, hc

    def get_waveform_geometric(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: Float = 0.0,
        phi: Float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Compute polarizations with exact sparse natural-cubic interpolation.

        The natural-cubic coefficients of every fixed EIM basis are prepared at
        model initialization. Each call therefore evaluates only the two native
        columns bracketing each requested sample, instead of reconstructing all
        native-grid modes and solving a spline system for every waveform.
        """
        if not self.sparse_cubic_enabled:
            return self.get_waveform_geometric_dense(time, params, theta, phi)

        time = jnp.asarray(time)
        brackets = self.get_spline_brackets(time)
        fit_params = self.get_fit_params(params)[None]

        mode_22 = self.data.modes[self.mode_22_index]
        amplitude_22 = self.get_eim_cubic_at_brackets(
            mode_22["amp"], fit_params, brackets
        )
        phase_22 = -self.get_eim_cubic_at_brackets(
            mode_22["phase"], fit_params, brackets
        )

        # T3(q, t) is eta(q)^(-3/8) times a q-independent time series,
        # so its spline has the same separable scaling.
        q = self.get_physical_mass_ratio(params)
        eta = q / (1.0 + q) ** 2
        t3_scale = (eta / 0.25) ** (-3.0 / 8.0)
        phase_22 += t3_scale * self.evaluate_spline_from_brackets(
            self.t3_phase_reference,
            self.t3_phase_reference_coefficients,
            brackets,
        )
        h_22 = amplitude_22 * jnp.exp(1j * phase_22)

        waveform = jnp.zeros_like(time, dtype=jnp.complex64)
        waveform += h_22 * SpinWeightedSphericalHarmonics(-2, 2, 2)(
            theta, phi
        )
        waveform += jnp.conj(h_22) * SpinWeightedSphericalHarmonics(
            -2, 2, -2
        )(theta, phi)

        for index, harmonics in enumerate(self.harmonics):
            mode = self.mode_no22[index]
            real = self.get_eim_cubic_at_brackets(
                mode["real"], fit_params, brackets
            )
            imag = self.get_eim_cubic_at_brackets(
                mode["imag"], fit_params, brackets
            )
            complex_mode = real + 1j * imag
            waveform += complex_mode * harmonics(theta, phi)
            waveform += (
                self.negative_mode_prefactor[index]
                * jnp.conj(complex_mode)
                * self.negative_harmonics[index](theta, phi)
            )

        in_domain = (time >= self.data.sur_time[0]) & (
            time <= self.data.sur_time[-1]
        )
        return (
            jnp.where(in_domain, waveform.real, 0.0),
            jnp.where(in_domain, -waveform.imag, 0.0),
        )
