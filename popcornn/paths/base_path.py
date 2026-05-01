import torch
import numpy as np
import scipy as sp
from dataclasses import dataclass
from einops import rearrange
from popcornn.tools import Images, SamplesCache, wrap_positions
from popcornn.potentials.base_potential import BasePotential, PotentialOutput
from typing import Callable, Any, Literal
from ase import Atoms
from ase.io import read


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    time : torch.Tensor
        The time at which the path was evaluated.
    positions : torch.Tensor
        The coordinates along the path.
    velocities : torch.Tensor, optional
        The velocities along the path (default is None).
    energies : torch.Tensor
        The potential energy along the path.
    forces : torch.Tensor, optional
        The force along the path (default is None).
    """
    time: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor = None
    energies: torch.Tensor = None
    energies_decomposed: torch.Tensor = None
    forces: torch.Tensor = None
    forces_decomposed: torch.Tensor = None

    def __len__(self):
        """
        Return the number of images.
        """
        return len(self.positions)


class BasePath(torch.nn.Module):
    """
    Base class for differentiable path representations.

    A path is a smooth mapping ``t -> x(t)`` from ``t in [0, 1]`` to a
    configuration, with ``x(0)`` pinned at the reactant and ``x(1)`` at
    the product. Subclasses implement ``get_positions``; this base
    class wires up

    - velocity computation via autograd (``calculate_velocities``),
    - periodic-cell wrapping (when ``images.pbc`` is set),
    - fixed-atom masking,
    - the ``forward`` interface that popcornn's optimizer drives,
    - the input/output reshaping that lets the integrator pass either
      ``[B, T]`` or ``[B, C, T]`` time tensors.

    Subclasses must populate trainable ``torch.nn.Parameter``\\s.
    """
    initial_position: torch.Tensor
    final_position: torch.Tensor

    def __init__(
            self,
            images: Images,
            device: torch.device,
            dtype: torch.dtype,
            find_ts: bool = True,
        ) -> None:
        """
        Initialize the path.

        Parameters
        ----------
        images : Images
            Processed images. The first frame's positions become
            ``self.initial_position``, the last frame's become
            ``self.final_position``. Periodic-cell info, fixed-atom
            masks, and tags are pulled from here.
        device : torch.device
        dtype : torch.dtype
        find_ts : bool, default=True
            Whether the optimization loop should attempt
            transition-state extraction. Currently a hint only — the
            extraction routine itself is paused under the torchpathint
            migration.
        """
        super().__init__()
        self.neval = 0
        self.find_ts = find_ts
        self.potential = None
        self.initial_position = images.positions[0]
        self.final_position = images.positions[-1]
        self._inp_reshaped = None
        if images.pbc is not None and images.pbc.any():
            def transform(positions, **kwargs):
                return wrap_positions(positions, images.cell, images.pbc, **kwargs)
            self.transform = transform
        else:
            self.transform = None
        self.fix_positions = images.fix_positions
        self.device = device
        self.dtype = dtype
        self.t_init = torch.tensor(
            [[0]], dtype=self.dtype, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=self.dtype, device=self.device
        )
        self.ts_time = None
        self.ts_region = None

    def set_potential(
            self,
            potential: BasePotential,
    ) -> None:
        """
        Attach a potential to evaluate energies/forces along the path.

        Each optimization leg constructs its own potential and calls
        this; the path holds onto the most recently set one.
        """
        self.potential = potential

    def get_positions(
            self,
            time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the geometric path at ``time``. Subclasses must override.

        Parameters
        ----------
        time : torch.Tensor
            Times in [0, 1]; shape ``[N, 1]``.

        Returns
        -------
        torch.Tensor
            Positions of shape ``[N, D]``.
        """
        raise NotImplementedError()


    def calculate_velocities(self, t, create_graph=True):
        """
        Compute path velocities via autograd.

        Differentiates ``get_positions`` with respect to ``t``. The
        ``torch.sum`` over the leading axis is a vectorization trick:
        summing collapses the per-time outputs so a single jacobian
        call returns ``dx_i/dt_i`` for every i in one pass.
        """
        return torch.autograd.functional.jacobian(
            lambda t: torch.sum(self.get_positions(t), axis=0),
            t,
            create_graph=create_graph,
            vectorize=True
        ).transpose(0, 1)[:, :, 0]
    
    def _check_output(
            self,
            potential_output,
            return_energies: bool,
            return_energies_decomposed: bool,
            return_forces: bool,
            return_forces_decomposed: bool,
        ):
        """
        Raise if the attached potential can't produce a requested field.

        Toy potentials skip force-decomposition; some MLIP wrappers
        skip energy-decomposition. Catch the missing field at the
        first call site rather than later in the loss layer.
        """
        name = type(self.potential).__name__
        if return_energies and potential_output.energies is None:
            raise ValueError(f"Potential {name} cannot calculate energies")
        if return_energies_decomposed and potential_output.energies_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate energies_decomposed")
        if return_forces and potential_output.forces is None:
            raise ValueError(f"Potential {name} cannot calculate forces")
        if return_forces_decomposed and potential_output.forces_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate forces_decomposed")
    
    def forward(
            self,
            time : torch.Tensor = None,
            return_velocities: bool = False,
            return_energies: bool = False,
            return_energies_decomposed: bool = False,
            return_forces: bool = False,
            return_forces_decomposed: bool = False,
    ) -> PathOutput:
        """
        Evaluate the path (and optionally the potential) at given times.

        Parameters
        ----------
        time : torch.Tensor, optional
            Times in [0, 1]. Accepts shape ``[B]``, ``[B, T]`` or
            ``[B, C, T]``; the input shape is restored on the output.
            ``None`` defaults to 101 points linearly spaced over
            ``[t_init, t_final]``.
        return_velocities : bool, default=False
        return_energies : bool, default=False
        return_energies_decomposed : bool, default=False
        return_forces : bool, default=False
        return_forces_decomposed : bool, default=False
            Each toggles whether to populate the corresponding field
            on the returned ``PathOutput``. Disabled-by-default to
            avoid paying for autograd / potential evaluations the
            caller doesn't need.

        Returns
        -------
        PathOutput
            With ``time`` and ``positions`` always populated; other
            fields populated when the matching flag is set.
        """
        time = self._reshape_in(time)

        self.neval += time.numel()

        positions = self.get_positions(time)
        if self.transform is not None:
            positions = self.transform(positions)
        if return_energies or return_energies_decomposed or return_forces or return_forces_decomposed:
            assert self.potential is not None, "Potential must be set by \'set_potential\' before calling \'forward\'"
            potential_output = self.potential(positions) 
            self._check_output(
                potential_output,
                return_energies=return_energies,
                return_energies_decomposed=return_energies_decomposed,
                return_forces=return_forces,
                return_forces_decomposed=return_forces_decomposed
            )
        else:
            potential_output = PotentialOutput()

        if return_velocities:
            velocities = self.calculate_velocities(time)
        else:
            velocities = None

        return PathOutput(
            time=self._reshape_out(time),
            positions=self._reshape_out(positions),
            velocities=self._reshape_out(velocities),
            energies=self._reshape_out(potential_output.energies),
            energies_decomposed=self._reshape_out(potential_output.energies_decomposed),
            forces=self._reshape_out(potential_output.forces),
            forces_decomposed=self._reshape_out(potential_output.forces_decomposed),
        )
    

    def _reshape_in(self, time):
        """
        Flatten an arbitrary-shape time tensor to ``[N, T]`` for batched
        evaluation. ``_reshape_out`` undoes the flatten on the way out.

        The integrator passes ``[B, C, T]`` (batch x quadrature-channel
        x time) but downstream layers want a flat batch dim, so cache
        the input shape, rearrange, and remember to invert.
        """
        if time is None:
            time = torch.linspace(self.t_init.item(), self.t_final.item(), 101, device=self.device, dtype=self.dtype)
        
        if len(time.shape) == 3:
            self._inp_reshaped = True
            self._inp_shape = time.shape
            time = rearrange(time, 'b c t -> (b c) t')
        elif len(time.shape) == 2:
            self._inp_reshaped = False
            B, C, = None, None
        elif len(time.shape) == 1:
            self._inp_reshaped = False
            B, C, = None, None
            time = torch.unsqueeze(time, -1)
        else:
            raise ValueError(f"Input path time must be of dimensions [B, C, T], [B, T], or [B] where T is the time dimsion and is generally 1: instead got {time.shape}")

        return time


    def _reshape_out(self, result):
        """Restore the original shape stashed by ``_reshape_in``."""
        if self._inp_reshaped is None:
            raise RuntimeError("Must call _reshape_in() before _reshape_out()")
        if self._inp_reshaped and result is not None:
            B, C, _ = self._inp_shape
            return rearrange(result, '(b c) d -> b c d', b=B, c=C)
        return result

    
    def ts_search(
        self,
        samples: SamplesCache,
        *,
        evaluate_ts: bool = False,
        criterion: Literal['energy', 'force', 'combined'] = 'combined',
        topk_E: int = 7,
        idx_shift: int = 4,
        N_interp: int = 10000,
    ):
        """
        Locate the predicted transition state on the current path.

        Operates on samples already collected by ``PathIntegrator`` —
        no extra path-forward calls are needed (apart from the optional
        single re-evaluation at the predicted TS time when
        ``evaluate_ts=True``).

        Algorithm
        ---------
        1. Pick the ``topk_E`` highest-energy quadrature points.
        2. For each, build a window of ``±idx_shift`` neighbouring
           samples and fit cubic interpolators of ``E(t)`` and ``F(t)``.
        3. Oversample each window at ``N_interp`` points; keep the top
           candidates by interp E (max) and interp |F| (min) inside it.
        4. Concatenate candidates across windows.
        5. Choose the final TS by ``criterion``:

           * ``'energy'``: ``argmax(E)`` over the candidates.
           * ``'force'``: ``argmin(|F|)`` over the candidates.
           * ``'combined'``: iteratively keep entries within 2σ of the
             max E (up to 3 passes), then ``argmin(|F|)`` — the
             saddle-point criterion the original popcornn used.

        Parameters
        ----------
        samples : SamplesCache
            Per-quadrature-point ``(time, energies, forces)`` from
            ``PathIntegrator.integrate_path(save_samples=True)``.
        evaluate_ts : bool, default=False
            If True, re-evaluate the path at the predicted TS time to
            replace the interpolator-derived ``ts_energy`` / ``ts_force``
            with model-truth values. Costs one extra path forward.
        criterion : {'energy', 'force', 'combined'}, default='combined'
            Final-pick rule. See Algorithm step 5.
        topk_E : int, default=7
            Number of high-energy quadrature points to seed windows around.
        idx_shift : int, default=4
            Half-width (in quadrature samples) of each interpolation window.
        N_interp : int, default=10000
            Oversampling resolution for the cubic interpolators.

        Notes
        -----
        Sets ``self.ts_time``, ``self.ts_energy``, ``self.ts_force``,
        ``self.ts_force_mag``, and ``self.ts_region`` (a small time
        window around ``ts_time`` used by the TS-region loss).
        """
        time = samples.time
        energies = samples.energies.flatten()
        forces = samples.forces
        N = energies.shape[0]
        if N < 2 * idx_shift + 2:
            raise ValueError(
                f"ts_search needs at least {2 * idx_shift + 2} samples to "
                f"build a cubic interpolation window of half-width "
                f"{idx_shift}; got {N}. Use a finer integrator method or "
                f"loosen idx_shift."
            )

        # Top-K energy seeds. Clamp window endpoints into [0, N].
        _, ts_idxs = torch.topk(energies, min(N, topk_E))
        idxs_min = torch.clamp(ts_idxs - idx_shift, min=0)
        idxs_max = torch.clamp(ts_idxs + idx_shift + 1, max=N)
        idx_ranges = {
            (idxs_min[i].item(), idxs_max[i].item())
            for i in range(len(ts_idxs))
        }

        interp_Es: list[np.ndarray] = []
        interp_Fs: list[np.ndarray] = []
        interp_magFs: list[np.ndarray] = []
        interp_times: list[np.ndarray] = []
        top_N = max(N_interp // 200, 1)
        ts_time_scale = 0.0
        for imin, imax in idx_ranges:
            t_window = time[imin:imax].detach().cpu().numpy()
            E_window = energies[imin:imax].detach().cpu().numpy()
            F_window = forces[imin:imax].detach().cpu().numpy()
            E_interp_fn = sp.interpolate.interp1d(t_window, E_window, kind='cubic')
            F_interp_fn = sp.interpolate.interp1d(t_window, F_window, axis=0, kind='cubic')
            t_dense = np.linspace(t_window[0] + 1e-12, t_window[-1] - 1e-12, N_interp)
            E_dense = E_interp_fn(t_dense)
            F_dense = F_interp_fn(t_dense)
            magF_dense = np.linalg.norm(F_dense, ord=2, axis=-1).flatten()

            E_top = np.argpartition(E_dense, -top_N)[-top_N:]
            F_top = np.argpartition(magF_dense, top_N)[:top_N]
            keep = np.unique(np.concatenate([E_top, F_top]))
            interp_Es.append(E_dense[keep])
            interp_Fs.append(F_dense[keep])
            interp_magFs.append(magF_dense[keep])
            interp_times.append(t_dense[keep])
            ts_time_scale = max(ts_time_scale, float(t_window[-1] - t_window[0]))

        E_cand = np.concatenate(interp_Es, axis=0)
        F_cand = np.concatenate(interp_Fs, axis=0)
        magF_cand = np.concatenate(interp_magFs, axis=0)
        t_cand = np.concatenate(interp_times, axis=0)

        if criterion == 'energy':
            pick = int(np.argmax(E_cand))
        elif criterion == 'force':
            pick = int(np.argmin(magF_cand))
        elif criterion == 'combined':
            # Iteratively trim entries more than 2σ below the running
            # max-E so the final argmin |F| is taken among high-energy
            # candidates only — the saddle-point criterion.
            mask = np.ones_like(E_cand, dtype=bool)
            for _ in range(3):
                live_E = E_cand[mask]
                if len(live_E) < 2:
                    break
                threshold = live_E.max() - 2 * live_E.std()
                trim = E_cand >= threshold
                if trim.sum() < 1:
                    break
                mask &= trim
            pick = int(np.argmin(np.where(mask, magF_cand, np.inf)))
        else:
            raise ValueError(
                f"Unknown criterion {criterion!r}; "
                f"expected one of 'energy', 'force', 'combined'."
            )

        self.ts_time = torch.tensor(t_cand[pick], device=self.device, dtype=self.dtype)
        self.ts_energy = torch.tensor(E_cand[pick], device=self.device, dtype=self.dtype)
        self.ts_force = torch.tensor(F_cand[pick], device=self.device, dtype=self.dtype)
        self.ts_force_mag = torch.tensor(magF_cand[pick], device=self.device, dtype=self.dtype)

        if evaluate_ts:
            ts_output = self.forward(
                torch.tensor([self.ts_time], device=self.device, dtype=self.dtype),
                return_velocities=True,
                return_energies=True,
                return_forces=True,
            )
            self.ts_energy = ts_output.energies
            self.ts_force = ts_output.forces
            self.ts_force_mag = torch.linalg.norm(self.ts_force, dim=-1)

        self.ts_region = torch.linspace(
            self.ts_time - ts_time_scale / idx_shift,
            self.ts_time + ts_time_scale / idx_shift,
            11,
            device=self.device,
        )