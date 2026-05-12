import torch
from dataclasses import dataclass
from einops import rearrange
from popcornn.tools import Images, SamplesCache, wrap_positions
from popcornn.potentials.base_potential import BasePotential, PotentialOutput
from typing import Callable, Any
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
            Whether the optimization loop should run ``ts_search`` each
            iteration and populate ``ts_time`` / ``ts_energy`` /
            ``ts_force`` / ``ts_force_mag``. Set False to skip the
            sign-change + fresh-eval step entirely.
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
    ):
        """
        Locate the predicted transition state on the current path.

        Picks the TS as the interior sign change of ``dE/dt`` on the
        cached quadrature samples, linearly interpolates ``t`` at the
        zero crossing, and then re-evaluates the path at that time to
        get model-truth ``ts_energy`` / ``ts_force`` — argmax-E on the
        same sample mesh overshoots the saddle force by ~6× because the
        cached F is evaluated at the sample times, not at the true
        sign-change time.

        Falls back to interior ``argmax(E)`` when no interior sign
        change exists (under-resolved paths, monotone profiles). The
        endpoints are excluded from both the bracket search and the
        argmax: paths are parameterized so the reactant/product minima
        sit at ``t=0`` and ``t=1`` where ``dE/dt`` is also ~0, and
        picking either of those is never the right call.

        Parameters
        ----------
        samples : SamplesCache
            Per-quadrature-point ``(time, energies, dE/dt)`` from
            ``PathIntegrator.integrate_path(save_samples=True)``.

        Notes
        -----
        Sets ``self.ts_time``, ``self.ts_energy``, ``self.ts_force``,
        and ``self.ts_force_mag`` (per-atom fmax = ``max_i ‖F_i‖_2``
        for atomistic systems; vector L2 norm for toy potentials with
        no ``n_atoms`` set).
        """
        time = samples.time
        energies = samples.energies.flatten()
        dEdt = samples.dEdt
        N = energies.shape[0]

        # Interior sign-change brackets. ``dE/dt`` touches zero at the
        # endpoints (reactant/product minima), so a global sign search
        # would falsely bracket the first/last segment; restrict to
        # interior segments [i, i+1] with i ∈ [1, N-3].
        #
        # Use ``<= 0`` rather than ``< 0`` so a sample that lands exactly
        # on the saddle (``dE/dt = 0``) is still bracketed; the linear
        # interp then degenerates to ``t = t_zero`` cleanly. Exclude the
        # both-zero case to avoid a 0/0 in the interp formula.
        ts_time = None
        if N >= 4:
            d_int = dEdt[1:N-1]
            both_zero = (d_int[:-1] == 0) & (d_int[1:] == 0)
            sign_change = ((d_int[:-1] * d_int[1:]) <= 0) & ~both_zero
            if sign_change.any():
                # Among all interior brackets, pick the one with the
                # highest mean energy — that's the real saddle when the
                # path has shoulders or multiple basins.
                brackets = sign_change.nonzero(as_tuple=False).flatten()
                e_int = energies[1:N-1]
                e_pair_mean = 0.5 * (e_int[brackets] + e_int[brackets + 1])
                best = brackets[int(torch.argmax(e_pair_mean).item())]
                i = int(best.item()) + 1  # map back to absolute index
                # Linear interp at dE/dt = 0:
                #   t_TS = t_i - dEdt_i * (t_{i+1} - t_i) / (dEdt_{i+1} - dEdt_i)
                t_i, t_j = time[i], time[i + 1]
                d_i, d_j = dEdt[i], dEdt[i + 1]
                ts_time = t_i - d_i * (t_j - t_i) / (d_j - d_i)

        if ts_time is None:
            # No interior sign change — fall back to interior argmax E.
            interior = energies[1:N-1]
            pick = int(torch.argmax(interior).item()) + 1
            ts_time = time[pick]

        self.ts_time = ts_time.to(device=self.device, dtype=self.dtype)

        # Always re-evaluate at the interpolated time to get model-truth
        # E and F: the cached F is per-quadrature-sample, not aligned to
        # the sign-change time, and the sample-time fmax overshoots the
        # true saddle by enough (~6×) to dominate any practical fmax
        # tolerance.
        ts_output = self.forward(
            self.ts_time.reshape(1),
            return_velocities=False,
            return_energies=True,
            return_forces=True,
        )
        self.ts_energy = ts_output.energies
        self.ts_force = ts_output.forces
        self.ts_force_mag = self._ts_fmax(self.ts_force)

    def _ts_fmax(self, force: torch.Tensor) -> torch.Tensor:
        """Per-atom fmax for atomistic systems; vector L2 norm otherwise.

        Atomistic force is laid out as ``[..., 3 * n_atoms]``; reshape
        to ``[..., n_atoms, 3]``, take the L2 norm over the xyz axis,
        and reduce to the per-atom maximum. Toy potentials (no
        ``n_atoms`` on the potential, e.g. Muller-Brown) get a plain
        vector L2 norm.
        """
        f = force.reshape(-1)
        n_atoms = getattr(self.potential, 'n_atoms', None) if self.potential is not None else None
        if n_atoms is None:
            return torch.linalg.norm(f)
        return torch.linalg.norm(f.reshape(n_atoms, 3), dim=-1).max()