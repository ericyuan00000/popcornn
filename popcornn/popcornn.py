import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from copy import deepcopy
import torch
from typing import Any
import time as time
from tqdm import tqdm
from ase import Atoms
from dataclasses import dataclass
import json

from popcornn.paths import get_path
from popcornn.optimization import initialize_path
from popcornn.optimization import PathOptimizer
from popcornn.tools import process_images, output_to_atoms
from popcornn.tools import PathIntegrator
from popcornn.potentials import get_potential


# Sparse iteration set for the per-leg progress printout. Dense at the
# start (where most of the descent happens), then every 50 after 250.
_PRINT_ITERS = (0, 5, 10, 25, 50, 75, 100, 150, 200, 250)


def _should_print(it: int, last_it: int) -> bool:
    return it in _PRINT_ITERS or it == last_it or (it > 250 and it % 50 == 0)


class _LegLogger:
    """Per-leg sparse-table logger + wall-time tracker.

    Streams a header at leg start, sparse per-iter rows via
    ``tqdm.write`` (so the rows interleave with the iteration bar),
    a convergence-trigger announcement, and a leg-end summary to
    stdout. When ``metrics_log_path`` is provided, also writes one
    JSONL row per iteration with the full scalar metric set —
    flushed each row so a killed run leaves a partial-but-valid
    file.
    """

    def __init__(self, leg_idx, integrand_terms, lr, threshold,
                 n_iter, n_params, metrics_log_path=None):
        self.leg_idx = leg_idx
        self.n_iter = n_iter
        terms_str = (
            " + ".join(f"{t.name}×{t.scale:g}" for t in integrand_terms)
            or "<none>"
        )
        thr = "—" if threshold is None else f"{threshold:.1e}"
        print(
            f"── leg {leg_idx} ──  integrand: {terms_str}   "
            f"lr={lr:.1e}   threshold={thr}   "
            f"n_iter={n_iter}   n_params={n_params}"
        )
        print(
            f"  {'iter':>6s}  {'loss':>12s}  "
            f"{'|g|_inf':>12s}  {'|g|_2':>12s}  {'step_s':>8s}"
        )
        self._t_start = time.perf_counter()

        self._metrics_fh = None
        if metrics_log_path is not None:
            parent = os.path.dirname(metrics_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._metrics_fh = open(metrics_log_path, 'w')

    def row(self, it, loss, g_inf, g_2, step_s):
        loss_str = "—" if loss is None else f"{loss:12.4e}"
        tqdm.write(
            f"  {it:>6d}  {loss_str:>12s}  "
            f"{g_inf:12.4e}  {g_2:12.4e}  {step_s:8.4f}"
        )

    def metrics(self, **fields):
        """Append one JSONL row with scalar metrics. No-op when no
        ``metrics_log_path`` was configured."""
        if self._metrics_fh is None:
            return
        self._metrics_fh.write(json.dumps(fields) + "\n")
        self._metrics_fh.flush()

    def wall_s(self):
        return time.perf_counter() - self._t_start

    def converged(self, it, g_2, threshold, patience):
        tqdm.write(
            f"converged at iter {it}  "
            f"(|g|_2={g_2:.3e} < threshold={threshold:.1e} "
            f"for patience={patience})"
        )

    def end(self, iters_done):
        wall = self.wall_s()
        ms_iter = 1000 * wall / max(1, iters_done)
        print(
            f"leg {self.leg_idx} done  iters={iters_done}  "
            f"wall={wall:.1f}s  ms/iter={ms_iter:.1f}"
        )

    def close(self):
        if self._metrics_fh is not None:
            self._metrics_fh.close()
            self._metrics_fh = None


class Popcornn:
    """
    High-level driver for popcornn reaction-path optimization.

    Wraps the path representation, image processing, and
    multi-leg optimization loop. The typical lifecycle is

    1. Construct with the reactant/product/intermediate images and a
       ``path_params`` dict that picks the path representation.
    2. Call ``optimize_path`` with one or more leg-config dicts.
    3. Receive the optimized path frames and (when TS extraction is
       active) a single predicted transition-state frame.
    """
    def __init__(
            self, 
            images: list[Atoms],
            unwrap_positions: bool = True,
            path_params: dict[str, Any] = {},
            num_record_points: int = 101,
            output_dir: str | None = None,
            device: str | None = None,
            dtype: str = "float32",
            seed: int | None = 0,
    ):
        """
        Initialize the Popcornn class.

        Args:
            images (list[Atoms]): List of ASE Atoms objects representing the images.
            unwrap_positions (bool): Whether to unwrap the positions of the images. Default is True.
            path_params (dict[str, Any]): Parameters for the path prediction method.
            num_record_points (int): Number of points to record along the path when returning and saving the optimized path.
            output_dir (str | None): Directory to save the output files. If None, no files will be saved.
            device (str | None): Device to use for optimization. If None, will use 'cuda' if available, otherwise 'cpu'.
            dtype (str): Data type to use for optimization. Can be 'float32' or 'float64'.
            seed (int | None): Random seed for reproducibility. If None, no seed is set.
        """
        # Set device
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        self.device = device

        # Set dtype
        if dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {dtype}. Use 'float32' or 'float64'.")

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # Process images
        self.images = process_images(images, unwrap_positions=unwrap_positions, device=self.device, dtype=self.dtype)

        # Get path prediction method
        self.path = get_path(images=self.images, **path_params, device=self.device, dtype=self.dtype)

        # Randomly initialize the path, otherwise a straight line
        if len(images) > 2:
            self.path = initialize_path(
                path=self.path, 
                times=torch.linspace(self.path.t_init.item(), self.path.t_final.item(), len(self.images), device=self.device, dtype=self.dtype),
                init_points=self.images.positions,
            )

        # Create output directories
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.num_record_points = num_record_points

    
    def optimize_path(
            self,
            *optimization_params: list[dict],
            output_ase_atoms: bool = True,
            metrics_log_path: str | None = None,
    ):
        """
        Run a chain of optimization legs and return the final path.

        Each entry of ``optimization_params`` is one leg. The path's
        trainable parameters persist across legs, so a typical pattern is
        a cheap ``repel`` + ``geodesic`` clash-resolution leg followed by
        an MLIP-driven leg targeting the transition state.

        Parameters
        ----------
        *optimization_params : dict
            One dict per leg. Recognized keys:

            ``potential_params``
                Forwarded to ``get_potential``.
            ``integrator_params``
                Forwarded to ``PathIntegrator``.
            ``optimizer_params``
                Forwarded to ``PathOptimizer``.
            ``num_optimizer_iterations``
                Hard cap on Adam steps for this leg.
        output_ase_atoms : bool, default=True
            If True and the input was ASE ``Atoms``, return ``Atoms``
            objects rather than raw ``PathOutput`` tensors.
        metrics_log_path : str, optional
            Directory; each leg writes one JSONL file
            ``<metrics_log_path>/opt_{i}.jsonl`` with one scalar
            metrics row per iteration (iter, loss, grad_norm_inf,
            grad_norm_2, lr, step_s, wall_s, converged). When
            ``output_dir`` is set on the ``Popcornn`` constructor and
            this kwarg is left as ``None``, defaults to
            ``<output_dir>/metrics/`` so any run that already saves
            the heavy per-iter JSON dump also gets the lightweight
            scalar log next to it.

        Returns
        -------
        images : list[ase.Atoms] or PathOutput
            ``num_record_points`` frames sampled along the optimized path.
        ts_image : ase.Atoms or PathOutput or None
            Predicted transition state as a single frame, or ``None``
            when the optimizer ran with ``find_ts=False``.
        """
        # Optimize the path. When output_dir is set but the caller didn't
        # specify a metrics path, default to <output_dir>/metrics so the
        # lightweight scalar log lands next to the heavy per-iter JSONs.
        if metrics_log_path is None and self.output_dir is not None:
            metrics_log_path = os.path.join(self.output_dir, 'metrics')
        if metrics_log_path is not None:
            os.makedirs(metrics_log_path, exist_ok=True)
        for i, params in enumerate(optimization_params):
            if self.output_dir is not None:
                output_dir = f"{self.output_dir}/opt_{i}"
            else:
                output_dir = None
            if metrics_log_path is not None:
                leg_metrics_path = os.path.join(metrics_log_path, f"opt_{i}.jsonl")
            else:
                leg_metrics_path = None

            self._optimize(
                **params,
                output_dir=output_dir,
                output_ase_atoms=output_ase_atoms,
                leg_idx=i,
                metrics_log_path=leg_metrics_path,
            )

        # Evaluate points along the optimized path and return
        time = torch.linspace(self.path.t_init.item(), self.path.t_final.item(), self.num_record_points, device=self.device, dtype=self.dtype)
        path_output = self.path(time, return_velocities=True, return_energies=True, return_forces=True)
        # ts_time stays None when the optimizer ran with find_ts=False; in
        # that case there's no predicted TS and ts_output is None.
        if self.path.ts_time is not None:
            ts_time = torch.tensor([self.path.ts_time], device=self.device, dtype=self.dtype)
            ts_output = self.path(ts_time, return_velocities=True, return_energies=True, return_forces=True)
        else:
            ts_output = None
        if issubclass(self.images.image_type, Atoms) and output_ase_atoms:
            images = output_to_atoms(path_output, self.images)
            ts_image = output_to_atoms(ts_output, self.images)[0] if ts_output is not None else None
            return images, ts_image
        else:
            return path_output, ts_output

    def _optimize(
            self,
            potential_params: dict[str, Any] = {},
            integrator_params: dict[str, Any] = {},
            optimizer_params: dict[str, Any] = {},
            num_optimizer_iterations: int = 1000,
            output_dir: str | None = None,
            output_ase_atoms: bool = True,
            leg_idx: int = 0,
            metrics_log_path: str | None = None,
    ):
        """
        Run a single optimization leg.

        Builds the potential, integrator, and optimizer for this leg,
        then steps Adam until either the convergence trigger fires or
        ``num_optimizer_iterations`` is reached.

        Parameters
        ----------
        potential_params : dict
            Forwarded to ``get_potential``. ``name`` is required.
        integrator_params : dict
            Forwarded to ``PathIntegrator``.
        optimizer_params : dict
            Forwarded to ``PathOptimizer``. ``threshold`` controls the
            convergence trigger; see ``docs/convergence.md``.
        num_optimizer_iterations : int, default=1000
            Iteration cap.
        output_dir : str, optional
            If set, dump per-iteration JSON state under
            ``{output_dir}/logs/output_<i>.json``.
        output_ase_atoms : bool, default=True
            Reserved; kept for parity with ``optimize_path``. Logging
            here always uses tensor form.
        leg_idx : int, default=0
            Index of this leg in the parent ``optimize_path`` chain;
            used only for the stdout progress header.
        metrics_log_path : str, optional
            If set, write one JSONL row per iteration to this exact
            file path with scalar metrics (iter, loss, grad_norm_inf,
            grad_norm_2, lr, step_s, wall_s, converged). Flushed after
            each row so a killed run leaves a partial-but-valid file.
        """
        # Create output directories
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Get potential energy function
        potential = get_potential(images=self.images, **potential_params, device=self.device, dtype=self.dtype)
        self.path.set_potential(potential)

        # Path optimization tools
        integrator = PathIntegrator(**integrator_params, device=self.device, dtype=self.dtype)

        # Gradient descent path optimizer
        optimizer = PathOptimizer(path=self.path, **optimizer_params, device=self.device, dtype=self.dtype)

        # Sample harvesting is the per-iter input to ts_search; only enable
        # it when the optimizer is actually going to consume the result.
        integrator.save_samples = bool(optimizer.find_ts)
        # The per-iter big JSON dump (gated on output_dir) reads .t and .y
        # off the returned IntegralOutput; ask torchpathint to populate them.
        if output_dir is not None:
            integrator.full_output = True

        # Create output directories
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
        
        # Per-leg progress logger: header now, sparse rows during the loop,
        # convergence/leg-end summary on exit. Optional JSONL written
        # via the same logger when metrics_log_path is set.
        logger = _LegLogger(
            leg_idx=leg_idx,
            integrand_terms=integrator._terms,
            lr=optimizer.optimizer.param_groups[0]['lr'],
            threshold=optimizer.threshold,
            n_iter=num_optimizer_iterations,
            n_params=sum(p.numel() for p in self.path.parameters()),
            metrics_log_path=metrics_log_path,
        )

        last_iter = num_optimizer_iterations - 1
        iters_done = 0

        # Optimize the path
        for optim_idx in tqdm(range(num_optimizer_iterations), leave=False):
            lr = optimizer.optimizer.param_groups[0]['lr']
            t_step = time.perf_counter()
            try:
                integral_output = optimizer.optimization_step(self.path, integrator)
            except ValueError as e:
                print("ValueError", e)
                raise e
            step_s = time.perf_counter() - t_step
            iters_done = optim_idx + 1

            loss_attr = getattr(integral_output, 'loss', None)
            loss_v = float(loss_attr[0].item()) if loss_attr is not None else None
            g_inf = integral_output.grad_norm.item()
            g_2 = integral_output.grad_norm_2.item()

            if _should_print(optim_idx, last_iter):
                logger.row(optim_idx, loss_v, g_inf, g_2, step_s)

            logger.metrics(
                iter=optim_idx,
                loss=loss_v,
                grad_norm_inf=g_inf,
                grad_norm_2=g_2,
                lr=lr,
                step_s=step_s,
                wall_s=logger.wall_s(),
                converged=bool(optimizer.converged),
            )

            # Save the path
            if output_dir is not None:
                t_grid = integral_output.t.flatten()
                path_output = self.path(t_grid, return_velocities=True, return_energies=True, return_forces=True)
                if self.path.ts_time is not None:
                    ts_time = torch.tensor([self.path.ts_time], device=self.device, dtype=self.dtype)
                    ts_output = self.path(ts_time, return_velocities=True, return_energies=True, return_forces=True)
                    ts_record = {
                        "ts_time": ts_time.tolist(),
                        "ts_positions": ts_output.positions.tolist(),
                        "ts_energies": ts_output.energies.tolist(),
                        "ts_velocities": ts_output.velocities.tolist(),
                        "ts_forces": ts_output.forces.tolist(),
                    }
                else:
                    ts_record = {
                        "ts_time": None,
                        "ts_positions": None,
                        "ts_energies": None,
                        "ts_velocities": None,
                        "ts_forces": None,
                    }

                record = {
                    "time": t_grid.tolist(),
                    "positions": path_output.positions.tolist(),
                    "energies": path_output.energies.tolist(),
                    "velocities": path_output.velocities.tolist(),
                    "forces": path_output.forces.tolist(),
                    "loss_evals": integral_output.y.tolist(),
                    "grad_norm": integral_output.grad_norm.item(),
                    "grad_norm_2": integral_output.grad_norm_2.item(),
                    **ts_record,
                }
                loss = getattr(integral_output, 'loss', None)
                if loss is not None:
                    record["loss"] = loss.tolist()
                with open(os.path.join(log_dir, f"output_{optim_idx}.json"), 'w') as file:
                    json.dump(record, file)

            # Check for convergence
            if optimizer.converged:
                logger.converged(
                    optim_idx,
                    integral_output.grad_norm_2.item(),
                    optimizer.threshold,
                    optimizer.patience,
                )
                break

        logger.end(iters_done)
        logger.close()

