from dataclasses import dataclass

import torch

from torchpathint import path_integral

from .integrand import build_integrand_terms, evaluate_integrand_sum


@dataclass(frozen=True)
class SamplesCache:
    """Per-quadrature-point energy/force samples harvested during integration.

    Shape contract:
    - ``time``: ``[N*K]``, flattened from ``IntegralOutput.t``'s ``[N, K]``.
    - ``energies``: ``[N*K, E]`` (typically ``E=1``; potentials may emit a
      decomposed energy tensor).
    - ``forces``: ``[N*K, D]`` where ``D`` is the flattened atomic dof.

    The integrator populates this object from the same evaluations that
    produced the gradient integral, so consuming it for transition-state
    finding adds zero extra path-forward calls.
    """

    time: torch.Tensor
    energies: torch.Tensor
    forces: torch.Tensor


class PathIntegrator:
    """
    Adaptive-quadrature path integrator.

    Wraps ``torchpathint.path_integral`` with the parts of the popcornn API
    the optimizer uses:

    - per-iteration ``integrand_scales`` updates so schedulers take effect,
    - direct scatter of ``∂L/∂θ`` into ``path.parameters().grad``
      (no separate ``.backward()`` call),
    - a ``grad_integral`` alias on the returned object for the flat
      ``[D]`` integrated gradient, and ``grad_norm`` / ``grad_norm_2``
      fields holding ``‖∫∇L dt‖_∞`` (used by the convergence check) and
      ``‖∫∇L dt‖_2`` (for monitoring),
    - an optional detached pass that integrates the loss itself, exposed
      as a ``loss`` field for monitoring,
    - an optional ``save_samples`` mode that captures per-quadrature-point
      ``(t, energies, forces)`` for transition-state finding without any
      extra path evaluations.
    """

    def __init__(
            self,
            method='gk21',
            path_integrand_names=None,
            path_integrand_scales=None,
            rtol=1e-6,
            atol=1e-7,
            max_batch=None,
            track_loss=False,
            loss_rtol=None,
            loss_atol=None,
            save_samples=False,
            full_output=False,
            device=None,
            dtype=None,
        ):
        """
        Parameters
        ----------
        method : str, default="gk21"
            torchpathint quadrature rule. ``gk21`` is Gauss–Kronrod 21pt.
        path_integrand_names : str or list of str, optional
            Per-point integrand (or list of them) to integrate. Looked up
            by name in ``PATH_INTEGRANDS``. See ``docs/loss-functions.md``.
        path_integrand_scales : float or list, optional
            Weighting per term when ``path_integrand_names`` is a list.
        rtol, atol : float
            Adaptive-quadrature tolerances on the gradient integral.
        max_batch : int, optional
            Hard cap on the number of quadrature points evaluated in
            one batch. ``None`` lets torchpathint auto-size.
        track_loss : bool, default=False
            Run a separate detached integral of the scalar loss
            ``∫L(t) dt`` for monitoring. Costs an extra pass; when
            enabled the result is attached as ``IntegralOutput.loss``.
        loss_rtol, loss_atol : float, optional
            Tolerances for the detached loss integral. Default to
            ``rtol``/``atol``.
        save_samples : bool, default=False
            If True, capture ``(t, energies, forces)`` at every quadrature
            point evaluated during ``integrate_path`` and attach a
            ``SamplesCache`` to the returned ``IntegralOutput.samples``.
            Energies and forces are forced into resolution via
            ``evaluate_integrand_sum(also_resolve=('energies', 'forces'))``;
            since ``BasePath.forward`` calls the potential exactly once
            per evaluation regardless of which fields are requested, this
            adds no path-forward calls beyond the existing gradient pass.
            Implies ``full_output=True`` (``_stitch_samples`` needs ``.t``).
        full_output : bool, default=False
            Forwarded to ``torchpathint.path_integral``. When True, the
            returned ``IntegralOutput`` carries the per-interval mesh
            ``.t`` and per-point evaluations ``.y``; when False those
            are ``None``. Set this on the integrator from outside when
            something downstream needs the diagnostic mesh — e.g.
            popcornn's per-iter JSON dump (``output_dir`` set). The
            effective value is OR-ed with ``save_samples``.
        device : torch.device
        dtype : torch.dtype
        """
        self.method = method
        self.atol = atol
        self.rtol = rtol
        # Loss integral is debug-only; let it run at its own (looser by
        # default) tolerance rather than paying for gradient-grade accuracy.
        self.track_loss = track_loss
        self.loss_rtol = loss_rtol if loss_rtol is not None else rtol
        self.loss_atol = loss_atol if loss_atol is not None else atol
        self.max_batch = max_batch
        self.save_samples = save_samples
        self.full_output = full_output
        self.device = device
        self.dtype = dtype
        self.N_integrals = 0
        self.integral_output = None

        if path_integrand_names is None:
            self._terms = []
        else:
            self._terms = build_integrand_terms(path_integrand_names, path_integrand_scales)

    def update_integrand_scales(self, **kwargs):
        """Replace one or more per-term scales. Used by schedulers to ramp
        terms up/down between iterations."""
        for name, scale in kwargs.items():
            for i, term in enumerate(self._terms):
                if term.name == name:
                    term.scale = float(scale)
                    break
            else:
                raise KeyError(
                    f"No integrand named {name!r} in this integrator "
                    f"(have: {[t.name for t in self._terms]})."
                )

    def integrate_path(
            self,
            path,
            integrand_scales=None,
            t_init=torch.tensor([0.]),
            t_final=torch.tensor([1.]),
        ):
        """
        Integrate the gradient of the loss along the path.

        Sets ``param.grad`` for each path parameter to the integrated
        gradient ``∫₀¹ ∂L/∂θ dt`` (accumulated, so multiple calls
        between ``optimizer.zero_grad()`` compose). Also returns the
        underlying ``IntegralOutput`` enriched with popcornn-level
        fields:

        - ``.grad_integral``: alias for the flat ``[D]`` integrated
          gradient (same tensor as ``.integral`` from torchpathint;
          named for clarity at popcornn call sites).
        - ``.grad_norm``: ``‖∫∇L dt‖_∞`` (convergence trigger).
        - ``.grad_norm_2``: ``‖∫∇L dt‖_2`` (monitoring only).
        - ``.loss``: scalar ``∫L(t) dt`` when ``track_loss=True``.
        - ``.samples``: ``SamplesCache`` (or ``None``) holding
          per-quadrature-point energies/forces when ``save_samples=True``.

        Parameters
        ----------
        path : BasePath
            Holds the trainable parameters and potential.
        integrand_scales : dict, optional
            Updated values for per-term scales (from schedulers).
        t_init, t_final : torch.Tensor
            Integration bounds, in [0, 1]. Squeezed to 0-d for
            torchpathint.
        """
        if integrand_scales:
            self.update_integrand_scales(**integrand_scales)

        # torchpathint requires 0-d bounds; popcornn historically passed 1-d.
        t_init_0d = torch.as_tensor(t_init).squeeze()
        t_final_0d = torch.as_tensor(t_final).squeeze()

        params = list(path.parameters())
        sizes = [p.numel() for p in params]

        # Side-buffer for transition-state-finding samples. Each entry is a
        # ``(t_chunk, E_chunk, F_chunk)`` triplet captured inside ``f`` and
        # later reassembled in IntegralOutput.t.flatten() order via byte-keyed
        # lookup (the same t tensor is passed to f and indexed into accepted_t
        # by torchpathint, so byte equality holds).
        sample_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        also_resolve = ('energies', 'forces') if self.save_samples else ()

        def f(t_flat):
            l, variables = evaluate_integrand_sum(
                self._terms,
                t_flat.unsqueeze(-1),
                path,
                also_resolve=also_resolve,
            )
            if self.save_samples:
                sample_buffer.append((
                    t_flat.detach().cpu(),
                    variables['energies'].detach().cpu(),
                    variables['forces'].detach().cpu(),
                ))
            l_per_t = l.reshape(t_flat.shape[0], -1).sum(dim=-1)  # [N], graph live
            n = l_per_t.shape[0]
            grad_out = torch.eye(n, device=l_per_t.device, dtype=l_per_t.dtype)
            grads = torch.autograd.grad(
                outputs=l_per_t,
                inputs=params,
                grad_outputs=grad_out,
                is_grads_batched=True,
            )
            return torch.cat([g.reshape(n, -1) for g in grads], dim=-1)  # [N, D]

        # full_output gates whether torchpathint populates .t and .y on the
        # returned IntegralOutput. Required when something downstream reads
        # them: _stitch_samples (save_samples=True) or popcornn._optimize's
        # per-iter JSON dump (caller sets self.full_output=True). Off by
        # default to avoid the diagnostic-buffer overhead.
        full_output = self.save_samples or self.full_output
        integral_output = path_integral(
            f,
            t_init_0d,
            t_final_0d,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            max_batch=self.max_batch,
            full_output=full_output,
            device=self.device,
            dtype=self.dtype,
        )

        # ``integral_output.integral`` is torchpathint's generic name for the
        # integrated function value — here it's the flat [D] gradient. Alias
        # to ``.grad_integral`` so popcornn-internal call sites read clearly
        # without disambiguating against the loss-integral pass below.
        integral_output.grad_integral = integral_output.integral

        # Scatter the [D] integrated gradient into param.grad. Accumulate so
        # multiple integrate_path calls between optimizer.zero_grad() compose.
        offset = 0
        flat = integral_output.grad_integral.detach()
        for p, k in zip(params, sizes):
            chunk = flat[offset:offset + k].reshape(p.shape)
            p.grad = chunk if p.grad is None else p.grad + chunk
            offset += k

        # No scalar loss graph in this design. Surface ‖∫∇L dt‖_∞ (per-component
        # max) as the convergence signal consumed by PathOptimizer's threshold
        # check. L∞ is closer to MLP-size-independent than L2 — the latter scales
        # with √D and forces the threshold to be retuned per parameter count.
        # Also expose L2 for monitoring (cheap on the same flat tensor).
        integral_output.grad_norm = flat.abs().max()
        integral_output.grad_norm_2 = flat.norm()

        if self.save_samples:
            integral_output.samples = self._stitch_samples(sample_buffer, integral_output.t)
        else:
            integral_output.samples = None

        if self.track_loss:
            def fval(t_flat):
                l, _ = evaluate_integrand_sum(
                    self._terms,
                    t_flat.unsqueeze(-1),
                    path,
                )
                return l.reshape(t_flat.shape[0], -1).detach()
            loss_output = path_integral(
                fval,
                t_init_0d,
                t_final_0d,
                method=self.method,
                atol=self.loss_atol,
                rtol=self.loss_rtol,
                max_batch=self.max_batch,
                device=self.device,
                dtype=self.dtype,
            )
            integral_output.loss = loss_output.integral.detach()

        self.integral_output = integral_output
        self.N_integrals += 1
        return integral_output

    def _stitch_samples(self, sample_buffer, accepted_t):
        """Reassemble per-call (t, E, F) buffer entries into a flat
        ``SamplesCache`` aligned with ``accepted_t.flatten()``.

        torchpathint passes the same ``t_eval_pending`` tensor to ``f``
        and indexes it into ``accepted_t_eval``, so the float bytes of
        an accepted-point t are byte-identical to one of the rows we
        captured. Refinement iterations may leave rejected-interval
        points in the buffer; the byte-keyed lookup just skips them.
        """
        lookup: dict[bytes, tuple[torch.Tensor, torch.Tensor]] = {}
        for t_chunk, e_chunk, f_chunk in sample_buffer:
            t_np = t_chunk.numpy()
            for i in range(t_np.shape[0]):
                lookup[t_np[i].tobytes()] = (e_chunk[i], f_chunk[i])

        flat_t = accepted_t.detach().cpu().flatten()
        es: list[torch.Tensor] = []
        fs: list[torch.Tensor] = []
        flat_t_np = flat_t.numpy()
        for i in range(flat_t.shape[0]):
            key = flat_t_np[i].tobytes()
            entry = lookup.get(key)
            if entry is None:
                raise RuntimeError(
                    "save_samples: accepted t has no matching captured "
                    "evaluation. The byte-identity invariant between "
                    "torchpathint's t_eval_pending and accepted_t was "
                    "violated; check whether the integrator was modified."
                )
            es.append(entry[0])
            fs.append(entry[1])

        time = flat_t.to(self.device)
        energies = torch.stack(es, dim=0).to(self.device)
        forces = torch.stack(fs, dim=0).to(self.device)
        return SamplesCache(time=time, energies=energies, forces=forces)
