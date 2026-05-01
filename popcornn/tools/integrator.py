import torch

from torchpathint import path_integral

from .integrand import build_integrand_terms, evaluate_integrand_sum


class PathIntegrator:
    """
    Adaptive-quadrature path integrator.

    Wraps ``torchpathint.path_integral`` with the parts of the popcornn API
    the optimizer uses:

    - per-iteration ``integrand_scales`` updates so schedulers take effect,
    - direct scatter of ``∂L/∂θ`` into ``path.parameters().grad``
      (no separate ``.backward()`` call),
    - a ``loss`` field on the returned object holding ``‖∫∇L dt‖_∞``
      for the convergence check,
    - an optional detached pass that integrates the loss itself for
      monitoring.
    """

    def __init__(
            self,
            method='gk21',
            path_integrand_names=None,
            path_integrand_scales=None,
            rtol=1e-6,
            atol=1e-7,
            max_batch=None,
            path_integrand_energy_idx=1,
            path_integrand_force_idx=2,
            track_loss=False,
            loss_rtol=None,
            loss_atol=None,
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
        path_integrand_energy_idx, path_integrand_force_idx : int
            Reserved indices for the ``[loss, E, F]`` concat; unused
            with ``save_energy_force=False`` but kept for API parity.
        track_loss : bool, default=False
            Run a separate detached integral of the loss itself for
            monitoring. Costs an extra pass.
        loss_rtol, loss_atol : float, optional
            Tolerances for the detached loss integral. Default to
            ``rtol``/``atol``.
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
        self.device = device
        self.dtype = dtype
        self.N_integrals = 0
        self.integral_output = None

        self.path_integrand_energy_idx = path_integrand_energy_idx
        self.path_integrand_force_idx = path_integrand_force_idx

        if path_integrand_names is None:
            self._terms = []
        else:
            self._terms = build_integrand_terms(path_integrand_names, path_integrand_scales)

        # save_energy_force=False: gradient pipeline only consumes the scalar
        # integrand component, not the [E, F] concat.
        self._save_energy_force = False

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
        underlying ``IntegralOutput`` with ``.loss`` set to
        ``‖∫∇L dt‖_∞`` for the convergence check.

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

        def f(t_flat):
            l, _ = evaluate_integrand_sum(
                self._terms,
                t_flat.unsqueeze(-1),
                path,
                save_energy_force=self._save_energy_force,
            )
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

        integral_output = path_integral(
            f,
            t_init_0d,
            t_final_0d,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            max_batch=self.max_batch,
            device=self.device,
            dtype=self.dtype,
        )

        # Scatter the [D] integrated gradient into param.grad. Accumulate so
        # multiple integrate_path calls between optimizer.zero_grad() compose.
        offset = 0
        flat = integral_output.integral.detach()
        for p, k in zip(params, sizes):
            chunk = flat[offset:offset + k].reshape(p.shape)
            p.grad = chunk if p.grad is None else p.grad + chunk
            offset += k

        # No scalar loss graph in this design. Surface ‖∫∇L dt‖_∞ (per-component
        # max) as the convergence signal consumed by PathOptimizer's threshold
        # check. L∞ is closer to MLP-size-independent than L2 — the latter scales
        # with √D and forces the threshold to be retuned per parameter count.
        integral_output.loss = flat.abs().max()

        if self.track_loss:
            def fval(t_flat):
                l, _ = evaluate_integrand_sum(
                    self._terms,
                    t_flat.unsqueeze(-1),
                    path,
                    save_energy_force=self._save_energy_force,
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
            integral_output.loss_integral = loss_output.integral.detach()

        self.integral_output = integral_output
        self.N_integrals += 1
        return integral_output
