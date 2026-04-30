import torch

from torchpathint import path_integral

from .metrics import Metrics, get_loss_fxn


_PATH_INTEGRAL_LOSSES = {None, 'path_integral', 'integral'}


class ODEintegrator(Metrics):
    """
    Adaptive-quadrature path integrator.

    Wraps ``torchpathint.path_integral`` with the parts of the
    popcornn API the optimizer uses:

    - per-iteration ``ode_fxn_scales`` updates so schedulers take effect,
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
            path_loss_name=None,
            path_loss_params=None,
            path_ode_names=None,
            path_ode_scales=None,
            rtol=1e-6,
            atol=1e-7,
            max_batch=None,
            path_ode_energy_idx=1,
            path_ode_force_idx=2,
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
        path_loss_name : {"path_integral", "integral", None}, optional
            Outer-loss wrapper. Other choices (``EnergyWeight``,
            ``GrowingString``) reach into ``IntegralOutput`` fields
            that no longer exist under torchpathint and will raise.
        path_loss_params : dict, optional
            Forwarded to the loss wrapper's constructor.
        path_ode_names : str or list of str, optional
            Per-point quantity (or list of them) to integrate. Looked
            up by name on the parent ``Metrics`` class. See
            ``docs/loss-functions.md``.
        path_ode_scales : float or list, optional
            Weighting per term when ``path_ode_names`` is a list.
        rtol, atol : float
            Adaptive-quadrature tolerances on the gradient integral.
        max_batch : int, optional
            Hard cap on the number of quadrature points evaluated in
            one batch. ``None`` lets torchpathint auto-size.
        path_ode_energy_idx, path_ode_force_idx : int
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
        # save_energy_force=False: gradient pipeline only consumes the scalar
        # metric component of ode_fxn output, not the [E, F] concat.
        super().__init__(device, save_energy_force=False)

        if path_loss_name not in _PATH_INTEGRAL_LOSSES:
            raise NotImplementedError(
                f"path_loss_name={path_loss_name!r} is not supported under "
                f"torchpathint yet — only {_PATH_INTEGRAL_LOSSES} are migrated. "
                "EnergyWeight / GrowingString reach into IntegralOutput fields "
                "(t_optimal, sum_steps, y0) that no longer exist."
            )

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

        self.path_ode_energy_idx = path_ode_energy_idx
        self.path_ode_force_idx = path_ode_force_idx

        if path_ode_names is None:
            self.eval_fxns = None
            self.eval_fxn_scales = None
            self.ode_fxn = None
        else:
            self.create_ode_fxn(path_ode_names, path_ode_scales)

        self.loss_name = path_loss_name
        self.loss_fxn = get_loss_fxn(path_loss_name, **(path_loss_params or {}))

    def integrate_path(
            self,
            path,
            ode_fxn_scales=None,
            loss_scales=None,
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
        ode_fxn_scales : dict, optional
            Updated values for per-ODE-term scales (from schedulers).
        loss_scales : dict, optional
            Updated values for outer-loss parameters.
        t_init, t_final : torch.Tensor
            Integration bounds, in [0, 1]. Squeezed to 0-d for
            torchpathint.
        """
        if ode_fxn_scales:
            self.update_ode_fxn_scales(**ode_fxn_scales)
        self.loss_fxn.update_parameters(**(loss_scales or {}))

        # torchpathint requires 0-d bounds; popcornn historically passed 1-d.
        t_init_0d = torch.as_tensor(t_init).squeeze()
        t_final_0d = torch.as_tensor(t_final).squeeze()

        params = list(path.parameters())
        sizes = [p.numel() for p in params]

        def f(t_flat):
            # ode_fxn returns [N, K=1, 1] with save_energy_force=False.
            l = self.ode_fxn(t_flat.unsqueeze(-1), path)
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
                l = self.ode_fxn(t_flat.unsqueeze(-1), path)
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
