import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.functional import interpolate
from popcornn.tools import scheduler
from popcornn.tools.scheduler import get_schedulers

from popcornn.tools import Metrics


class PathOptimizer():
    """
    Single-leg path optimizer.

    Owns the underlying ``torch.optim`` optimizer, the (optional) LR
    scheduler, all per-loss-term schedulers, the convergence-trigger
    state, and the optional transition-state loss machinery. One
    instance per ``Popcornn._optimize`` leg.
    """

    def __init__(
            self,
            path,
            optimizer=None,
            find_ts=None,
            lr_scheduler=None,
            path_loss_schedulers=None,
            path_ode_schedulers=None,
            ts_time_loss_names=None,
            ts_time_loss_scales=torch.ones(1),
            ts_time_loss_schedulers=None,
            ts_region_loss_names=None,
            ts_region_loss_scales=torch.ones(1),
            ts_region_loss_schedulers=None,
            threshold=None,
            patience=5,
            device='cpu',
            dtype=None,
            **config
        ):
        """
        Parameters
        ----------
        path : BasePath
            The path whose parameters this optimizer steps.
        optimizer : dict
            ``{"name": <torch.optim class>, ...kwargs}``. Required.
        find_ts : bool, optional
            Force-enable / force-disable transition-state extraction.
            ``None`` lets the path's own ``find_ts`` flag decide.
        lr_scheduler : dict, optional
            ``{"name": <torch.optim.lr_scheduler class>, ...kwargs}``.
        path_loss_schedulers, path_ode_schedulers : dict, optional
            Schedules for outer-loss params and per-ODE-term scales.
        ts_time_loss_names, ts_time_loss_scales, ts_time_loss_schedulers : optional
            Loss applied at the predicted TS time. Currently a no-op
            because TS extraction is paused (see
            ``TODO(restore-ts-extraction)`` in ``popcornn.py``).
        ts_region_loss_names, ts_region_loss_scales, ts_region_loss_schedulers : optional
            Loss applied across a small window around the predicted TS.
        threshold : float, optional
            Convergence trigger on ``‖∫∇L dt‖_∞``. ``None`` disables.
        patience : int, default=5
            Number of consecutive iterations the trigger must hold.
        device : str
        dtype : torch.dtype
        **config
            Reserved for forward compatibility.
        """
        super().__init__()

        self.find_ts = find_ts
        self.device=device
        self.dtype=dtype
        self.iteration = 0
        self.threshold = threshold
        self.patience = patience
        self._below_threshold_count = 0
        
        ####  Initialize transition state loss information  #####
        self.has_ts_time_loss = ts_time_loss_names is not None
        self.has_ts_region_loss = ts_region_loss_names is not None
        self.has_ts_loss = self.has_ts_time_loss or self.has_ts_region_loss
        if self.has_ts_loss:
            if self.find_ts is None or self.find_ts:
                self.find_ts = True
            else:
                raise ValueError("Cannot have transition state losses and set find_ts=False")
        
        self.ts_time_loss_names = ts_time_loss_names
        self.ts_time_loss_scales = ts_time_loss_scales
        if self.has_ts_time_loss:
            self.ts_time_metrics = Metrics(device)
            self.ts_time_metrics.create_ode_fxn(
                self.ts_time_loss_names, self.ts_time_loss_scales
            )
        
        self.ts_region_loss_names = ts_region_loss_names
        self.ts_region_loss_scales = ts_region_loss_scales
        if self.has_ts_region_loss:
            self.ts_region_metrics = Metrics(device)
            self.ts_region_metrics.create_ode_fxn(
                self.ts_region_loss_names, self.ts_region_loss_scales
            )
        
        #####  Initialize schedulers  #####
        self.ode_fxn_schedulers = get_schedulers(path_ode_schedulers)
        self.path_loss_schedulers = get_schedulers(path_loss_schedulers)
        self.ts_time_loss_schedulers = get_schedulers(ts_time_loss_schedulers)
        self.ts_region_loss_schedulers = get_schedulers(ts_region_loss_schedulers)
        
        #####  Initialize optimizer  #####
        self.path = path
        if optimizer is not None:
            self.set_optimizer(**optimizer)
        else:
            raise ValueError("Must specify optimizer parameters (dict) with key 'optimizer'")

        #####  Initialize learning rate scheduler  #####
        if lr_scheduler is not None:
            self.set_lr_scheduler(**lr_scheduler)
        else:
            self.lr_scheduler = None
        self.converged = False

    def set_optimizer(self, name, **config):
        """
        Build and attach a ``torch.optim`` optimizer by class name.

        ``name`` matches against ``dir(torch.optim)`` case-insensitively
        — i.e. ``"adam"`` finds ``torch.optim.Adam``, ``"lbfgs"`` finds
        ``torch.optim.LBFGS``. Extra kwargs forward to the constructor.
        """
        optimizer_dict = {key.lower(): key for key in dir(optim) if not key.startswith('_')}
        name = optimizer_dict[name.lower()]
        optimizer_class = getattr(optim, name)
        self.optimizer = optimizer_class(self.path.parameters(), **config)

    def set_lr_scheduler(self, name, **config):
        """
        Attach a ``torch.optim.lr_scheduler`` by class name. Same
        case-insensitive name resolution as ``set_optimizer``.
        """
        scheduler_dict = {key.lower(): key for key in dir(lr_scheduler) if not key.startswith('_')}
        assert name.lower() in scheduler_dict,\
            f"Scheduler class does not support '{name}', either add this functionality or select from {list(scheduler_dict.keys())}"
        name = scheduler_dict[name.lower()]
        scheduler_class = getattr(lr_scheduler, name)
        self.lr_scheduler = scheduler_class(self.optimizer, **config)

    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.]),
            t_final=torch.tensor([1.]),
            update_path=True
        ):
        """
        Run one optimizer step.

        Steps:

        1. Read the current scheduled values for all per-loss scales.
        2. Hand the path + loss to ``integrator.integrate_path``,
           which scatters ``∂L/∂θ`` into ``path.parameters().grad``.
        3. (Currently no-op) Add gradients from any TS-time / TS-region
           losses.
        4. Step Adam, all schedulers, and the LR scheduler.
        5. Update the convergence-trigger counter.

        Returns
        -------
        IntegralOutput
            The integrator output, with ``.loss`` overridden to the
            ``‖∫∇L dt‖_∞`` value used for convergence checks.
        """
        self.optimizer.zero_grad()
        t_init = t_init.to(self.dtype).to(self.device)
        t_final = t_final.to(self.dtype).to(self.device)
        ode_fxn_scales = {
            name : schd.get_value() for name, schd in self.ode_fxn_schedulers.items()
        }
        path_loss_scales = {
            name : schd.get_value() for name, schd in self.path_loss_schedulers.items()
        }
        path_loss_scales['iteration'] = self.iteration,
        
        if self.has_ts_loss:
            ts_time_loss_scales = {
                name : schd.get_value() for name, schd in self.ts_time_loss_schedulers.items()
            }
            ts_region_loss_scales = {
                name : schd.get_value() for name, schd in self.ts_region_loss_schedulers.items()
            }
        path_integral = integrator.integrate_path(
            path,
            ode_fxn_scales=ode_fxn_scales,
            loss_scales=path_loss_scales,
            t_init=t_init,
            t_final=t_final,
        )
        # integrate_path scatters dL/dθ into path.parameters().grad directly;
        # no .backward() call here.


        #####  Transition State  #####
        # ts_search consumes the integrator's quadrature points and assumes
        # the old torchpathdiffeq RK layout. Skipped under the torchpathint
        # migration; revisit once the quadrature-output adaptor lands.

        # Evaluate transition state losses
        if self.find_ts and path.ts_time is not None:
            if self.has_ts_time_loss:
                self.ts_time_metrics.update_ode_fxn_scales(**ts_time_loss_scales)
                ts_time_loss = self.ts_time_metrics.ode_fxn(
                    torch.tensor([[path.ts_time]]), path
                )[:,0]
                ts_time_loss.backward()
            if self.has_ts_region_loss:
                self.ts_region_metrics.update_ode_fxn_scales(
                    **ts_region_loss_scales
                )
                ts_region_loss = self.ts_region_metrics.ode_fxn(
                    path.ts_region[:,None], path
                )[:,0]
                ts_region_loss.backward()

        #####  Update Optimization  #####
        # Path update step
        if update_path:
            self.optimizer.step()
        # Update schedulers
        for name, sched in self.ode_fxn_schedulers.items():
            sched.step() 
        for name, sched in self.path_loss_schedulers.items():
            sched.step()
        if self.has_ts_loss:
            for name, sched in self.ts_time_loss_schedulers.items():
                sched.step() 
            for name, sched in self.ts_region_loss_schedulers.items():
                sched.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Convergence: ‖∫∇L dt‖ below threshold for `patience` consecutive
        # iterations. Patience guards against single-step dips driven by
        # adaptive-quadrature error wiggling around the threshold.
        if self.threshold is not None:
            if path_integral.loss.item() < self.threshold:
                self._below_threshold_count += 1
                if self._below_threshold_count >= self.patience:
                    self.converged = True
            else:
                self._below_threshold_count = 0

        self.iteration = self.iteration + 1

        return path_integral

    