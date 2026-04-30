from typing import Any, Iterable
from collections import defaultdict
import torch


class LossBase():
    """
    Base class for outer-loss wrappers around the path integral.

    Subclasses implement ``__call__(integral_output, **kwargs)`` to
    return a scalar loss. The default ``LossBase.__call__`` reweights
    per-time integrand contributions by ``get_weights`` —
    deprecated under torchpathint because it relies on
    ``IntegralOutput`` fields (``y0``, ``sum_steps``, ``t_optimal``)
    that no longer exist; ``PathIntegral`` is the only loss currently
    supported.
    """

    def __init__(self, weight_scale=None) -> None:
        self.weight_scale = weight_scale
        self.iteration = None
        self.time_midpoint = None
    
    def update_parameters(self, **kwargs):
        if 'weight' in kwargs:
            self.weight_scale = kwargs['weight']
        if 'iteration' in kwargs:
            self.iteration = torch.tensor([kwargs['iteration']])
        # Find the center of the path in time
        if 'integral_output' in kwargs:
            self.time_midpoint = kwargs['integral_output'].t_optimal[:,0]
            t_idx = len(self.time_midpoint)//2
            if len(self.time_midpoint) % 2 == 1:
                self.time_midpoint = self.time_midpoint[t_idx]
            else:
                self.time_midpoint = self.time_midpoint[t_idx-1] + self.time_midpoint[t_idx]
                self.time_midpoint = self.time_midpoint/2.

    def _check_parameters(self, weight_scale=None, **kwargs):
        assert self.weight_scale is not None or weight_scale is not None,\
            "Must provide 'weight_scale' to update_parameters or loss call."
        self.weight_scale = self.weight_scale if weight_scale is None else weight_scale
    
    def get_weights(self, integral_output):
        raise NotImplementedError
    
    def __call__(self, integral_output, **kwargs) -> Any:
        self._check_parameters(**kwargs)
        weights = self.get_weights(
            torch.mean(integral_output.t[:,:,0], dim=1),
            integral_output.t_init,
            integral_output.t_final,
        )
        """
        print("WEIGHTS", self.iteration, weights)
        print(torch.mean(integral_output.t[:,:,0], dim=1))
        fig, ax = plt.subplots()
        ax.set_title(str(self.time_midpoint))
        ax.plot(t_mean, weights)
        ax.plot([0,1], [0,0], ':k')
        ax.set_ylim(-0.1, 1.05)
        fig.savefig(f"test_weights_{self.iteration[0]}.png")
        """

        return integral_output.y0\
            + torch.sum(weights*integral_output.sum_steps[:,0])



class PathIntegral(LossBase):
    """
    Trivial loss: return the path integral as-is.

    The default and currently only fully-supported outer loss.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, integral_output, **kwargs):
        return integral_output.integral[0]


class EnergyWeight(LossBase):
    """
    Energy-weighted reweighting of the path integral.

    .. note::
       Reaches into ``integral_output.y[1]`` and the legacy
       ``y0``/``sum_steps`` fields, which torchpathint does not
       expose. Kept for reference; raises if you select it.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_weights(self, integral_output):
        return torch.mean(integral_output.y[1], dim=1)


class GrowingString(LossBase):
    """
    Growing-string-style envelope reweighting.

    Builds a weighting function over time that's small in the middle
    (where the path is making the fastest progress) and full near the
    endpoints, with tunable envelope shape (``gauss``, ``poly``,
    ``sine``, ``sine-gauss``, ``butter``).

    .. note::
       Same legacy-field issue as ``EnergyWeight`` — currently
       unselectable under torchpathint.
    """

    def __init__(
            self,
            weight_type='inv_sine',
            time_scale=10,
            envelope_scale=1000,
            **kwargs
    ) -> None:
        super().__init__()
        self.iteration = torch.zeros(1)
        self.time_scale = time_scale
        self.envelope_scale = envelope_scale
        self.time_midpoint = 0.5

        idx1 = weight_type.find("_")
        #idx2 = weight_type.find("_", idx1 + 1)
        envelope_key = weight_type[idx1+1:]
        #envelope_key = weight_type[idx1+1:idx2]
        if envelope_key == 'gauss':
            self.envelope_fxn = self._guass_envelope
        elif envelope_key == 'poly':
            self.order = 1 if 'order' not in kwargs else kwargs['order']
            self.envelope_fxn = self._poly_envolope
        elif envelope_key == 'sine':
            self.envelope_fxn = self._sine_envelope
        elif envelope_key == 'sine-gauss' or envelope_key == 'gauss-sine':
            self.envelope_fxn = self._sine_gauss_envelope
        elif envelope_key == 'butter':
            self.order = 8 if 'order' not in kwargs else kwargs['order']
            self._butter_envelope
        else:
            raise ValueError(f"Cannot make envelope type {envelope_key}")
        """
        decay_key = weight_type[idx2+1:]
        if decay_key == 'exp':
            def decay_fxn(iteration, time_scale):
                return self.envelope_scale*torch.exp(-1*iteration*time_scale)
        else:
            raise ValueError(f"Cannot make decay type {decay_key}")
        """

        fxn_key = weight_type[:idx1]
        if fxn_key == 'inv':
            self.get_weights = self._inv_weights
        else:
            raise ValueError(f"Cannot make weight function type {fxn_key}")
    
        
    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)    
        #assert 'variance' in kwargs, "Must provide 'variance' to update_parameters."
        if 'variance' in kwargs:
            self.variance_scale = kwargs['variance']
        if 'order' in kwargs:
            self.order = kwargs['order']

    def _inv_weights(self, time, time_init, time_final):
        envelope = self.envelope_fxn(time, time_init, time_final)
        return 1./(1 + self.weight_scale*envelope)
    
    def _guass_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0:
            left = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.time_midpoint - time_left)*4\
                /(time_init - self.time_midpoint))**2
            )
            time_left = (time_left - time_left[0])\
                /(self.time_midpoint - time_left[0])
            left = left - (left[0] - time_left*left[0])
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.time_midpoint - time_right)*4\
                /(time_final - self.time_midpoint))**2)
            time_right = (time_right - time_right[-1])\
                /(self.time_midpoint - time_right[-1])
            right = right - (right[-1] - time_right*right[-1])
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])
    
    def _sine_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0:
            left = (1 - torch.cos(
                (time_left - time_init)*torch.pi/((self.time_midpoint - time_init))
            ))/2.
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = (1 + torch.cos(
                (time[torch.logical_not(mask)] - self.time_midpoint)\
                    *torch.pi/((time_final - self.time_midpoint))
            ))/2.
        else:
            right = None

        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])

    def _poly_envolope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0: 
            left = torch.abs(
                (time_left - time_init)/((self.time_midpoint - time_init))
            )**self.order
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = torch.abs((time[torch.logical_not(mask)] - time_final)\
                /(time_final - self.time_midpoint))**self.order
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.abs(torch.concatenate([left, right]))

    def _sine_gauss_envelope(self, time, time_init, time_final):
        guass_envelope = self._guass_envelope(time, time_init, time_final)
        sine_envelope = self._sine_envelope(time, time_init, time_final)
        return guass_envelope*sine_envelope


    def _butter_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0: 
            dt = self.time_midpoint - time_left
            left = 1./torch.sqrt(1 + (dt*2/(self.time_midpoint - time_init))**self.order)
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            dt = time_right - self.time_midpoint
            right = 1./torch.sqrt(1 + (dt*2/(self.time_midpoint - time_init))**self.order)
        else:
            right = None

        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])
    
   
LOSS_FXNS = {
    'path_integral' : PathIntegral,
    'integral' : PathIntegral,
    'energy_weight' : EnergyWeight,
    'growing_string' : GrowingString
}

def get_loss_fxn(name, **kwargs):
    """
    Construct a ``LossBase`` subclass by name.

    ``name=None`` returns the default ``PathIntegral``.
    """
    if name is None:
        return LOSS_FXNS['path_integral']()
    assert name in LOSS_FXNS, f"Cannot find loss {name}, must select from {list(LOSS_FXNS.keys())}"
    return LOSS_FXNS[name](**kwargs)
        


class Metrics():
    """
    Registry of per-point integrand functions.

    Each ``ode_fxn`` method takes ``eval_time`` plus the path and
    returns a per-time tensor of loss values plus a dict of cached
    intermediate quantities (energies, forces, velocities). The
    integrator selects one or more by name via ``create_ode_fxn``;
    selected functions are summed (with optional per-term scales)
    inside ``ode_fxn``.

    To add a new integrand, add a method here that follows the
    ``(get_required_variables=False, **kwargs)`` protocol and append
    its name to ``all_ode_fxn_names``.
    """

    all_ode_fxn_names = [
        'projected_variational_reaction_energy',
        'variable_reaction_energy',
        'vre_variational_error',
        'projected_variational_reaction_energy_mag',
        'E',
        'E_mean',
        'F_mag'
    ]

    def __init__(self, device, save_energy_force=False):
        self.save_energy_force = save_energy_force
        self.device = device
        self._ode_fxn_scales = None
        self._ode_fxns = None

    def create_ode_fxn(self, fxn_names, fxn_scales=None):
        """
        Resolve metric names to bound methods and remember per-term scales.

        After this returns, ``self.ode_fxn`` evaluates the weighted
        sum at any ``eval_time``.
        """
        # Parse and check input
        assert fxn_names is not None or len(fxn_names) != 0
        if isinstance(fxn_names, str):
            fxn_names = [fxn_names]
        if fxn_scales is None:
            fxn_scales = torch.ones(len(fxn_names), device=self.device)
        elif not isinstance(fxn_scales, Iterable):
            fxn_scales = [fxn_scales]

        assert len(fxn_names) == len(fxn_scales), f"The number of metric function names {fxn_names} does not match the number of scales {fxn_scales}"

        for idx, fname in enumerate(fxn_names):
            if fname not in dir(self):
                metric_fxns = [
                    attr for attr in dir(Metrics)\
                        if attr[0] != '_' and callable(getattr(Metrics, attr))
                ]
                raise ValueError(f"Can only integrate metric functions, either add a new function to the Metrics class or use one of the following:\n\t{metric_fxns}")
            if fname in fxn_names[idx+1:]:
                raise ValueError(f"Cannot use the same metric function twice in the same class instantiation: {fname}")
        self._ode_fxns = [getattr(self, fname) for fname in fxn_names]
        self._ode_fxn_scales = {
            fxn.__name__ : scale for fxn, scale in zip(self._ode_fxns, fxn_scales)
        }

        self._get_required_variables()


    def _get_required_variables(self):
        assert self._ode_fxns is not None
        self.required_variables = defaultdict(lambda : False)
        for fxn in self._ode_fxns:
            for var in fxn(get_required_variables=True):
                self.required_variables[f"requires_{var}"] = True
    
    def add_required_variable(self, variable_name):
        self.required_variables[variable_name] = True
    

    def ode_fxn(self, eval_time, path, **kwargs):
        """
        Evaluate ``Σ scale_i * metric_i(eval_time, path)``.

        Per-term metrics share their cached path evaluation through
        ``kwargs``, so when several terms need the same energies/forces
        the path is only walked once per ``eval_time`` batch.
        """
        loss = 0
        for fxn in self._ode_fxns:
            scale = self._ode_fxn_scales[fxn.__name__]
            ode_loss, ode_variables = fxn(
                eval_time=eval_time,
                path=path,
                **self.required_variables,
                **kwargs
            )
            kwargs.update(ode_variables)
            loss = loss + scale*ode_loss
        
        if self.save_energy_force:
            nans = torch.tensor(
                [torch.nan,]*len(kwargs['time']),
                device=self.device
            ).unsqueeze(-1)

            keep_variables = [
                kwargs[name] if name in kwargs and kwargs[name] is not None else nans\
                    for name in ['energies', 'forces']
            ]
            loss = torch.concatenate([loss] + keep_variables, dim=-1)

        return loss

    
    def update_ode_fxn_scales(self, **kwargs):
        """
        Replace one or more per-term scales. Used by schedulers to
        ramp terms up/down between iterations.
        """
        for name, scale in kwargs.items():
            assert name in self._ode_fxn_scales
            self._ode_fxn_scales[name] = scale


    def _parse_input(
            self,
            eval_time,
            path,
            # Returns a dict of every variable any metric might want,
            # re-using cached fields when the input matches a previous
            # eval_time and only re-walking the path when something is
            # missing or stale.
            time=None,
            positions=None,
            velocities=None,
            energies=None,
            energies_decomposed=None,
            forces=None,
            forces_decomposed=None,
            requires_velocities=False,
            requires_energies=False,
            requires_energies_decomposed=False,
            requires_forces=False,
            requires_forces_decomposed=False,
        ):
        
        # Do input and previous time match
        time_match = time is not None\
            and (time.shape == eval_time.shape\
                 and torch.allclose(time, eval_time, atol=1e-10)
            )

        # Is energies missing and required 
        requires_energies = requires_energies and energies is None
        requires_energies_decomposed = requires_energies_decomposed and energies_decomposed is None
        missing_any_energy = requires_energies or requires_energies_decomposed
        
        # Is forces missing and required 
        requires_forces = requires_forces and forces is None
        requires_forces_decomposed = requires_forces_decomposed and forces_decomposed is None
        missing_any_force = requires_forces or requires_forces_decomposed

        # We must evaluate path if time do not match, or, forces or energies is missing
        path_output = None
        if not time_match or missing_any_energy or missing_any_force:
            path_output = path(
                eval_time,
                return_velocities=requires_velocities,
                return_energies=requires_energies, 
                return_energies_decomposed=requires_energies_decomposed, 
                return_forces=requires_forces,
                return_forces_decomposed=requires_forces_decomposed
            )
            time = eval_time
            velocities = velocities if path_output.velocities is None\
                else path_output.velocities
            energies = energies if path_output.energies is None\
                else path_output.energies
            energies_decomposed = energies_decomposed if path_output.energies_decomposed is None\
                else path_output.energies_decomposed
            forces = forces if path_output.forces is None\
                else path_output.forces
            forces_decomposed = forces_decomposed if path_output.forces_decomposed is None\
                else path_output.forces_decomposed

        else:
           # Calculate velocities if missing and required
            if requires_velocities and velocities is None:
                velocities = path.calculate_velocities(time)
                requires_velocities = False
            
        return {
            'time' : time,
            'positions' : positions,
            'velocities' : velocities,
            'energies' : energies,
            'energies_decomposed' : energies_decomposed,
            'forces' : forces,
            'forces_decomposed' : forces_decomposed
        }


    def geodesic(self, get_required_variables=False, **kwargs):
        """
        Path-length on a force-decomposed metric:
        ``‖F_decomposed · v‖₂``. Use with the ``repel`` potential
        for geodesic interpolation (clash resolution).
        """
        if get_required_variables:
            return ('forces_decomposed', 'velocities')
        variables = self._parse_input(**kwargs)

        projection = torch.einsum(
            'bki,bi->bk',
            variables['forces_decomposed'],
            variables['velocities']
        )
        Egeo = torch.linalg.norm(projection, dim=-1, keepdim=True)
        return Egeo, variables


    def variable_reaction_energy(self, get_required_variables=False, **kwargs):
        """
        Magnitudes-only product: ``‖F‖ · ‖v‖``. Used in combination
        with PVRE so that ``VRE - PVRE`` is the angular mismatch.
        """
        if get_required_variables:
            return ('forces', 'velocities')
        variables = self._parse_input(**kwargs)

        F = torch.linalg.norm(variables['forces'], dim=-1, keepdim=True)
        V = torch.linalg.norm(variables['velocities'], dim=-1, keepdim=True)
        return F*V, variables


    def projected_variational_reaction_energy(self, get_required_variables=False, **kwargs):
        """
        ``|v · F|``. Drives configurations where the force is
        perpendicular to the path direction — the saddle-point
        condition. Default loss for transition-state search.
        """
        if get_required_variables:
            return ('forces', 'velocities')
        variables = self._parse_input(**kwargs)
        overlap = torch.sum(
            variables['velocities']*variables['forces'],
            dim=-1,
            keepdim=True
        )
        return torch.abs(overlap), variables


    def projected_variational_reaction_energy_mag(self, get_required_variables=False, **kwargs):
        """``‖v ⊙ F‖₂`` — per-component product, then norm."""
        if get_required_variables:
            return ('forces', 'velocities')
        variables = self._parse_input(**kwargs)

        magnitude = torch.linalg.norm(variables['velocities']*variables['forces'], dim=-1, keepdim=True)
        return magnitude, variables


    def E(self, get_required_variables=False, **kwargs):
        """Raw potential energy."""
        if get_required_variables:
            return ('energies',)
        variables = self._parse_input(**kwargs)

        return variables['energies'], variables


    def E_mean(self, get_required_variables=False, **kwargs):
        """Mean energy across the (possibly batched) trailing dim."""
        if get_required_variables:
            return ('energies',)
        variables = self._parse_input(**kwargs)

        mean_E = torch.mean(variables['energies'], dim=-1, keepdim=True)
        return mean_E, variables


    def vre_variational_error(self, get_required_variables=False, **kwargs):
        """
        ``VRE - PVRE`` — force-velocity angular mismatch. Approaches
        zero on a true MEP where forces are tangent to the path.
        """
        if get_required_variables:
            return (
                *self.projected_variational_reaction_energy(get_required_variables=True),
                *self.variable_reaction_energy(get_required_variables=True)
            )
        variables = self._parse_input(**kwargs)

        pvre, _ = self.projected_variational_reaction_energy(
            eval_time=kwargs['eval_time'], path=kwargs['path'], **variables
        )
        vre, _ = self.variable_reaction_energy(
            eval_time=kwargs['eval_time'], path=kwargs['path'], **variables
        )
        return vre - pvre, variables


    def F_mag(self, get_required_variables=False, **kwargs):
        """``‖F‖₂`` — force magnitude. Useful as a TS-time loss."""
        if get_required_variables:
            return ('forces',)
        variables = self._parse_input(**kwargs)

        Fmag = torch.linalg.norm(variables['forces'], dim=-1, keepdim=True)
        return Fmag, variables