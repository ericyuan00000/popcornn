# Popcornn
Path Optimization with a Continuous Representation Neural Network for reaction path with machine learning interatomic potentials

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library. Please install (or load) conda and then proceed with the following commands:
```
conda create --name popcornn python=3.12
conda activate popcornn
```

Now, you can install Popcornn in the conda environment by cloning this repository:
```
git clone https://github.com/khegazy/popcornn.git

pip install -e ./popcornn
```
Several machine learning potentials have been interfaced with Popcornn, such as CHGNet, EScAIP, LEFTNet, MACE, [NewtonNet](https://github.com/THGLab/NewtonNet), Orb, [UMA](https://github.com/facebookresearch/fairchem), etc. Please refer to the respective codespaces for the installation guides. To run the latest UMA model, you'll also have to set it up through [HuggingFace](https://rowansci.com/blog/how-to-run-open-molecules-2025).

## Quick Start
You can find several run files inside the `examples` directory that rely on the implemented modules in the Popcornn library. We provide a simple run script, which needs to be accompanied with a yaml config file. You can run an example optimization script with the following command in the `examples` directory:
```
cd popcornn/examples

python run.py --config configs/rxn0003.yaml
```
All Popcornn parameters are specified in the config file. This example should complete in under an hour. Convergence is driven by `threshold` — the threshold on the L∞ norm of the path-integrated gradient, checked for `patience` consecutive iterations before the optimizer exits. `num_optimizer_iterations` is the cap if the trigger never fires. See the *Convergence* section below for picking these values.

For fast development and playing around with popcornn we offer two examples based on the the Wolfe potential
```
python run.py --config configs/wolfe.yaml
```
will run the fast Wolfe example and 
```
python run.py --config configs/loss_example.yaml
```
will run the fast Wolfe example with more advanced loss capabilities. We note that values in the `loss_example.yaml` files are chosen to demonstrate the capabilities of popcornn and are not optimal values for either Wolfe or other systems.

## Set up your own Popcornn
The config file is read in the run script as a dictionary, so you can also directly specify the configs in your own python script, giving you more handles on the inputs and outputs.

### Initialize the path
The first step for you would be to specify the endpoints of the reaction you are working on:
```
from ase.io import read

images = read('configs/rxn0003.xyz', index=':')
for image in images:
    image.info = {"charge": 0, "spin": 1}  # if required by the MLIP, set the total charge and multiplicity
```
It can be a list of ASE Atoms, or if it's a string, we can also read `xyz` or `traj` files. If there are more than 2 frames provided, the path will be first fitted to go through the intermediate frames, but they are not fixed. Note that the reactant and product should be index-mapped, rotationally/translationally aligned, and ideally unwrapped if periodic. By default, if the periodic boundary conditions are applied, we unwrap the product according to the minimum image convention with respect to the reactant; however, if the cell is small and some atoms are expected to move more than half a cell, you should unwrap the frames manually and disable `unwrap_positions` in `path_params` (see below).

Next, you can set up the path using the images:
```
from popcornn import Popcornn

path = Popcornn(images=images, path_params={'name': 'mlp', 'n_embed': 1, 'depth': 2})
```
Optional initialization parameters for Popcornn include `num_record_points` for the number of frames to be recorded after optimization, `output_dir` for optional debug outputs, `device`, `dtype`, and `seed`. For simpler reactions, `depth` of 2 helps limit the complexity of the reaction, while more complicated reactions may require a deeper path neural network.

### Optimize the path
Machine learning potentials are vulnerable to unphysical, out-of-distribution configurations, and it's important to resolve atom clashing as an interpolation step. Luckily, you can do both the interpolation and the optimization with Popcornn. The state of the art regarding the interpolation is to optimize the path on a monotonic, repulsive potential with respect to the geodesic loss, or so-called [geodesic interpolation](https://pubs.aip.org/aip/jcp/article/150/16/164103/198363/Geodesic-interpolation-for-reaction-pathways). In general, therefore, you need multiple optimizations by providing multiple `optimization_params`, each with a different potential, integral loss, and optimizer:
```
final_images, ts_image = path.optimize_path(
    {
        'potential_params': {'potential': 'repel'},
        'integrator_params': {'path_ode_names': 'geodesic'},
        'optimizer_params': {'optimizer': {'name': 'adam', 'lr': 1.0e-1}},
        'num_optimizer_iterations': 1000,
    },
    {
        'potential_params': {'potential': 'uma', 'model_name': 'uma-s-1p1', 'task_name': 'omol'},
        'integrator_params': {'path_ode_names': 'projected_variational_reaction_energy', 'rtol': 1.0e-2, 'atol': 1.0e-2},
        'optimizer_params': {'optimizer': {'name': 'adam', 'lr': 1.0e-3}, 'threshold': 1.0e-1},
        'num_optimizer_iterations': 1000,
    },
)
```
Finally, after optimization, you can save the optimized path as a list of Atoms for visualization and further optimization:
```
from ase.io import write

write('popcornn.xyz', final_images)
write('popcornn_ts.xyz', ts_image)
```
In this example, you should get a barrier of ~3.6 eV. To be fully rigorous, we [suggest](https://www.nature.com/articles/s41467-024-52481-5) doing a subsequent saddle point optimization for the Popcornn transition state followed by forward/reverse intrisic reaction coordinate calculations, since Popcornn is not actually returning a minimum energy path but just targeting the transition state directly. Both saddle point optimization and intrisic reaction coordinate calculation are supported by [Sella](https://github.com/zadorlab/sella/tree/master) as ASE Optimizers.

### Convergence
Each `optimization_params` leg exits when the L∞ norm of the path-integrated gradient `‖∫∇L dt‖_∞` falls below `threshold` for `patience` consecutive iterations, or when `num_optimizer_iterations` is reached, whichever comes first. The L∞ norm (per-component max) was chosen over the L2 norm because it does not rescale with √(parameter count), so the same threshold transfers more cleanly across MLP sizes.

The threshold is **system-dependent**: gradient magnitudes differ by orders of magnitude between toy potentials and real MLIP-driven runs. A reasonable recipe:
1. Run a short pilot (e.g. 50 iterations) with `threshold` omitted (defaults to no trigger) and read the early-iteration `‖∫∇L dt‖_∞` from the integrator output.
2. Set `threshold` roughly one order of magnitude below the early reading. For a system with initial `|g|_inf ≈ 20`, `threshold = 1.0` is a sensible default; for a system with initial `|g|_inf ≈ 1.5`, try `0.1`.
3. Leave `patience` at its default of 5; this absorbs single-iteration dips during the damped-oscillation phase Adam exhibits as it settles. Override only if you have a strong reason to.

The Müller-Brown example (`configs/muller_brown.yaml`) uses `threshold: 1.0`; the UMA-driven `configs/rxn0003.yaml` uses `threshold: 1.0e-1`. Both are calibrated to their respective systems' gradient scales.

If you want to monitor `∫L dt` (the actual loss integral) per iteration without using it for convergence, set `track_loss: true` in `integrator_params`. The integrator runs a separate detached quadrature with `loss_rtol`/`loss_atol` (defaulting to `rtol`/`atol`) and exposes the result on `path_integral.loss_integral`.

### Managing memory and handling potential OOM errors
Popcornn uses [torchpathdiffeq](https://github.com/khegazy/torchpathdiffeq/tree/main) to numerically calculate the path integral used in the loss. When calculating the path integral, torchpathdiffeq employs an adaptive evaluation mesh. This adaptive mesh results in varying batch sizes within a single torchpathdiffeq evaluation. The adaptive batch size may grow and can lead to out-of-memory errors (OOM). To avoid these OOM errors, torchpathdiffeq adaptively limits the batch size based on the cached and free GPU memory. While infrequent, it is possible for OOM errors to occur. There are two reasons for this, each with a solution. The user should apply these solutions in this order if an OOM error occurs.
1. PyTorch memory management is optimized to run a single batch size many times, however, the adaptive path integration scheme in torchpathdiffeq varies the batch size. To allow PyTorch to resize the memory used for batch evaluations, the user must specify the following environment flag
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
or in Python, place
```
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```
at the top of the first Python file being called. This will solve most OOM errors.

2. If the OOM error persists, then torchpathdiffeq is struggling to calculate the memory footprint for your problem. To solve this, the user must limit the available GPU memory torchpathdiffeq is allowed to use via the `total_mem_usage` flag. `total_mem_usage` is a ratio between 0 and 1 that determines how much of the available GPU memory will be allocated for the next batch evaluation. `total_mem_usage` is set in the run config file, inside the `integration_params`, with a default value of 0.9.
```
integrator_params: 
    path_ode_names: projected_variational_reaction_energy
    rtol: 1.0e-5
    atol: 1.0e-7
    total_mem_usage: 0.75
```


## Support

Popcornn is still under active development, and we welcome any feedback or contributions. Please open a GitHub issue if any problems are encountered!
