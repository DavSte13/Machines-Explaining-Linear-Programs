# Machines-Explaining-Linear-Programs

This is the repository for the paper Machines Explaining Linear Programs. It provides the implementation of several
linear programs (LPs) and integer programs (ILPs), as well as for three different attribution methods. Furthermore, it
provides a way to evaluate the attribution methods on these problems. All components are implemented in PyTorch, the LPs
are based on cvxpylayers, and the ILPs are based on [CombOptNet](https://github.com/martius-lab/CombOptNet).

## Installation:

* Install the `requirements.txt`
* Update the git submodule for CombOptNet: Run `git submodule init` and afterwards `git submodule update`
* CombOptNet and therefore also the ILPs are based on the Gurobi solver. To use it, it is necessary to
  obtain a [license](https://www.gurobi.com/documentation/9.1/quickstart_mac/obtaining_a_grb_license.html) and download/set it.
  This step is not necessary for the LPs. (This step is not required to evaluate the LPs).
* To allow the CombOptNet submodule to work properly, it is necessary to slightly change the inputs in the file `CombOptNet/models/comboptnet.py`.
  In lines 10 and 12, it is necessary to change the import statements from `utils.[...]` to `CombOptNet.utils.[...]`.

## Usage:

The objective of this repository is to provide a way to explain LPs. This is done by generating attributions for the LPs
via several neural attribution methods. The attribution methods provide an explanation to the LPs by quantifying the 
influence of each input parameter on the output of the LP. The LP parameters are in general called (a, b, c)
in the code, which refer to the naming of (A, b, w) in the paper. The following four attribution methods are supported:

* Saliency (Pure gradients)
* Gradient times Input
* Integrated Gradients
* Occlusion

To evaluate how effective these methods are at explaining an LP, they were evaluated by hand (similar to heatmaps of 
images) on several problems:

* Resource Optimization (LP)
* Maximum Flow (LP)
* Knapsack (ILP)
* Shortest Path (ILP)
* PlexPlain (Real-World LP example)
* Voting (A small LP instantiation of the MAP problem)

The first four problems can be evaluated with `evaluation/evaluate_problems.py`. The settings for the evaluation are
defined in `evaluation/config.py`. The results of this evaluation are stored in `evaluation/results`, which already
contains the results of the paper. It is possible to display the stored results with `evaluation/print_cases.py`.

The PlexPlain example can be accessed directly through `evaluation/evaluate_plexplain.py` and the MAP example is available
under `evaluation/map/voting_example.py`.
