import torch
import pandas as pd
import numpy as np
from problems.max_flow import MaxFlow
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients
from attribution_methods.occlusion import Occlusion
import xlp_utils


def evaluate_max_flow(evaluation_function, input_parameters, attribution_method, baselines, descriptions):
    """
    :param evaluation_function: String which signals how the optimal solution should be processed: 'opt' - remains as
        the optimal solution; 'cost' - apply the cost function to the optimal solution.
    :param input_parameters: list of: graph specifications: tuples (V, E) where V is a list of node indices and E is a
        dict of edges including weights
    :param attribution_method: list of: Which attribution method should be applied, one of: 'grad': gradients; 'gxi': gradient
        times input; 'ig': integrated gradients; 'occ': occlusion
    :param baselines: list of: None or a valid baseline. For each entry in attribution methods which is 'ig', a baseline
        has to be given.
    :param descriptions: list of: strings which are added to the resulting dataframe entry as a description
    """
    num_runs = len(input_parameters)

    # set the parameters for the first case
    m = len(input_parameters[0][1])
    n = len(input_parameters[0][0])

    mf = MaxFlow(m, n, evaluation_function)
    mf_reduced = MaxFlow(m, n, evaluation_function, reduce_dimension=True)

    # prepare the attribution methods
    gxi = GradientXInput(mf)
    ig = IntegratedGradients(mf)
    occ = Occlusion(mf, mf_reduced, evaluation_function, reduce_rows=False, max_flow=True)

    # prepare the result dataframe
    col_list = ['method', 'eval_func', 'input', 'result', 'attributions', 'baseline', 'description']
    attributions = pd.DataFrame(columns=col_list)

    # compute the attributions for the specified attribution method and all inputs
    for i in range(num_runs):

        # if the size of the problem has changed:
        if len(input_parameters[i][1]) != m or len(input_parameters[i][0]) != n:
            # create a new problem
            m = len(input_parameters[i][1])
            n = len(input_parameters[i][0])
            mf = MaxFlow(m, n, evaluation_function)
            mf_reduced = MaxFlow(m, n, evaluation_function, reduce_dimension=True)

            # create the new attribution methods
            gxi = GradientXInput(mf)
            ig = IntegratedGradients(mf)
            occ = Occlusion(mf, mf_reduced, evaluation_function, reduce_rows=False, max_flow=True)

        # get input parameters from graph
        a, b, c = xlp_utils.convert_graph(*input_parameters[i], 'mf', plot=False)
        a_tch = torch.tensor(a, requires_grad=True)
        b_tch = torch.tensor(b, requires_grad=True)
        c_tch = torch.tensor(c, requires_grad=True)
        inp = (a_tch, b_tch, c_tch)

        # compute the result of the LP
        result = np.around(mf.forward(*inp).detach().numpy(), 4)
        result = result.tolist()
        
        # create attributions and detach them
        if attribution_method[i] == 'grad':
            attr = xlp_utils.detach_tuple(torch.autograd.functional.jacobian(mf, inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'gxi':
            attr = xlp_utils.detach_tuple(gxi.attribute(inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'ig':
            # get the baseline parameters from the baseline graph
            base_a, base_b, base_c = xlp_utils.convert_graph(*baselines[i], 'mf', plot=False)
            base = (torch.Tensor(base_a), torch.Tensor(base_b), torch.Tensor(base_c))

            attr, _ = ig.attribute(inp, base, steps=100)
            # convert the numpy arrays to lists for storing later
            attr = xlp_utils.numpy_tuple_to_list(attr)
        elif attribution_method[i] == 'occ':
            attr = xlp_utils.detach_tuple(occ.attribute(inp), to_list=True)

        # add the results to the attributions dataframe and detach them
        inp = xlp_utils.numpy_tuple_to_list(input_parameters[i])
        base = xlp_utils.numpy_tuple_to_list(baselines[i])
        values = pd.DataFrame([[attribution_method[i], evaluation_function, inp, result, attr,
                                base, descriptions[i]]], columns=col_list)
        attributions = attributions.append(values, ignore_index=True)

    return attributions
