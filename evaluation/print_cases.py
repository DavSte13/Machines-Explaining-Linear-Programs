import pandas as pd
import numpy as np
import xlp_utils


def print_case(problem, case, methods=None, evaluation_functions=None, file_path=None):
    """
    Print the results of the specified experiments in a nice matter.
    :param problem: Problem specification (One of: 'blp', 'mf', 'ks' or 'sp')
    :param case: Case of the problem (number from 1 onwards)
    :param methods: For which methods the results should be printed (Tuple of any from: 'grad', 'gxi', 'ig' or 'gc')
        If not specified: all methods are used.
    :param evaluation_functions: Which evaluation functions should be considered: (Tuple of any from: 'cost', 'opt')
        If not specified: both functions are printed.
    :param file_path: The path to the result data (if not specified, the default path is used).
    """

    if file_path is None:
        file_path = f'results/{problem}_grad_gxi_ig_gc.json'

    data = pd.read_json(file_path, orient='split')

    # filter the specific case
    data = data[data['case'] == case].reset_index()

    # filter for method and evaluation function
    if methods is not None:
        data = data[data['method'] in methods].reset_index()
    else:
        methods = ('grad', 'gxi', 'ig', 'gc')
    if evaluation_functions is not None:
        data = data[data['evaluation_function'] in evaluation_functions].reset_index()
    else:
        evaluation_functions = ('cost', 'opt')

    # for printing
    problem_dict = {
        'blp': 'Basic Linear Program',
        'mf': 'Maximum Flow Problem',
        'ks': 'Knapsack Problem',
        'sp': 'Shortest Path Problem',
    }
    method_dict = {
        'grad': 'Gradients',
        'gxi': 'Gradient times Input',
        'ig': 'Integrated Gradients',
        'gc': 'Granger Causal Attributions',
    }
    eval_dict = {
        'cost': 'objective function',
        'opt': 'optimal solution',
    }

    header = f"================= {problem_dict[problem]} =================\n" \
             f"Case {case}: {data['description'][0]}\n"

    if problem in ['blp']:
        a, b, c = data['input'][0]
        inp = f"Input for A: {a}\n" \
              f"Input for b: {b}\n" \
              f"Input for c: {c}\n"

    elif problem in ['ks']:
        items, limit = data['input'][0]
        inp = f"Input weights:\t {[i[1] for i in items]}\n" \
              f"Input values:\t {[i[0] for i in items]}\n" \
              f"Input limit:\t {limit}\n"

    elif problem in ['mf', 'sp']:
        v, e = data['input'][0]
        inp = f"Input nodes: {v}\n" \
              f"Input edges: {e}\n"
        # also print the input as a, b and c

    result = f"Result\t\t\t\t {data['result'][0]}\n" \
             f"Optimal Solution\t {data[data['eval_func']== 'opt'].reset_index()['result'][0]}\n\n"

    print(header)
    print(inp)
    print(result)

    # print attributions for all methods (sub-header for the method, then all attributions divided by input)
    for ev in evaluation_functions:
        sub_header = f"\nAttributions for the {eval_dict[ev]}:\n"
        print(sub_header)
        data_ev = data[data['eval_func'] == ev]

        for m in methods:
            if m in ['grad', 'gxi']:
                data_m = data_ev[data_ev['method'] == m].reset_index()
                a, b, c = data_m['attributions'][0]
                attr = f"{method_dict[m]}:\n" \
                       f"Attributions for A: {np.around(np.array(a), 4).tolist()}\n" \
                       f"Attributions for b: {np.around(np.array(b), 4).tolist()}\n" \
                       f"Attributions for c: {np.around(np.array(c), 4).tolist()}\n"
                print(attr)

            elif m == 'ig':
                data_m = data_ev[data_ev['method'] == m].reset_index()
                print(f"{method_dict[m]}:")
                for attributions, baseline in zip(data_m['attributions'], data_m['baseline']):
                    a, b, c = attributions
                    if problem in ['blp']:
                        a_base, b_base, c_base = baseline
                        base = f"Baseline for A: {np.around(np.array(a_base), 4).tolist()}\n" \
                               f"Baseline for b: {np.around(np.array(b_base), 4).tolist()}\n" \
                               f"Baseline for c: {np.around(np.array(c_base), 4).tolist()}\n"
                    elif problem in ['ks']:
                        i_base, l_base = baseline
                        base = f"Baseline and Attributions for {method_dict[m]}:\n" \
                               f"Baseline items: {i_base}\n" \
                               f"Baseline limit: {l_base}\n"
                    elif problem in ['mf', 'sp']:
                        v_base, e_base = baseline

                        base = f"Baseline for V: {v_base}\n" \
                               f"Baseline for E: {e_base}\n"

                    attr = f"Attributions for A: {np.around(np.array(a) * - 1, 4).tolist()}\n" \
                           f"Attributions for b: {np.around(np.array(b) * - 1, 4).tolist()}\n" \
                           f"Attributions for c: {np.around(np.array(c) * - 1, 4).tolist()}\n"
                    print(base)
                    print(attr)

            elif m == 'gc':
                data_m = data_ev[data_ev['method'] == m].reset_index()
                attributions = data_m['attributions'][0]
                attr = f"{method_dict[m]}:\n"

                if problem == 'blp':
                    for i in range(len(attributions)):
                        attr += f"Attributions for constraint {i+1}: {np.around(np.array(attributions[i]), 4).tolist()}\n"
                elif problem in ['mf', 'sp']:
                    for i in range(len(attributions)):
                        attr += f"Attributions for edge {i+1}: {xlp_utils.round_array_with_none(np.array(attributions[i]), 4).tolist()}\n"
                elif problem == 'ks':
                    for i in range(len(attributions)):
                        attr += f"Attributions for item {i+1}: {xlp_utils.round_array_with_none(np.array(attributions[i]), 4).tolist()}\n"

                print(attr)


print_case('ks', 4)
