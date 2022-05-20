import final_evaluation.evaluate_max_flow as evaluate_max_flow
import final_evaluation.evaluate_basic_lp as evaluate_basic_lp
import final_evaluation.config as config

import pandas as pd

#todo: Import the other two problem files - handle the dependencies to comboptnet

# from ex_ilp import evaluate_knapsack_final
# from ex_ilp import evaluate_sp_final


def evaluate_problems(problem, methods=('grad', 'gxi', 'ig', 'gc'), output_file=None):
    """
    Evaluates the given problem. By default, all methods are evaluated on all cases for that problem, but it is possible
        to overwrite the default methods parameter to specify exactly which methods should be evaluated. If no output
        filename is specified, the results are saved in a json called problem_methods.json, where methods are all used
        methods separated by underscores.
    problem is one of: Basic LP: 'blp', Maximum Flow: 'mf', Knapsack: 'ks', Shortest Path: 'sp'
    methods is a tuple of at least one of the following: Gradients: 'grad', Gradient times Input 'gxi',
        Integrated Gradients 'ig', Granger Causal attributions: 'gc'

    """

    eval_problem = {
        'blp': evaluate_basic_lp.evaluate_basic_lp,
        'mf': evaluate_max_flow.evaluate_max_flow,
        # 'ks': evaluate_knapsack_final.evaluate_knapsack,  # todo
        # 'sp': evaluate_sp_final.evaluate_sp,  # todo
    }

    # fetch all problem cases for the specified problem:
    cases = config.PROBLEM_CASES[problem]

    # prepare the result dataframe
    col_list = ['eval_func', 'method', 'input', 'result', 'attributions', 'baseline', 'description']
    results = pd.DataFrame(columns=col_list)

    # evaluation_function, input_parameters, attribution_method, attribution_params=None

    case_numbers = []
    # iterate over all cases:
    for eval_func in ['cost', 'opt']:
        # accumulate the data for all runs
        description_list = []
        input_list = []
        method_list = []
        baseline_list = []
        case_num = 0
        for case in cases:
            case_num += 1
            for method in methods:
                if not method == 'ig':
                    # append the data for the current run to the lists (the method is not ig -> only one run)
                    input_list.append(case['input'])
                    method_list.append(method)
                    baseline_list.append(None)
                    description_list.append(case['info'])
                    case_numbers.append(case_num)
                else:
                    # for integrated gradients, it is necessary to add one run per baseline to the lists
                    for i in range(case['num_of_bases']):
                        input_list.append(case['input'])
                        method_list.append(method)
                        baseline_list.append(case[f'base{i}'])
                        description_list.append(case['info'])
                        case_numbers.append(case_num)

        # evaluate the problem cases and append the intermediate results to the dataframe
        tmp = eval_problem[problem](eval_func, input_list, method_list, baseline_list, description_list)
        results = results.append(tmp, ignore_index=True)

    # add the problem column to the results dataframe
    results['problem'] = [problem] * results.shape[0]
    results['case'] = case_numbers

    # create the filename for the json
    if output_file is None:
        save_path = f'results/{problem}_{"_".join(methods)}.json'
    else:
        save_path = f'results/{output_file}.json'

    results.to_json(save_path, orient='split')


evaluate_problems('mf')
