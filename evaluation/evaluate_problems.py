import evaluation.evaluate_max_flow as evaluate_max_flow
import evaluation.evaluate_resource_optimization as evaluate_resource_optimization
import evaluation.evaluate_knapsack as evaluate_knapsack
import evaluation.evaluate_shortest_path as evaluate_shortest_path
import evaluation.config as config
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Evaluate attribution methods on different LP and ILP problems.')
parser.add_argument("problem", help="Select the problem to evaluate on. Possible options: Resource optimization 'ro'"
                                    ", Maximum Flow 'mf', Knapsack 'ks' or Shortest Path 'sp'.",
                    choices=['ro', 'mf', 'ks', 'sp'])
parser.add_argument("--methods", nargs='+', help="Select which methods should be evaluated. It is possible to select"
                                                 "one or multiple of: Gradients 'grad', Gradient times Input 'gxi'"
                                                 "Integrated Gradients 'ig' or Occlusion 'occ'. By default, all "
                                                 "methods are used.")
parser.add_argument("--output_file", help="Specify an output file name. By default, the filename is "
                                          "'problem_methods.json', where methods are the abbreviation of all used "
                                          "methods separated by underscores.")
args = parser.parse_args()


def evaluate_problems(problem, methods=('grad', 'gxi', 'ig', 'occ'), output_file=None):
    """
    Evaluates the given problem. By default, all methods are evaluated on all cases for that problem, but it is possible
      to overwrite the default methods parameter to specify exactly which methods should be evaluated. If no output
      filename is specified, the results are saved in a json called problem_methods.json, where methods are the
      abbreviation of all methods all used separated by underscores.
    problem is one of: Resource Optimization 'ro', Maximum Flow: 'mf', Knapsack: 'ks', Shortest Path: 'sp'
    methods is a tuple of at least one of the following: Gradients: 'grad', Gradient times Input 'gxi',
      Integrated Gradients 'ig', Occlusion: 'occ'
    """
    eval_problem = {
        'ro': evaluate_resource_optimization.evaluate_ro,
        'mf': evaluate_max_flow.evaluate_max_flow,
        'ks': evaluate_knapsack.evaluate_knapsack,
        'sp': evaluate_shortest_path.evaluate_sp,
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


evaluate_problems(args.problem, args.method, args.output_file)
