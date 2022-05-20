import pandas as pd
import numpy as np
import ex_lp_utils

problem_dict = {
    'blp': 'basic linear program',
    'mf': 'maximum flow problem',
    'ks': 'knapsack problem',
    'sp': 'shortest path problem',
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


def table_skeleton(columns, cap_short, cap_long, label):
    """
    Returns the basic skeleton for a latex table, with the given column specifier (in latex-style),
    a short and long caption as well as a label
    The return consists of two different string, the initial string and the end string
    """
    start = "\\begin{table}\n" \
            "\t \\centering\n" \
            "\t \\begin{tabular}{" + columns + "}\\toprule"
    end = "\t \\bottomrule\n" \
          "\t \\end{tabular}\n" \
          "\t \\caption[" + cap_short + "]{" + cap_long + "}\n" \
          "\t \\label{tab:" + label + "}\n" \
          "\\end{table}\n"

    return start, end


def la_mat(mat):
    """
    Converts the matrix to a matrix entry for latex.
    """
    assert len(mat.shape) == 2
    cols = "c" * mat.shape[1]

    start = "$\\left(\\begin{array}{" + cols + "}\n"
    entries = ""
    for row in mat:
        row_s = [str(x) for x in row.tolist()]
        tmp = " & ".join(row_s)
        tmp = "\t" + tmp + "\\\\ \n"
        entries = entries + tmp
    entries = entries[:-4] + "\n"
    end = "\\end{array}\\right)$"

    return start + entries + end


def la_vec(vec):
    """
    Converts the vector to a vector entry for latex.
    """
    assert(len(vec.shape) == 1)

    if vec.shape[0] == 1:
        return str(vec[0])
    else:
        return "$(" + ", ".join([str(x) for x in vec.tolist()]) + ")$"


def conv_entry(entry):
    if len(entry.shape) == 0:
        return str(entry)
    elif len(entry.shape) == 1:
        return la_vec(entry)
    elif len(entry.shape) == 2:
        return la_mat(entry)
    else:
        raise ValueError("only 1 or 2 dimensional arrays are allowed as entries. Entry:", entry)


def print_table(data, cap_short, cap, label, header, use_first_col, first_col, col_type="r"):
    """
    Prints the data as a latex table
    data: table data as a 2d list
    cap_short: short caption
    cao: long caption
    label: latex label
    header: header line for the table (list of strings)
    use_first_col: If the first column should be additional information
    first_col: additional information as list of strings (only used if use_first_col == True)
    col_type: column type for the data columns (default "r", can also be "s")
    """
    data_cols = len(data[0])
    data_rows = len(data)
    if use_first_col:
        assert data_cols + 1 == len(header)
        assert len(first_col) == data_rows
        cols = "l" + col_type * data_cols
    else:
        assert data_cols == len(header)
        cols = col_type * data_cols

    table_start, table_end = table_skeleton(cols, cap_short, cap, label)
    first_row = "\t\t" + " & ".join(header) + "\\\\ \\midrule"
    other_rows = ""
    for i in range(data_rows):
        tmp = "\t\t"
        if use_first_col:
            tmp += first_col[i] + " & "
        tmp += " & ".join([conv_entry(e) for e in data[i]])
        other_rows = other_rows + tmp + "\\\\ \n"
    other_rows = other_rows[:-2]

    print(table_start)
    print(first_row)
    print(other_rows)
    print(table_end)


def print_params(problem, case):
    """
    Print the parameters of the problem (and case) as a latex table.
    """

    file_path = f'results/{problem}_grad_gxi_ig_gc.json'
    data = pd.read_json(file_path, orient='split')

    if problem == 'blp':
        baselines = []
        inputs = []
        # load baselines, convert to arrays
        data_f = data[data['case'] == 1]
        data_f = data_f[(data_f['eval_func'] == 'opt') & (data_f['method'] == 'ig')]
        for base in data_f['baseline']:
            tmp = [np.array(i) for i in base]
            tmp = ex_lp_utils.round_tuple(tmp)
            baselines.append(tmp)

        # load inputs, convert to arrays
        data_f = data[(data['method'] == 'grad') & (data['eval_func'] == 'opt')]
        for inp in data_f['input']:
            tmp = [np.array(i) for i in inp]
            tmp = ex_lp_utils.round_tuple(tmp)
            inputs.append(tmp)

        inputs.extend(baselines)
        table_data = inputs
        header = ["", "A", "b", "c"]
        first_col = ["Parameters case 1", "Parameters case 2", "Parameters case 3", "Parameters case 4",
                     "Parameters case 5", "IG-nz", "IG-c1", "IG-c2", "IG-bo"]

        cap = f"Parameters and baselines for the basic linear program"
        cap_short = cap
        label = f"param_blp"

    elif problem == 'ks':
        data_f = data[(data['case'] == case) & (data['eval_func'] == 'opt')]
        data_f = data_f[data_f['method'] == 'ig'].reset_index()
        inp = data_f['input'][0]
        tmp = np.array(ex_lp_utils.round_tuple(inp[0]))
        tmp = np.append(tmp, np.array([[np.nan, inp[1]]]), axis=0)
        inputs = tmp.T
        num_items = inputs.shape[1] - 1

        baselines = []
        for base in data_f['baseline']:
            tmp = np.array(ex_lp_utils.round_tuple(base[0]))
            tmp = np.append(tmp, np.array([[np.nan, base[1]]]), axis=0)
            tmp = tmp.T
            baselines.append(tmp)

        baselines = np.vstack(baselines)
        inputs = np.vstack((inputs, baselines))
        table_data = inputs

        header = ["Item"] + [f"{i + 1}" for i in range(num_items)] + ["Limit (c)"]

        first_col = ["Values (c)", "Weights (A)", "IG-nz values", "IG-nz weights", "IG-av values", "IG-av weights"]

        cap = f"Parameters and baselines for the knapsack problem, case {case}"
        cap_short = cap
        label = f"param_ks_c{case}"

    elif problem in ['mf', 'sp']:

        data_f = data[(data['case'] == case) & (data['eval_func'] == 'opt')]
        data_f = data_f[data_f['method'] == 'ig'].reset_index()
        edges = data_f['input'][0][1]
        weights = np.array([e['w'] for (_, _, e) in edges])
        inputs = np.around(weights, 3)

        num_edges = inputs.shape[0]

        baselines = []
        for base in data_f['baseline']:
            edges = np.array(base[1])
            weights = np.array([e['w'] for (_, _, e) in edges])
            weights = np.around(weights, 3)
            baselines.append(weights)

        baselines = np.vstack(baselines)
        inputs = np.vstack((inputs, baselines))
        table_data = inputs

        header = ["Edge"] + [f"{i + 1}" for i in range(num_edges)]

        first_col = ["Parameters", "IG-nz", "IG-as"]

        cap = f"Parameters and baselines for the {problem_dict[problem]} problem, case {case}"
        cap_short = cap
        label = f"param_{problem}_c{case}"
    else:
        raise ValueError("The specified method is not implemented.")

    print_table(table_data, cap_short, cap, label, header, True, first_col, col_type="S")


def print_attributions_obj(problem, case):
    """
    Print the attributions for the objective function (for the given problem and case) in latex.
    """

    file_path = f'results/{problem}_grad_gxi_ig_gc.json'
    data = pd.read_json(file_path, orient='split')

    data = data[(data['problem'] == problem) & (data['case'] == case) & (data['eval_func'] == 'cost')]

    if problem == 'blp':
        attributions = []
        for attr in data['attributions']:
            tmp = [np.array(i) for i in attr]
            tmp = ex_lp_utils.round_tuple(tmp)
            attributions.append(tmp)

        table_data = attributions[:-1]
        header = ["Method", "A", "b", "c"]
        first_col = ["Grad", "GxI", "IG-nz", "IG-c1",
                     "IG-c2", "IG-bo"]

        cap = f"Attributions for the objective function of the basic linear program, case {case}"
        cap_short = f"Attributions for the basic linear program, objective function, case {case}"
        label = f"attr_blp_obj_c{case}"
        col_type = "c"

    elif problem == 'ks':
        attributions = [[], [], []]
        data_m = data[(data['method'] == 'grad') | (data['method'] == 'gxi') | (data['method'] == 'ig')]
        for attr in data_m['attributions']:
            tmp = np.round(np.array(attr[0]) * -1, 3)
            attributions[0].append(tmp.T)
            tmp = np.round(attr[1][0][0] * -1, 3)
            attributions[1].append(tmp)
            tmp = np.round(np.array(attr[2]) * -1, 3)
            attributions[2].append(tmp.reshape(-1, 1))

        table_data = attributions
        header = ["Parameter", "Grad", "GxI", "IG-nz", "IG-av"]
        first_col = ["A", "b", "c"]

        cap = f"Attributions for the objective function of the knapsack program, case {case}"
        cap_short = f"Attributions for the knapsack, objective function, case {case}"
        label = f"attr_{problem}_obj_c{case}"
        col_type = "c"

    elif problem in ['sp', 'mf']:
        attributions = []
        for attr in data['attributions']:
            tmp1 = np.round(np.array(attr[0]), 3)
            tmp2 = np.round(np.array(attr[1]), 3).reshape(-1, 1)
            tmp3 = np.round(np.array(attr[2]), 3).reshape(-1, 1)
            attributions.append([tmp1, tmp2, tmp3])

        table_data = attributions[:-1]
        header = ["Method", "A", "b", "c"]
        first_col = ["Grad", "GxI", "IG-nz", "IG-as"]

        cap = f"Attributions for the objective function of the {problem_dict[problem]}, case {case}"
        cap_short = f"Attributions for the {problem_dict[problem]}, objective function, case {case}"
        label = f"attr_{problem}_obj_c{case}"
        col_type = "c"

    print_table(table_data, cap_short, cap, label, header, True, first_col, col_type=col_type)


def print_attributions_opt(problem, case):
    """
    Print the attributions for the optimal solution (for the given problem and case) in latex.
    """

    file_path = f'results/{problem}_grad_gxi_ig_gc.json'
    data = pd.read_json(file_path, orient='split')

    data = data[(data['problem'] == problem) & (data['case'] == case) & (data['eval_func'] == 'opt')]

    if problem == 'blp':
        attributions = []  # for x1 and x2
        data_m = data[(data['method'] == 'grad') | (data['method'] == 'gxi') | (data['method'] == 'ig')]
        for attr in data_m['attributions']:
            tmp0_x1 = np.round(np.array(attr[0][0]), 3)
            tmp0_x2 = np.round(np.array(attr[0][1]), 3)
            tmp1_x1 = np.round(np.array(attr[1][0]), 3)
            tmp1_x2 = np.round(np.array(attr[1][1]), 3)
            tmp2_x1 = np.round(np.array(attr[2][0]), 3)
            tmp2_x2 = np.round(np.array(attr[2][1]), 3)

            attributions.append([tmp0_x1, tmp1_x1, tmp2_x1])
            attributions.append([tmp0_x2, tmp1_x2, tmp2_x2])

        table_data = attributions
        header = ["Method", "A", "b", "c"]
        first_col = ["Grad $x_1$", "Grad $x_2$", "GxI $x_1$", "GxI $x_2$", "IG-nz $x_1$", "IG-nz $x_2$",
                     "IG-c1 $x_1$", "IG-c1 $x_2$", "IG-c2 $x_1$", "IG-c2 $x_2$", "IG-bo $x_1$", "IG-bo $x_2$"]

        cap = f"Attributions for the optimal solution of the basic linear program, case {case}"
        cap_short = f"Attributions for the basic linear program, optimal solution, case {case}"
        label = f"attr_blp_opt_c{case}"
        col_type = "c"

    elif problem == 'ks':
        attributions = []
        data_m = data[(data['method'] == 'grad') | (data['method'] == 'gxi') | (data['method'] == 'ig')]
        for attr in data_m['attributions']:
            tmp_a = np.round(np.array([i[0] for i in attr[0][0]]), 3)
            tmp_b = np.round(np.array([i[0] for i in attr[1][0]]), 3)
            tmp_c = np.round(np.array([i for i in attr[2][0]]), 3)

            attributions.append([tmp_a, tmp_b, tmp_c])

        table_data = attributions
        header = ["Parameter", "A", "b", "C"]
        first_col = ["Grad", "GxI", "IG-nz", "IG-av"]

        cap = f"Attributions for the optimal solution of the knapsack program, case {case}"
        cap_short = f"Attributions for the knapsack, optimal solution, case {case}"
        label = f"attr_{problem}_opt_c{case}"
        col_type = "c"

    elif problem == 'mf':
        attributions = []
        data_m = data[(data['method'] == 'grad') | (data['method'] == 'gxi') | (data['method'] == 'ig')]
        for attr in data_m['attributions']:
            tmp0_x1 = np.round(np.array(attr[0][0]), 3)
            tmp0_x2 = np.round(np.array(attr[0][1]), 3)
            tmp0_x3 = np.round(np.array(attr[0][2]), 3)
            tmp0_x4 = np.round(np.array(attr[0][3]), 3)
            tmp0_x5 = np.round(np.array(attr[0][4]), 3)
            tmp0_x6 = np.round(np.array(attr[0][5]), 3)
            tmp0_x7 = np.round(np.array(attr[0][6]), 3)
            tmp1_x1 = np.round(np.array(attr[1][0]), 3).reshape(-1, 1)
            tmp1_x2 = np.round(np.array(attr[1][1]), 3).reshape(-1, 1)
            tmp1_x3 = np.round(np.array(attr[1][2]), 3).reshape(-1, 1)
            tmp1_x4 = np.round(np.array(attr[1][3]), 3).reshape(-1, 1)
            tmp1_x5 = np.round(np.array(attr[1][4]), 3).reshape(-1, 1)
            tmp1_x6 = np.round(np.array(attr[1][5]), 3).reshape(-1, 1)
            tmp1_x7 = np.round(np.array(attr[1][6]), 3).reshape(-1, 1)
            tmp2_x1 = np.round(np.array(attr[2][0]), 3).reshape(-1, 1)
            tmp2_x2 = np.round(np.array(attr[2][1]), 3).reshape(-1, 1)
            tmp2_x3 = np.round(np.array(attr[2][2]), 3).reshape(-1, 1)
            tmp2_x4 = np.round(np.array(attr[2][3]), 3).reshape(-1, 1)
            tmp2_x5 = np.round(np.array(attr[2][4]), 3).reshape(-1, 1)
            tmp2_x6 = np.round(np.array(attr[2][5]), 3).reshape(-1, 1)
            tmp2_x7 = np.round(np.array(attr[2][6]), 3).reshape(-1, 1)

            attributions.append([tmp0_x1, tmp1_x1, tmp2_x1])
            attributions.append([tmp0_x2, tmp1_x2, tmp2_x2])
            attributions.append([tmp0_x3, tmp1_x3, tmp2_x3])
            attributions.append([tmp0_x4, tmp1_x4, tmp2_x4])
            attributions.append([tmp0_x5, tmp1_x5, tmp2_x5])
            attributions.append([tmp0_x6, tmp1_x6, tmp2_x6])
            attributions.append([tmp0_x7, tmp1_x7, tmp2_x7])

        table_data = attributions
        header = ["Method", "A", "b", "c"]
        first_col = ["Grad $E_1$", "Grad $E_2$", "Grad $E_3$", "Grad $E_4$", "Grad $E_5$", "Grad $E_6$", "Grad $E_7$",
                     "GxI $E_1$", "GxI $E_2$", "GxI $E_3$", "GxI $E_4$", "GxI $E_5$", "GxI $E_6$", "GxI $E_7$",
                     "IG-nz $E_1$", "IG-nz $E_2$", "IG-nz $E_3$", "IG-nz $E_4$", "IG-nz $E_5$", "IG-nz $E_6$", "IG-nz $E_7$",
                     "IG-as $E_1$", "IG-as $E_2$", "IG-as $E_3$", "IG-as $E_4$", "IG-as $E_5$", "IG-as $E_6$", "IG-as $E_7$"]

        cap = f"Attributions for the optimal solution of the {problem_dict[problem]}, case {case}"
        cap_short = f"Attributions for the {problem_dict[problem]}, optimal solution, case {case}"
        label = f"attr_{problem}_opt_c{case}"
        col_type = "c"

    elif problem == 'sp':
        attributions = []
        data_m = data[(data['method'] == 'grad') | (data['method'] == 'gxi') | (data['method'] == 'ig')]
        for attr in data_m['attributions']:
            tmp0_x1 = np.round(np.array(attr[0][0][0]), 3)
            tmp0_x2 = np.round(np.array(attr[0][0][1]), 3)
            tmp0_x3 = np.round(np.array(attr[0][0][2]), 3)
            tmp0_x4 = np.round(np.array(attr[0][0][3]), 3)
            tmp0_x5 = np.round(np.array(attr[0][0][4]), 3)
            tmp0_x6 = np.round(np.array(attr[0][0][5]), 3)
            tmp1_x1 = np.round(np.array(attr[1][0][0]), 3).reshape(-1, 1)
            tmp1_x2 = np.round(np.array(attr[1][0][1]), 3).reshape(-1, 1)
            tmp1_x3 = np.round(np.array(attr[1][0][2]), 3).reshape(-1, 1)
            tmp1_x4 = np.round(np.array(attr[1][0][3]), 3).reshape(-1, 1)
            tmp1_x5 = np.round(np.array(attr[1][0][4]), 3).reshape(-1, 1)
            tmp1_x6 = np.round(np.array(attr[1][0][5]), 3).reshape(-1, 1)
            tmp2_x1 = np.round(np.array(attr[2][0][0]), 3).reshape(-1, 1)
            tmp2_x2 = np.round(np.array(attr[2][0][1]), 3).reshape(-1, 1)
            tmp2_x3 = np.round(np.array(attr[2][0][2]), 3).reshape(-1, 1)
            tmp2_x4 = np.round(np.array(attr[2][0][3]), 3).reshape(-1, 1)
            tmp2_x5 = np.round(np.array(attr[2][0][4]), 3).reshape(-1, 1)
            tmp2_x6 = np.round(np.array(attr[2][0][5]), 3).reshape(-1, 1)

            attributions.append([tmp0_x1, tmp1_x1, tmp2_x1])
            attributions.append([tmp0_x2, tmp1_x2, tmp2_x2])
            attributions.append([tmp0_x3, tmp1_x3, tmp2_x3])
            attributions.append([tmp0_x4, tmp1_x4, tmp2_x4])
            attributions.append([tmp0_x5, tmp1_x5, tmp2_x5])
            attributions.append([tmp0_x6, tmp1_x6, tmp2_x6])

        table_data = attributions
        header = ["Method", "A", "b", "c"]
        first_col = ["Grad $E_1$", "Grad $E_2$", "Grad $E_3$", "Grad $E_4$", "Grad $E_5$", "Grad $E_6$",
                     "GxI $E_1$", "GxI $E_2$", "GxI $E_3$", "GxI $E_4$", "GxI $E_5$", "GxI $E_6$",
                     "IG-nz $E_1$", "IG-nz $E_2$", "IG-nz $E_3$", "IG-nz $E_4$", "IG-nz $E_5$", "IG-nz $E_6$",
                     "IG-as $E_1$", "IG-as $E_2$", "IG-as $E_3$", "IG-as $E_4$", "IG-as $E_5$", "IG-as $E_6$"]

        cap = f"Attributions for the optimal solution of the {problem_dict[problem]}, case {case}"
        cap_short = f"Attributions for the {problem_dict[problem]}, optimal solution, case {case}"
        label = f"attr_{problem}_opt_c{case}"
        col_type = "c"

    print_table(table_data, cap_short, cap, label, header, True, first_col, col_type=col_type)


print_attributions_opt('sp', 1)
print_attributions_opt('sp', 2)
