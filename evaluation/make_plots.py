import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# visualization of ks case 4
def vis_ks_4():
    items, limit = config.KS_CASE4['input']
    items = items.T
    values = items[0].reshape(-1, 1)
    weights = items[1].reshape(-1, 1)
    value_weight_ratio = np.divide(values, weights)
    value_weight_ratio = np.clip(value_weight_ratio, 0, 5).reshape(-1, 1)

    norm_ratio = value_weight_ratio / 5
    norm_ratio = norm_ratio.reshape(-1, 1)

    # attributions ks4
    optimal_solution = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]

    grad = [0.0015, 0.0016, 0.0018, 0.0, 0.0016, -0.0002, 0.0018, 0.0018, 0.0017, 0.0017, 0.0018, 0.0017, 0.0016, 0.0008, 0.0018, 0.0018, 0.0, 0.0016, 0.0018, 0.0017, 0.0004, -0.0001, 0.001, 0.0017, 0.0018, 0.0016, 0.0015, 0.0015, 0.0018, 0.0001, 0.0017, 0.0017, 0.0017, 0.0004, 0.0, 0.0018, -0.0002, 0.0016, 0.0018, 0.0018, 0.0016, 0.0018, 0.0017, 0.0016, 0.0017, 0.0004, 0.0016, 0.0017, 0.0017, 0.0001]
    gxi = [0.0014, 0.001, 0.0003, 0.0, 0.0011, -0.0002, 0.0004, 0.0003, 0.0009, 0.0005, 0.0002, 0.0006, 0.0012, 0.0004, 0.0001, 0.0003, 0.0, 0.0013, 0.0002, 0.0007, 0.0002, -0.0001, 0.0007, 0.0009, 0.0003, 0.0012, 0.0014, 0.0014, 0.0003, 0.0, 0.0005, 0.0006, 0.0009, 0.0003, 0.0, 0.0004, -0.0002, 0.0012, 0.0001, 0.0002, 0.001, 0.0001, 0.0006, 0.001, 0.0008, 0.0003, 0.0009, 0.0008, 0.0007, 0.0]
    ig_nz = [0.0014, 0.001, 0.0003, 0.0, 0.0011, -0.0002, 0.0004, 0.0003, 0.0009, 0.0005, 0.0002, 0.0006, 0.0012, 0.0004, 0.0001, 0.0003, 0.0, 0.0013, 0.0002, 0.0007, 0.0002, -0.0001, 0.0007, 0.0009, 0.0003, 0.0012, 0.0014, 0.0014, 0.0003, 0.0, 0.0005, 0.0006, 0.0009, 0.0003, 0.0, 0.0003, -0.0002, 0.0012, 0.0001, 0.0002, 0.001, 0.0001, 0.0006, 0.001, 0.0008, 0.0002, 0.0009, 0.0008, 0.0007, 0.0]
    ig_as = [0.0013, 0.0009, 0.0002, 0.0002, 0.0011, -0.0, 0.0003, 0.0002, 0.0008, 0.0004, 0.0002, 0.0006, 0.0012, 0.0004, -0.0, 0.0002, 0.0002, 0.0012, 0.0001, 0.0007, 0.0003, 0.0001, 0.0007, 0.0008, 0.0003, 0.0012, 0.0013, 0.0014, 0.0002, 0.0001, 0.0004, 0.0006, 0.0008, 0.0004, 0.0002, 0.0003, -0.0, 0.0011, 0.0, 0.0001, 0.001, 0.0, 0.0005, 0.001, 0.0007, 0.0003, 0.0009, 0.0008, 0.0007, 0.0001]
    gca = [0.0, 0.4732, 0.1306, 0.0, 0.3151, -0.0, 0.807, 0.1564, 0.1046, 0.3208, 0.5864, 0.1555, 0.1446, 0.0, 0.567, 0.5821, 0.0, 0.6542, 0.2792, 0.4846, 0.0, 0.0, 0.0, 0.112, 0.5213, 0.6835, 0.565, 0.2234, 0.0631, 0.0, 0.2775, 0.6921, 0.0558, 0.0, 0.0, 0.7468, 0.0, 0.4208, 0.7459, 0.333, 0.6043, 0.3055, 0.1998, 0.4708, 0.6875, 0.0, 0.5357, 0.5713, 0.3524, 0.0]

    optimal_solution = np.array(optimal_solution).reshape(-1, 1)
    grad = np.array(grad).reshape(-1, 1)
    gxi = np.array(gxi).reshape(-1, 1)
    ig_nz = np.array(ig_nz).reshape(-1, 1)
    ig_as = np.array(ig_as).reshape(-1, 1)
    gca = np.array(gca).reshape(-1, 1)

    min_grad, max_grad = np.amin(grad), np.amax(grad)
    min_gxi, max_gxi = np.amin(gxi), np.amax(gxi)
    min_ignz, max_ignz = np.amin(ig_nz), np.amax(ig_nz)
    min_igas, max_igas = np.amin(ig_as), np.amax(ig_as)
    min_gca, max_gca = np.amin(gca), np.amax(gca)

    gxi = (gxi - min_gxi) * (1/(max_gxi - min_gxi))
    grad = (grad - min_grad) * (1/(max_grad - min_grad))
    ig_nz = (ig_nz - min_ignz) * (1/(max_ignz - min_ignz))
    ig_as = (ig_as - min_igas) * (1/(max_igas - min_igas))
    gca = (gca - min_gca) * (1/(max_gca - min_gca))

    # visualize the problem data
    # prob_data = np.hstack([values, weights, norm_ratio, optimal_solution])
    # labels_x_prob = ['value', 'weight', 'n-ratio', 'solution']
    #
    # ax = sns.heatmap(prob_data, linewidth=0.5, cmap='viridis', xticklabels=labels_x_prob, yticklabels=[])
    # plt.show()
    # plt.clf()

    # visualize all data
    all_values = np.hstack([values, weights, norm_ratio, optimal_solution, grad, gxi, ig_nz, ig_as, gca])
    labels_x_prob = ['val', 'wei', 'n-rat', 'sol', 'Grad', 'GxI', 'IG-nz', 'IG-as', 'GCA']

    ax = sns.heatmap(all_values, linewidth=0.5, xticklabels=labels_x_prob, yticklabels=[]) # cmap='viridis',
    plt.show()

    # compare the gxi attributions with the input characteristics (each input is multiplied with the optimal solution,
    # resulting in a zero characteristic if the item is not chosen
    weight_t_opt = np.multiply(weights, optimal_solution)
    value_t_opt = np.multiply(values, optimal_solution)
    norm_t_opt = np.multiply(norm_ratio, optimal_solution)

    show_gxi = np.hstack([value_t_opt, weight_t_opt, gxi, norm_t_opt])
    labels_gxi = ['value', 'weight', 'GxI', 'n-ratio']

    ax = sns.heatmap(show_gxi, linewidth=0.5, xticklabels=labels_gxi, yticklabels=[]) # cmap='viridis',
    plt.show()

    # compare the grad attributions with the input characteristics
    # show_gxi = np.hstack([value_t_opt, weight_t_opt, grad, optimal_solution])
    # labels_gxi = ['value', 'weight', 'grad', 'solution']
    #
    # ax = sns.heatmap(show_gxi, linewidth=0.5, cmap='viridis', xticklabels=labels_gxi, yticklabels=[])
    # plt.show()

    # compare the gca and gxi attributions with the input characteristics
    show_gxi = np.hstack([value_t_opt, gxi, weight_t_opt, norm_t_opt, value_t_opt, gca, weight_t_opt])
    labels_gxi = ['value', 'GxI', 'weight', 'n-ratio', 'value', 'GCA', 'weight']

    ax = sns.heatmap(show_gxi, linewidth=0.0, xticklabels=labels_gxi, yticklabels=[]) # cmap='viridis',
    plt.show()


def plot_comprehensibility():
    # load the data and select the third case of the maximum flow problem with gxi attributions
    data = pd.read_json("results/mf_grad_gxi_ig_gc.json", orient='split')
    data = data[data['case'] == 3]
    data = data[data['method'] == 'gc']
    data = data[data['eval_func'] == 'opt'].reset_index()

    attributions = np.array(data['attributions'][0], dtype=float)

    # replace the none values through zeros
    attributions = np.nan_to_num(attributions)

    ax = sns.heatmap(-attributions, linewidth=0.001, xticklabels=[], yticklabels=[])
    plt.show()

plot_comprehensibility()

