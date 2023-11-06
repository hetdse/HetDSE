import copy
from copy import deepcopy

import numpy as np

from simulation_metrics import get_var_list_index, get_dataset
from config import *


def is_pareto_efficient_dumb(sample_origin, costs_origin, return_index=0):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    # print(f"costs.sum(1)={costs.sum(1).argsort()}")
    # print(f"sample_origin={sample_origin}, costs_origin={costs_origin}")
    index = costs_origin.sum(1).argsort()
    # print(f"index={index}")
    costs = deepcopy(costs_origin)[index]
    # print(f"origin costs={costs}")
    # print(f"sample_origin={sample_origin}")
    sample = np.array(deepcopy(sample_origin))[index]
    # print(f"sample_origin={sample}, costs_origin={costs}")
    # print(f"costs={costs}")
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i in range(costs.shape[0]):
        # print(f"costs.shape={costs.shape}")
        # process each point in turn
        n = costs.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        # print(f"costs[i+1:]={costs[i+1:]}")
        # print(f"costs[i]={costs[i]}")
        # print(f"(costs[i+1:] < costs[i])={(costs[i+1:] < costs[i])}")
        is_efficient[i + 1:n] = (costs[i + 1:] < costs[i]).any(1)
        is_efficient[i] = True
        # keep points undominated so far
        # print(f"i={i}, n={n}, is_efficient={is_efficient[:n]}")
        costs = costs[is_efficient[:n]]
        # print(f"costs={costs}")
        sample = sample[is_efficient[:n]]
        index = index[is_efficient[:n]]
    if False:
        import matplotlib.pyplot as plt
        plt.scatter(costs_origin[:, 0:1], costs_origin[:, 1:2], c="gray", s=10)
        plt.scatter(costs[:, 0:1], costs[:, 1:2], c="red", marker="x", s=16)
        plt.show()
        exit(1)
    if return_index:
        return sample, costs, index
    else:
        return sample, costs


def scale_for_hv(obj_values, ref_point):
    obj_values_scale = copy.deepcopy(obj_values)
    obj_values_scale[:, 0] = obj_values[:, 0] / ref_point[0]
    obj_values_scale[:, 1] = obj_values[:, 1] / ref_point[1]
    return obj_values_scale


def get_pareto_optimality_from_file(filename):
    file = open(filename, 'r')
    pareto_points_x = []
    pareto_points_y = []
    points_config = []
    sample_num_i = 0
    for point in file:
        point_str = point.split(' ')
        if 2 < len(point_str):
            cpi = float(point_str[0])
            #if MULTIOBJ_DELAYXENERGY == multiobj_mode:
            #    metric_
            pareto_points_x.append(cpi)
            power = float(point_str[1])
            pareto_points_y.append(power)
            points_config.append(point_str[2].strip('\n'))
        elif 1 == len(point_str):
            sample_num_i = int(point.strip('\n'))
    file.close()
    return [pareto_points_x, pareto_points_y, points_config, sample_num_i]


def generate_pareto_optimality_perfect_filename(case_name, multiobj_mode):
    pareto_optimality_perfect_filename = 'real_pareto_optimality/'  # "../"
    # global case_name
    if 2304 != N_SPACE_SIZE:
        pareto_optimality_perfect_filename += str(N_SPACE_SIZE) + '_'
    pareto_optimality_perfect_filename += case_name + "_"
    if MULTIOBJ_DELAYXENERGY == multiobj_mode:
        pareto_optimality_perfect_filename += 'delay_energy-pareto_optimality_perfect.txt'
    elif MULTIOBJ_CPIXAREA == multiobj_mode:
        pareto_optimality_perfect_filename += 'area-pareto_optimality_perfect.txt'
    else:
        pareto_optimality_perfect_filename += 'power-pareto_optimality_perfect.txt'
    #print(f"handling {pareto_optimality_perfect_filename}")
    return pareto_optimality_perfect_filename


def get_pareto_optimality_from_file_ax_interface(case_name, multiobj_mode):
    pareto_optimality_perfect_filename = generate_pareto_optimality_perfect_filename(case_name, multiobj_mode=multiobj_mode)
    [real_pareto_points_x, real_pareto_points_y, real_pareto_points_config,
     sample_num_i] = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)
    config_vector_list = []
    for config in real_pareto_points_config:
        # print(config)
        config_vector = transfer_version_to_var(config)
        config_vector_list.append(config_vector)
    # print(config_vector_list)
    return [real_pareto_points_x, real_pareto_points_y, config_vector_list]


def cal_pareto_optimality_to_file(case_name, multiobj_mode):
    pareto_optimality_perfect_filename = generate_pareto_optimality_perfect_filename(case_name, multiobj_mode=multiobj_mode)

    from simulation_metrics import get_dataset
    version_list, y = get_dataset(case_name=case_name, multiobj_mode=multiobj_mode)

    real_pareto_points_x, real_pareto_points_y = is_pareto_efficient_dumb(np.asarray(version_list), np.asarray(y))

    file = open(pareto_optimality_perfect_filename, 'w')
    #file.write(f"{multiobj_mode_x_label} {multiobj_mode_y_label} version \n")  # header
    file.write(f"1 \n")  # header - simpoint
    for each_x, each_y in zip(real_pareto_points_x, real_pareto_points_y):
        file.write(f"{each_y[0]} {each_y[1]} {each_x}\n")
    file.close()


def transfer_version_to_var(config):
    #print(f"{len(config)}")
    config_vector = [int(config[var_index]) for var_index in get_var_list_index()]     
    return config_vector


def plot_histogram(case_name, mode='CPI', plot=True):
    version_list, y = get_dataset(case_name=case_name, multiobj_mode=0)
    if 'CPI' == mode:
        data = np.asarray(y)[:,0]
    elif 'Power' == mode:
        data = np.asarray(y)[:,1]

    response_transfer_method = 'box-cox'
    #response_transfer_method = 'StandardScaler'
    data, transfer_param = response_transfer(y=data, method=response_transfer_method)
    value_range[0] = min(value_range[0], np.min(data))
    value_range[1] = max(value_range[1], np.max(data))
    #print(np.shape(data))
    #density=False (default), return the num; True, return probility density
    # len(bin_edges) == len(hist) + 1
    #hist, bin_edges = np.histogram(a=y[:,0], bins=20)

    import matplotlib.pyplot as plt
    # matplotlib.axes.Axes.hist() interface
    bins = 20 #'auto'
    freq, bins, patches = plt.hist(x=data, range=(value_range[0],value_range[1]), bins=bins, color='darkblue', alpha=0.7, rwidth=0.85)
    freq /= np.sum(freq)
    if plot:
        #print(freq)
        #print(bins)
        freq_sum = np.sum(freq)
        #print(f"freq_sum={freq_sum}, bins = {len(freq)}")
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(mode + ' histogram of ' + case_name)
        maxfreq = freq.max()
        plt.xticks(bins, fontsize=5, rotation=90)
        #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        fig_name = "fig/histogram/" + mode + "_histogram_" + case_name + '_' +response_transfer_method
        fig_name += ".png"
        print(f"fig_name={fig_name}")
        plt.savefig(fig_name)
        plt.close()
    return freq, bins

cpi_range = [99, 0]
power_range = [99, 0]
value_range = [99, 0]
#cpi_range=[0.227356, 16.528889]
#power_range=[0.660114, 8.74837]


def response_transfer_inverse(y_transform, method='box-cox', transfer_param=None):
    if 'box-cox' == method:
        from scipy.special import inv_boxcox
        lmbda, _ = transfer_param
        y = inv_boxcox(y_transform, lmbda)
    elif 'StandardScaler' == method:
        _, standar_scaler = transfer_param
        y_transform = np.asarray(y_transform).reshape((-1, 1))
        y = standar_scaler.inverse_transform(y_transform).ravel()
    return y


def response_transfer(y, method='box-cox'):
    if 'box-cox' == method:
        from scipy.stats import boxcox
        y_transform, lmbda = boxcox(x=y)
        transfer_param = lmbda, None
    elif 'StandardScaler' == method:
        from sklearn.preprocessing import StandardScaler
        standar_scaler = StandardScaler()
        y = np.asarray(y).reshape((-1, 1))
        standar_scaler.fit(y)
        y_transform = standar_scaler.transform(y).ravel()
        transfer_param = None, standar_scaler
    return y_transform, transfer_param


if __name__ == '__main__':
    for case_name_config in case_names:
        cal_pareto_optimality_to_file(case_name=case_name_config, multiobj_mode=multiobj_mode)
    if 0:
        CPI_histogram = {}
        #for case_name in case_names:
        for case_name_iter in ["500.1-refrate-1"]:
            freq, bins = plot_histogram(case_name=case_name_iter, mode='CPI', plot=True)
            #CPI_histogram[case_name] = freq / 2304
            freq, bins = plot_histogram(case_name=case_name_iter, mode='Power', plot=True)
        print(f"value_range={value_range}")
        #print(f"power_range={power_range}")
