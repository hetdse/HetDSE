import copy
import itertools
import sys
import time
from datetime import datetime
import os
import numpy as np
import sklearn.metrics

from HeterDSE_config import get_HeterDSE_config
from config import *
from get_model import get_surrogate_model
from problem_model import Problem_Model, single_core_space
from simulation_metrics import *
from skopt import Space
from skopt.space import Integer
from schedule_function import schedule

#if 'LGX' in hostname:
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _combination(_handle, items, r):
    #print(f"items #= {len(items)}, r={r}")
    if r <= 0:
        yield []
    elif 1 == r:
        for item in items:
            yield [item]
    else:
        for i, item in enumerate(items):
            #print(f"items #= {len(items)}, r={r}, i={i}")
            this_one = copy.deepcopy([item])
            n = r - 1
            for cc in _combination(_handle, _handle(items, i), n):
                #print(f"cc={cc}")
                yield copy.deepcopy(this_one + cc)


def combinations(items, r):
    def afterIthItem(items, i):
        return items[i+1:]
    return _combination(afterIthItem, items, r)


def get_hmp_composement_space(core_space, multiobj_config=None):
    hmp_composement_space = []
    for var_index, var_iter in enumerate(core_space):
        hmp_composement_space.append(
            Integer(low=0, high=1 + int(AREA_CONSTRAIN / area_all[var_to_version(var_iter)[:-1]]), prior='uniform',
                    transform='identity'))
    if multiobj_config is not None and multiobj_config['sche_explore']:
        hmp_composement_space.append(
            Integer(low=0, high=1 + 8, prior='uniform', transform='identity'))
    return hmp_composement_space


def get_all_hmp(core_space):
    X = []
    result_map = {
        'pareto_filter': 1,
        'hmp_configs_num_pareto_filter': 0
    }
    hmp_configs_num_pareto_filter = 0
    if 1:
        core_area_space = [area_all[var_to_version(core_i)[:-1]] for core_i in core_space]
        hmp_space_args = [i for i in range(len(core_space))]
        for core_type_num in range(1, 1 + MAX_CORE_TYPES):
            #design_space_permutation = itertools.combinations(hmp_space_args, r=core_type_num)
            for each_core_type_space in combinations(hmp_space_args, r=core_type_num):
                hmps = []
                if 1 == len(core_space):
                    each_core_type_space = np.asarray(each_core_type_space)
                # print(f"each_core_type_space={each_core_type_space}")
                var_list_range2 = [[] for i in range(core_type_num)]
                '''
                for var_index, var_iter in enumerate(each_core_type_space):
                    var_list_range2[var_index] = [(1 + value) for value in
                                                  range(int(AREA_CONSTRAIN / core_area_space[var_iter]))]
                '''
                area_each_core_type_space = [area_all[var_to_version(core_space[var_iter])[:-1]] for var_iter in each_core_type_space]
                rest_area = (AREA_CONSTRAIN - sum(area_each_core_type_space))
                if rest_area < 0:
                    continue
                for var_index, var_iter in enumerate(each_core_type_space):
                    num_max = 1 + int(rest_area / area_each_core_type_space[var_index])
                    var_list_range2[var_index] = [value for value in range(num_max, 0, -1)]

                hmp_space_args2 = (var_list_range2[i] for i in range(core_type_num))
                core_type_space_product2 = itertools.product(*hmp_space_args2)
                for each_hmp in core_type_space_product2:
                    # print(f"each_hmp={each_hmp}")
                    if 1 == core_type_num:
                        each_hmp = np.asarray(each_hmp)
                    hmp = [0 for _ in range(len(core_space))]
                    for each_core_type_index, each_core_num in zip(each_core_type_space, each_hmp):
                        hmp[each_core_type_index] = each_core_num
                    if hmp_is_under_constrain(core_space, hmp, core_area_space):
                        if result_map['pareto_filter']:
                            hmp_configs_num_pareto_filter_flag = False
                            for each in hmps:
                                if (np.asarray(each_hmp) <= np.asarray(each)).all():
                                    hmp_configs_num_pareto_filter_flag = True
                                    #print(f"each_hmp={each_hmp} filtered")
                                    break
                            if hmp_configs_num_pareto_filter_flag:
                                hmp_configs_num_pareto_filter += 1
                                continue
                        #hmp_configs_num += 1
                        X.append(hmp)
                        hmps.append(each_hmp)
                        # print(f"hmp={hmp}")
        print(f"get_all_hmp: hmp_configs_num_pareto_filter={hmp_configs_num_pareto_filter}")
        result_map['hmp_configs_num_pareto_filter'] = hmp_configs_num_pareto_filter
    else:
        var_list_range = [[] for i in range(len(core_space))]
        for var_index, var_iter in enumerate(core_space):
            var_list_range[var_index] = [value for value in
                                         range(1+int(AREA_CONSTRAIN / area_all[var_to_version(var_iter)[:-1]]))]
            # print(f"var_list[{var_list[var_index]}] # = {var_iter}")
        var_max = [(var_range[-1]) for var_range in var_list_range]

        hmp_space_args = (var_list_range[i] for i in range(len(core_space)))
        hmp_space_size = np.product(var_max)
        print(f"BruteForce hmp_space_size={hmp_space_size}")
        if hmp_space_size < 1:
            print(f"hmp_space_size={hmp_space_size}, error because its too large to store in var")
        if 1e5 < hmp_space_size:
            print(f"1e5 < hmp_space_size={hmp_space_size}, exit because of memory error ")
            exit(0)
        design_space_product = itertools.product(*hmp_space_args)
        for each in design_space_product:
            X.append(each)

    print(f"get_all_hmp: single core space size = {len(core_space)}, hmp space size = {len(X)} ")
    #print(f"get_all_hmp: space = {X}")
    return X, result_map


def get_by_kmeans(X, Y, n_clusters, selection, random_state=0):

    if 'kmeansy-scale' == selection:
        points = np.asarray(copy.deepcopy(Y))
    elif 'kmeansy' == selection:
        points = np.asarray(copy.deepcopy(Y))
    elif 'kmeansx' == selection:
        points = np.asarray(copy.deepcopy(X))
    else:
        print(f"get_by_kmeans no def selection={selection}")
        exit(0)

    from pyclustering.utils.metric import type_metric, distance_metric

    feature_weight = [1 for _ in range(len(points[0]))]
    user_function = lambda point1, point2: np.sum(np.square(point1 - point2) * feature_weight)
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    samples = []
    from pyclustering.cluster.kmeans import kmeans
    X_index = np.arange(0, len(points))
    import random
    random.shuffle(X_index)
    start_centers = np.asarray(points)[X_index[:n_clusters]]
    kmeans_instance = kmeans(points, start_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    # clusters = kmeans_instance.get_clusters()
    clusters = kmeans_instance.get_centers()

    for each_center in clusters:
        x_distance = []
        for x_id in range(len(points)):
            x_distance.append([x_id, np.sum(np.square(points[x_id] - each_center) * feature_weight)])
        x_distance_sorted = np.asarray(sorted(x_distance, key=lambda obj_values: obj_values[1]))
        #print(f"x_distance_sorted={x_distance_sorted}")
        equal_flag = (np.abs(x_distance_sorted[:,1] - x_distance_sorted[0,1])) < 1e-4
        equal_index = np.argmin(equal_flag)
        #print(f"equal_flag={equal_flag}, equal_index={equal_index}")
        x_selected = int(x_distance_sorted[int(random.randint(0,equal_index))][0])
        #print(f"x_distance_sorted={x_distance_sorted[:10]}")
        #x_selected = np.argmin(x_distance)
        samples.append(X[x_selected])
        #samples.append(X[x_selected].tolist())

    if 0:
        #colors = ['red', 'green', 'blue', 'black']
        #colors = ['red', 'green', 'blue', 'black']
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3.7))
        fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=0.2, hspace=None)
        ax = fig.add_subplot(111)

        plt.rcParams['pdf.fonttype'] = 42
        # plt.rcParams['ps.fonttype'] = 42

        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.size'] = '10'
        # plt.rcParams['font.weight'] = 'bold'

        font = {'family': 'Times New Roman',
                'weight': 'bold',
                'size': 10,
                }

        fontsize = 20
        font2 = font
        font2['weight'] = 'bold'
        font2['size'] = fontsize
        for cluster_i, cluster in enumerate(kmeans_instance._kmeans__clusters):
            plt.scatter(points[cluster][:,0], points[cluster][:,1], label=str(cluster_i), s=2)
            ax.set_xlabel(multiobj_mode_x_label, fontsize=fontsize, font=font, labelpad=10)
            ax.set_ylabel(multiobj_mode_y_label, fontsize=fontsize, font=font, labelpad=10)
            #plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)            
            #plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)
            #plt.set_yticklabels(['%.1f'%(i/4) for i in range(0,5)], fontsize=fontsize, font=font)            
            #plt.xlabel(var_list[0])
            #plt.ylabel(var_list[1])
        plt.savefig('fig/design_point_selection_kmeans' + '.png')
    # print(f"samples={samples}")
    return samples


def design_point_selection(core_space, core_space_metrics, program_queue, target_core_num, selection='kmeans', models=None):
    if 'kmeansx' == selection:
        selected_core_space = get_by_kmeans(X=core_space, Y=None, n_clusters=target_core_num, selection=selection)
    elif 'kmeansy' in selection:
        selected_core_space = get_by_kmeans(X=core_space, Y=core_space_metrics, n_clusters=target_core_num, selection=selection)
    elif 'LUCIE' in selection:
        from SOTA_LUCIE import SOTA_LUCIE
        LUCIE = SOTA_LUCIE(core_space=core_space, benchmarks=program_queue)
        selected_core_space = LUCIE.get_LUCIE(target_core_num=target_core_num)
    elif 'Clustering' in selection:
        from SOTA_Clustering import SOTA_Clustering
        Clustering = SOTA_Clustering(core_space=core_space, benchmarks=program_queue, selection=selection)
        selected_core_space = Clustering.get_Clustering(target_core_num=target_core_num, models=models)
    elif 'Random' == selection:
        X_index = np.arange(0, len(core_space))
        import random
        random.shuffle(X_index)
        selected_core_space = np.asarray(core_space)[X_index[:target_core_num]]
        #print(f"selection={selection}: x_selecteds={X_index[:target_core_num]}")
        #exit(0)
    elif 'Hetsel' in selection:
        from compose_selection import compose_selection
        selected_core_space = compose_selection(core_space, program_queue, selection, target_core_num, random_state=0, models=models)
    else:
        print(f"design_point_selection: no def selection={selection}")
        exit(0)
    print(f"after design_point_selection {selection}, space size => {len(selected_core_space)}")
    #exit(0)
    return selected_core_space


def pareto_domain_filter(core_space_paretocrossover, models):
    filter_ids = []
    from get_real_pareto_frontier import is_pareto_efficient_dumb
    core_space_paretocrossover_y = []
    for each_x_id, each_x in enumerate(core_space_paretocrossover):
        try_predict_y = []
        for model in models:
            if 0 < len(model):
                model_1, model_2 = model
                for freq in range(len(SIMULATOR_CYCLES_PER_SECOND_map)):
                    each_x_copy = copy.deepcopy(each_x).tolist()
                    each_x_copy.append(freq)
                    try_predict_y += [model_1.predict([each_x_copy])[0], model_2.predict([each_x_copy])[0]]
        try_predict_y = np.asarray(try_predict_y)
        try_predict_y = [min(try_predict_y[b*4:(b+1)*4]) for b in range(len(models))]
        core_space_paretocrossover_y.append(try_predict_y)
    core_space_paretocrossover_y = np.asarray(core_space_paretocrossover_y)
    try_new_pareto_frontier_x, try_new_pareto_frontier_y = is_pareto_efficient_dumb(core_space_paretocrossover, core_space_paretocrossover_y)
    return try_new_pareto_frontier_x, try_new_pareto_frontier_y


def get_predict_core_space(file_name):
    print(f"get_predict_core_space: file_name={file_name}")
    core_space = []
    file = open(file_name,'r')
    for each in file:
        if 'selection' in each:
            time_selection = float(each.split(' ')[2])
        else:
            core_config = each[1:-2].split(',')
            core_config = [int(x) for x in core_config if x != '']
            core_space += [core_config]
    file.close()
    return core_space, time_selection


def get_hmp_data(hostname_tag, program_queue_name, tag, evaluation_index, arg_core_type_num=0):
    hmp_configs = []
    hmp_times = []
    hmp_energys = []
    result = {}
    result['hmp_configs_num'] = 0
    result['hmp_compose_space'] = 0
    result['core_space'] = 0
    result['origin_core_space_size'] = 0
    result['selected_core_space_size'] = 0
    result['schedule_result_database_num'] = 0
    result['time_used_map_single_core_model'] = 0
    result['core_type_num-' + str(0)] = {'end_ids': 0}
    filename = 'log_hmp' + hostname_tag + '/' + program_queue_name + '/' + tag + '.txt'
    if not os.path.exists(filename):
        return hmp_configs, hmp_times, hmp_energys, result
    #print(f"handling {filename}")
    file = open(filename, 'r')
    hmp_configs_following_flag = False
    best_each_core_space_continue = False
    for line_id, each_line in enumerate(file):
        try:
            if 'schedule_mode = ' in each_line:
                result['schedule_mode'] = each_line.split(' ')[2]
            elif 'core_space #=' == each_line[:len('core_space #=')]:
                result['core_space'] = int(each_line.split(' ')[2])
            elif 'origin_core_space_size #= ' in each_line:
                result['origin_core_space_size'] = int(each_line.split(' ')[2])
            elif 'selected_core_space_size #= ' in each_line:
                result['selected_core_space_size'] = int(each_line.split(' ')[2])
            elif 'history_f' in each_line:
                history_f_list = '[' + each_line.split('[')[1]
                result['history_f'] = eval(history_f_list)
                continue
            elif 'history_aimFunc' in each_line:
                history_aimFunc_list = '[' + each_line.split('[')[1]
                result['history_aimFunc'] = eval(history_aimFunc_list)
                continue
            elif 'history_schedule_result_database' in each_line:
                history_schedule_result_database_list = '[' + each_line.split('[')[1]
                result['history_schedule_result_database'] = eval(history_schedule_result_database_list)
                continue
            elif 'hmp_size_list' in each_line:
                history_hmp_size_list = '[' + each_line.split('[')[1]
                result['hmp_size_list'] = eval(history_hmp_size_list)
                continue
            elif 'hmp_metric_best_list' in each_line:
                history_hmp_metric_best_list = '[' + each_line.split('[')[1]
                result['hmp_metric_best_list'] = eval(history_hmp_metric_best_list)
                continue
            elif 'aimFunc #= ' in each_line:
                result['aimFunc'] = int(each_line.split(' ')[2])
            elif 'schedule_result_database #= ' in each_line:
                result['schedule_result_database_num'] = int(each_line.split(' ')[2])
            elif 'last_mean #= ' in each_line:
                last_mean = '[' + each_line.split('[')[1]
                result['last_mean'] = eval(last_mean)
            elif 'last_ucb #= ' in each_line:
                last_ucb = '[' + each_line.split('[')[1]
                result['last_ucb'] = eval(last_ucb)
            elif 'X_all_go #= ' in each_line:
                X_all_go = '[' + each_line.split('[')[1]
                result['X_all_go'] = eval(X_all_go)
            elif 'time_used =' in each_line:
                if ':' in each_line:
                    if 'day' in each_line:
                        #print(f"tofix: each_line={each_line}")
                        #result['Total Time'] = 0
                        each_line_splits = each_line.split(' ')
                        time_str = each_line_splits[4].split(':')
                        result['Total Time'] = ((int(each_line_splits[2]) * 24 + int(time_str[0]) * 60 + int(time_str[1])) * 60 +  int(time_str[2]))
                    else:
                        time_str = each_line.split(' ')[2].split(':')
                        result['Total Time'] = (int(time_str[0]) * 60 + int(time_str[1])) * 60 +  int(time_str[2])
                else:
                    result['Total Time'] = int(float(each_line.split(' ')[2]))
            elif 'time_used_map[single_core_model]' in each_line and 'HeterDSE' in tag:
                result['time_used_map_single_core_model'] = int(float(each_line.split(' ')[2]))
            elif 'hmp_configs # = ' in each_line:
                result['hmp_configs_num'] = int(each_line.split(' ')[3])
                if 'HeterDSE' in tag:
                    result['hmp_size_list'] = np.arange(1, result['hmp_configs_num']+1)
                #elif 'SA' in tag or 'GA' in tag:
                #    result['hmp_size_list'] = result['hmp_size_list']
                #    result['hmp_metric_best_list'] = result['hmp_metric_best_list']
                    #result['hmp_size_list'] = result['history_aimFunc']
                    #result['hmp_metric_best_list'] = result['history_f']
                hmp_configs_following_flag = False
                continue
            elif 'hmp_compose_space # =' in each_line:
                #result['hmp_configs_num'] = int(each_line.split(' ')[3])
                result['hmp_compose_space'] = int(each_line.split(' ')[3])
                #print(f"hmp_compose_space={result['hmp_compose_space']} for {tag}")
            elif 'hmp_configs = ' == each_line[:len('hmp_configs = ')]:
                hmp_configs_following_flag = True
                continue
            elif 'hmp_configs_num_pareto_filter =' in each_line:
                result['hmp_configs_num_pareto_filter'] = int(each_line.split(' ')[2])
            #elif 'time_used = ' in each_line:
            #    result['time_used'] = 0 #float(each_line.split(' ')[3])
            if best_each_core_space_continue:
                if 'best_hmp' in each_line:
                    continue
                elif 'best_core_types' in each_line:
                    best_each_core_space_continue = False
                    continue
                else:
                    continue
            if hmp_configs_following_flag:
                if 'core_type_num <= ' in each_line:
                    if 'best_each_core_space' in each_line:
                        core_type_num = int(each_line.split(' ')[2][:-1])
                        result['core_type_num-'+str(core_type_num)] = {'end_ids': len(hmp_times)}
                        best_each_core_space_continue = True
                    #if 2 == each_line.count('['):
                        #each_line_strs_1 = each_line.split(']')
                elif 'time_used =' in each_line:
                    result['Total Time'] = int(float(each_line.split(' ')[2]))
                    hmp_configs_following_flag = False
                else:
                    if (0 < each_line.count('[')):
                        each_line_strs_1 = each_line.split(']')
                        hmp_config = each_line_strs_1[0] + ']'
                    else:
                        each_line_strs_1 = each_line.split('}')
                        hmp_config = each_line_strs_1[0] + '}'
                    hmp_configs.append(hmp_config)
                    if len(each_line_strs_1) < 2:
                        continue
                    each_line_strs = each_line_strs_1[1].split(' ')
                    each_line_strs = [each for each in each_line_strs if each != '']
                    hmp_time = float(each_line_strs[0])
                    hmp_times.append(hmp_time)
                    hmp_energy = float(each_line_strs[1])
                    hmp_energys.append(hmp_energy)
        except:
            print(f'error: {filename} line {line_id} {each_line}')
            continue

    file.close()

    if result['selected_core_space_size'] != result['core_space']:
        if 0 != result['selected_core_space_size']:
            print(f"warining: tag={tag} core_space# {result['selected_core_space_size']} != {result['core_space']}")
        result['selected_core_space_size'] = result['core_space']
    if 'BruteForce' in tag:
        result['aimFunc'] = result['hmp_configs_num']
        result['schedule_result_database_num'] = result['hmp_configs_num']
        result['hmp_compose_space'] = result['hmp_configs_num']
        '''
        if 0 < arg_core_type_num:
            begin_ids = result['core_type_num-' + str(arg_core_type_num - 1)]['end_ids']
            end_ids = result['core_type_num-' + str(arg_core_type_num)]['end_ids']
            hmp_configs = hmp_configs[begin_ids:end_ids]
            hmp_times = hmp_times[begin_ids:end_ids]
            hmp_energys = hmp_energys[begin_ids:end_ids]
        '''
        if 0:
            filename = 'motivation_maxcorenum' + '-EI' + str(evaluation_index) + '.txt'
            file_bf = open(filename, 'a')
            for core_type_num in range(1, 1+MAX_CORE_TYPES):
                eval_metrics = []
                #begin_ids = result['core_type_num-' + str(core_type_num-1)]['end_ids']
                end_ids = result['core_type_num-' + str(core_type_num)]['end_ids']
                for each_id in range(0, end_ids):
                    eval_metrics.append(get_hmp_metrics(delay=hmp_times[each_id], energy=hmp_energys[each_id])[evaluation_index])
                best_id = np.argmin(eval_metrics)
                best_metric = eval_metrics[best_id]
                file_bf.write(f'%-15s %-100s %2d %6d %6.6f \n' % (program_queue_name, tag, core_type_num, end_ids, best_metric))
            file_bf.close()

    if 'HeterDSE' in tag:
        str_idx = tag.find('_N') + len('_N')
        schedule_result_database_num = int(tag[str_idx:str_idx+3])
        result['schedule_result_database_num'] = schedule_result_database_num

        metric_values = []
        for each_1, each_2 in zip(hmp_times, hmp_energys):
            metric_values.append(get_hmp_metrics(delay=each_1, energy=each_2)[evaluation_index])
        #metric_values = np.asarray(hmp_times) * np.asarray(hmp_energys)
        result['hmp_metric_best_list'] = [min(metric_values[:ids + 1]) for ids in range(len(metric_values))]
        #result['hmp_metric_best_list'] = metric_values
        #result['hmp_metric_best_list'] = [np.mean(np.sort(metric_values[:ids + 1])[:min(5,ids+1)]) for ids in range(len(metric_values))]

    return hmp_configs, hmp_times, hmp_energys, result


def get_HeterDSE_prediction_model(workload_i, program_queue_name, multiobj_config, model_load=False):
    #print(f"get_HeterDSE_prediction_model : workload_i={workload_i}")
    program_singlecore = program_queue_name, [case_names[workload_i]], [], workload_i

    surrogate_model_tag_singlecore = multiobj_config['metric_prediction_model']
    surrogate_model_config_singlecore = get_surrogate_model(surrogate_model_tag_singlecore)
    _, _, surrogate_model_dict_singlecore = surrogate_model_config_singlecore
    if smoke_test:
        multiobj_config['n_initial_points'] = 2
        multiobj_config['n_generation_points'] = 0
    else:
        multiobj_config['n_initial_points'] = N_SAMPLES_INIT if surrogate_model_dict_singlecore['model_is_iterative'] else N_SAMPLES_ALL
        multiobj_config['n_generation_points'] = N_SAMPLES_ALL - multiobj_config['n_initial_points']
    single_core_model_workload_i = Problem_Model(problem_space=single_core_space,
                                                 surrogate_model_tag=surrogate_model_tag_singlecore,
                                                 surrogate_model_config=surrogate_model_config_singlecore,
                                                 multiobj_config=multiobj_config,
                                                 program_queue_info=program_singlecore,
                                                 schedule_mode=None,
                                                 obj_info=(2, (metric_1, metric_2), None),
                                                 model_load=model_load,
                                                 tag='',                                                 
                                                 )

    global time_used_map
    
    if model_load:
        result = single_core_model_workload_i.run_from_model()
    else:
        result = single_core_model_workload_i.run()
        #time_used_map['register'] += result.used_map['register']
        #time_used_map['single_core_model'] += result.result_array_sample_map['time_valid'][-1][workload_i]
        #exit(0) only for model_data generation
    time_used_map['register'] = max(time_used_map['register'], result.used_map['register']) 
    time_used_map['single_core_model'] = max(time_used_map['single_core_model'], result.result_array_sample_map['time_valid'][-1][0])
    
    pareto_x = [x[:-1] for x in result.x]
    pareto_y = result.used_map['current_pareto_y']
    if multiobj_config['train_score_based']:
        pareto_x = [x[:-1] for x in result.result_array_sample_map['train_score_based_pareto_x']]
        pareto_y = result.result_array_sample_map['train_score_based_pareto_y']

    if multiobj_config['pareto_crossover']:
        global core_space_paretocrossover
        global core_space_paretocrossover_metrics
        #print(f"pareto_crossover enable : core_space_paretocrossover add #{len(pareto_x)}")
        if core_space_paretocrossover is not None:
            core_space_paretocrossover = np.append(core_space_paretocrossover, pareto_x, axis=0)
            core_space_paretocrossover_metrics = np.append(core_space_paretocrossover_metrics, pareto_y, axis=0)
        else:
            core_space_paretocrossover = pareto_x
            core_space_paretocrossover_metrics = pareto_y

    #print(f"pareto_x={pareto_x}")
    if multiobj_config['metric_prediction_model']:
        global models_all_workloads
        models_all_workloads[workload_i] = result.models
    #print(f"done get_HeterDSE_prediction_model #{workload_i}")
    #return 


core_space_paretocrossover = None
core_space_paretocrossover_metrics = None
time_used_map = {}
models_all_workloads = None

def hmp_evaluate(program_queue_info, core_space, core_space_metrics, tag, SOTA, exp_info, selection, target_core_num,
                 schedule_mode='Naive',
                 sche_evaluation_index=2):

    program_queue_name, program_queue, program_bitmap, program_queue_ids = program_queue_info
    startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")

    global time_used_map
    time_used_map = {}
    time_used_map['register'] = 0
    time_used_map['single_core_model'] = 0
    time_used_map['selection'] = 0
    model_load = True
    selection_load = False
    hmp_model_load = False
    #hmp_model_load = True

    if 'EER' in schedule_mode:
        use_eer = True
    else:
        use_eer = False

    exp_info_map = {}
    if core_space is not None:
        exp_info_map['origin_core_space_size'] = len(core_space)
    if selection and (core_space is not None):
        tag += '_sel-' + selection + '-' + str(target_core_num)
        if target_core_num <=  len(core_space):
            core_space = design_point_selection(core_space, core_space_metrics, program_queue, target_core_num=target_core_num, selection=selection)
            time_used_map['selection'] = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime
    if core_space is not None:
        exp_info_map['selected_core_space_size'] = len(core_space)

    if 'HeterDSE' in SOTA or 'SA' in SOTA:

        multiobj_config = get_HeterDSE_config(tag=SOTA)

        multiobj_config['exp_id'] = exp_info
        multiobj_config['sche_evaluation_index'] = sche_evaluation_index

        #surrogate_model_tag_hmp = 'MLP' if smoke_test else 'BagGBRT' #'AdaGBRT'
        #surrogate_model_tag_hmp = 'MLP' if smoke_test else 'RF_custom'
        surrogate_model_tag_hmp = 'MLP' if smoke_test else 'RF'
        #surrogate_model_tag_hmp = 'MLP' if smoke_test else 'GP'
        #surrogate_model_tag_hmp = 'GBRT'
        #surrogate_model_tag_hmp = 'PolyLinear'
        #surrogate_model_tag_hmp = 'smoke_test'

        surrogate_model_config_hmp = get_surrogate_model(surrogate_model_tag_hmp)
        _, _, surrogate_model_dict_hmp = surrogate_model_config_hmp
        surrogate_model_dict_hmp['initial_point_generator'] = 'random'

        #core_space_paretocrossover = None
        global core_space_paretocrossover
        global core_space_paretocrossover_metrics
        global models_all_workloads

        core_space_paretocrossover = None
        core_space_paretocrossover_metrics = None
        models_all_workloads = None

        # for single-core -exploration
        if multiobj_config['pareto_crossover'] or multiobj_config['metric_prediction_model']:
            print(f"metric_prediction_model enable")
            models_all_workloads = [[] for _ in range(len(case_names))]
            th = [None for _ in range(len(program_queue_ids))]
            import threading
            for workload_id, workload_i in enumerate(program_queue_ids):
                args_i = (workload_i, program_queue_name, multiobj_config, model_load)
                th[workload_id] = threading.Thread(target=get_HeterDSE_prediction_model,name="get_HeterDSE_prediction_model-"+str(workload_i), args=args_i)
                th[workload_id].start()
                #for workload_id, workload_i in enumerate(program_queue_ids):
                th[workload_id].join()
            print("all get_HeterDSE_prediction_model ends")
            if multiobj_config['pareto_crossover']:
                exp_info_map['origin_core_space_size'] = len(core_space_paretocrossover)
            multiobj_config['models_all_workloads'] = models_all_workloads
        elif multiobj_config['transfer_learning_model']:
            surrogate_model_tag_singlecore = multiobj_config['metric_prediction_model']
            surrogate_model_config_singlecore = get_surrogate_model(surrogate_model_tag_singlecore)
            _, _, surrogate_model_dict_singlecore = surrogate_model_config_singlecore
            if surrogate_model_dict_singlecore['model_is_iterative']:
                multiobj_config['n_initial_points'] = N_SAMPLES_INIT
            else:
                multiobj_config['n_initial_points'] = N_SAMPLES_ALL
            multiobj_config['n_generation_points'] = N_SAMPLES_ALL - multiobj_config['n_initial_points']
            single_core_model_workload_i = Problem_Model(problem_space=single_core_space,
                                                         surrogate_model_tag=surrogate_model_tag_singlecore,
                                                         surrogate_model_config=surrogate_model_config_singlecore,
                                                         multiobj_config=multiobj_config,
                                                         program_queue_info=program_queue_info,
                                                         schedule_mode=None,
                                                         obj_info=(2, (metric_1, metric_2), None),
                                                         )

        elif multiobj_config['aggr_workloads_models']:
            program_singlecore = program_queue_info
            surrogate_model_tag_singlecore = multiobj_config['aggr_workloads_models']
            surrogate_model_config_singlecore = get_surrogate_model(surrogate_model_tag_singlecore)
            _, _, surrogate_model_dict_singlecore = surrogate_model_config_singlecore
            multiobj_config['n_initial_points'] = N_SAMPLES_INIT if surrogate_model_dict_singlecore['model_is_iterative'] else N_SAMPLES_ALL
            multiobj_config['n_generation_points'] = N_SAMPLES_ALL - multiobj_config['n_initial_points']
            single_core_model_workload = Problem_Model(problem_space=single_core_space,
                                                         surrogate_model_tag=surrogate_model_tag_singlecore,
                                                         surrogate_model_config=surrogate_model_config_singlecore,
                                                         multiobj_config=multiobj_config,
                                                         program_queue_info=program_singlecore,
                                                         schedule_mode=None,
                                                         obj_info=(2, (metric_1, metric_2), None),
                                                         )
            single_core_model_workload_result = single_core_model_workload.run()
            time_used_map['register'] += single_core_model_workload_result.used_map['register']
            multiobj_config['models_all_workloads'] = single_core_model_workload_result.models
            print(f"models = {multiobj_config['models_all_workloads']}")
            core_space_paretocrossover = single_core_model_workload_result.x
            core_space_paretocrossover_metrics = single_core_model_workload_result.used_map['current_pareto_y']
            #pareto_x = [x[:-1] for x in single_core_model_workload_i.opt_gp.current_pareto_x]

        time_used_map['prepare'] = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime
        if model_load:
            startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")

        # for HMP-exploration
        if smoke_test:
            multiobj_config['n_initial_points'] = 2
            multiobj_config['n_generation_points'] = 0
        #elif 'BF-202106011024' == hostname:
        elif 'DESKTOP-A4P9S3E' == hostname:
            multiobj_config['n_initial_points'] = 2
            multiobj_config['n_generation_points'] = 3
        else:
            multiobj_config['n_initial_points'] = N_SAMPLES_INIT_HMP if surrogate_model_dict_hmp['model_is_iterative'] else N_SAMPLES_ALL_HMP
            multiobj_config['n_generation_points'] = N_SAMPLES_ALL_HMP - multiobj_config['n_initial_points']

        result_name = multiobj_config['tag'] + '_M' + surrogate_model_dict_hmp['tag']
        result_name += '_N' + str(multiobj_config['n_initial_points'] + multiobj_config['n_generation_points'])

        if core_space_paretocrossover is not None:
            core_space_paretocrossover, core_space_paretocrossover_index = np.unique(core_space_paretocrossover, return_index=True, axis=0)
            core_space_paretocrossover_metrics = core_space_paretocrossover_metrics[core_space_paretocrossover_index]
            print(f"core_space_paretocrossover size after unique = {len(core_space_paretocrossover)}")
            exp_info_map['core_space_size_unique'] = len(core_space_paretocrossover)
            #core_space_paretocrossover, core_space_paretocrossover_metrics = pareto_domain_filter(core_space_paretocrossover, multiobj_config['models_all_workloads'])
            #print(f"core_space_paretocrossover size after pareto_domain_filter = {len(core_space_paretocrossover)}")
            #exp_info_map['core_space_size_pareto_domain_filter'] = len(core_space_paretocrossover)
            #exit(0)
            if selection:
                #target_core_num = multiobj_config['n_clusters']
                tag += '_sel-' + selection + '-' + str(target_core_num)
                if target_core_num < len(core_space_paretocrossover):
                    #result_name += '_' + tag
                    #return ' ', 0, 0, 0, result_name
                    #return hmp_config, hmp_time, hmp_energy, hmp_schedule_time_used, result_name
                    #else:
                    '''
                    #file_name = 'log_hmp' + host_tag + '/' + program_queue_name
                    core_space_database_filename = 'core_space_database' + '/' + program_queue_name
                    core_space_database_filename += '/' + multiobj_config['tag'] + '_' + tag
                    core_space_database_filename += '_core-space.txt'
                    if os.path.exists(core_space_database_filename):
                        core_space_paretocrossover, time_used_map['selection'] = get_predict_core_space(core_space_database_filename)
                        selection_load = True
                    '''
                    #if len(core_space_paretocrossover) < 1:
                    selection_load = False
                    startTime2 = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
                    core_space_paretocrossover = design_point_selection(
                        core_space=core_space_paretocrossover,
                        core_space_metrics=core_space_paretocrossover_metrics,
                        program_queue=program_queue_ids,
                        target_core_num=target_core_num, 
                        selection=selection,
                        models=multiobj_config['models_all_workloads'],)
                    time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime2
                    time_used_map['selection'] = time_used.total_seconds()
            print(f"core_space_paretocrossover size = {len(core_space_paretocrossover)}")
            core_space = core_space_paretocrossover

        exp_info_map['selected_core_space_size'] = len(core_space)
        hmp_compose_space = Space(get_hmp_composement_space(core_space, multiobj_config))

    if 'HeterDSE' in SOTA:
        #core_types = ['Core-' + str(i) for i in range(len(core_space))]
        obj_args = None, core_space, use_eer

        result_name += '_' + tag
        print(f"result_name={result_name}...")
        explorer = Problem_Model(problem_space=hmp_compose_space,
                                      surrogate_model_tag=surrogate_model_tag_hmp,
                                      surrogate_model_config=surrogate_model_config_hmp,
                                      multiobj_config=multiobj_config,
                                      program_queue_info=program_queue_info,
                                      schedule_mode=schedule_mode,
                                      obj_info=(1, schedule, obj_args),
                                      model_load=False,
                                      tag=tag,
                                      )
        if hmp_model_load:
            print(f"hmp_model_load")
            hmp_explorer_result = explorer.run_from_model()
            x = hmp_explorer_result.used_map['opt_gp'].X_unlabeled
            y_predict = []
            for model in hmp_explorer_result.models:
                y_predict.append(model.predict(x))
            hmp_configs = hmp_explorer_result.used_map['opt_gp'].X_unlabeled
            hmp_times = []
            hmp_energys = []
            for hmp_config in hmp_configs:
                core_ids = []
                core_each_valid = []
                core_space_valid = []
                for each_id, each in enumerate(hmp_config):
                    if 0 < each:
                        core_ids.append(each_id)
                        core_each_valid.append(each)
                        core_space_valid.append(core_space[each_id])
                core_types_valid = ['Core-' + str(i) for i in range(len(core_ids))]
                core_counts = {f"{core_types_valid[i]}": core_each_valid[i] for i in range(len(core_ids))}

                f_val, f_val2 = schedule(program_bitmap, program_queue_ids,
                                         core_types=core_types_valid, core_counts=core_counts,
                                         core_vars=core_space_valid, use_eer=use_eer,
                                         models=None,
                                         multiobj_config=multiobj_config,
                                         )
                hmp_times.append(f_val)
                hmp_energys.append(f_val2)
            y_mape = sklearn.metrics.mean_absolute_percentage_error(hmp_times, y_predict[0])
            y2_mape = sklearn.metrics.mean_absolute_percentage_error(hmp_energys, y_predict[1])
            exp_info_map['y_mape']  = y_mape
            exp_info_map['y2_mape']  = y2_mape            
            #exp_info_map['aimFunc'] = explorer.statistics['aimFunc']
            #exp_info_map['schedule_result_database'] = len(schedule_result_database)
            #hmp_configs = hmp_explorer_result.used_map['opt_gp'].Xi
            hmp_configs = None
            print(f"")
            #exit(0)
        else:
            hmp_explorer_result = explorer.run()
            exp_info_map['aimFunc'] = explorer.statistics['aimFunc']
            exp_info_map['schedule_result_database'] = len(schedule_result_database)
            hmp_configs = explorer.opt_gp.Xi

        exp_info_map['last_mean'] = explorer.statistics['last_mean']
        exp_info_map['last_ucb'] = explorer.statistics['last_ucb']
        exp_info_map['X_all_go'] = explorer.opt_gp.result_array_sample_map['X_all_go']

        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime

        if not hmp_model_load:
            # objectives from models is not accurate
            # so, get real objectives by schedule
            hmp_times = []
            hmp_energys = []
            for hmp_config in hmp_configs:
                hmp_key = ''.join([str(s) for s in hmp_config])
                if hmp_key in schedule_result_database:
                    f_val, f_val2 = schedule_result_database[hmp_key]
                else:
                    print(f"warning hmp_config={hmp_config} should be evaluted")
                    core_ids = []
                    core_each_valid = []
                    core_space_valid = []
                    for each_id, each in enumerate(hmp_config):
                        if 0 < each:
                            core_ids.append(each_id)
                            core_each_valid.append(each)
                            core_space_valid.append(core_space[each_id])
                    core_types_valid = ['Core-' + str(i) for i in range(len(core_ids))]
                    core_counts = {f"{core_types_valid[i]}": core_each_valid[i] for i in range(len(core_ids))}

                    f_val, f_val2 = schedule(program_bitmap, program_queue_ids,
                                             core_types=core_types_valid, core_counts=core_counts,
                                             core_vars=core_space_valid, use_eer=use_eer,
                                             models=None,
                                             multiobj_config=multiobj_config,
                                             )
                    schedule_result_database[hmp_key] = (f_val, f_val2)
                hmp_times.append(f_val)
                hmp_energys.append(f_val2)
            #hmp_times = explorer.opt_gp.yi
            #hmp_energys = explorer.opt_gp.yi2
        #else:
        #    hmp_configs = explorer.opt_gp.current_pareto_x
        #    hmp_times = explorer.opt_gp.current_pareto_y[:, 0]
        #    hmp_energys = explorer.opt_gp.current_pareto_y[:, 1]

    elif 'GA' == SOTA:
        from GA_explore import GA_explore
        multiobj_config = {}
        #multiobj_config['tag'] = SOTA + '_' + tag
        multiobj_config['exp_id'] = exp_info
        multiobj_config['sche_evaluation_index'] = sche_evaluation_index

        surrogate_model_tag_hmp = None#"smoke_test2"
        surrogate_model_config_hmp = None#get_surrogate_model(surrogate_model_tag_hmp)

        #multiobj_config['n_initial_points'] = N_SAMPLES_INIT
        #multiobj_config['n_generation_points'] = N_SAMPLES_ALL - multiobj_config['n_initial_points']
        multiobj_config['pop'] = N_SAMPLES_INIT_HMP #20 #2000
        multiobj_config['gen'] = 30

        result_name = SOTA + '5_' + tag
        result_name += '_pop-' + str(multiobj_config['pop'])
        result_name += '_gen-' + str(multiobj_config['gen'])
        multiobj_config['result_name'] = result_name
        multiobj_config['gen'] = 50        

        hmp_compose_space = Space(get_hmp_composement_space(core_space))
        core_types = ['Core-' + str(i) for i in range(len(core_space))]
        obj_args = core_types, core_space, use_eer

        print(f"result_name={result_name}...")
        explorer = GA_explore(problem_space=hmp_compose_space,
                              surrogate_model_tag=surrogate_model_tag_hmp,
                              surrogate_model_config=surrogate_model_config_hmp,
                              multiobj_config=multiobj_config,
                              program_queue_info=program_queue_info,
                              program_queue_ids=program_queue_ids,
                              schedule_mode=schedule_mode,
                              obj_info=(1, schedule, obj_args),
                              )
        hmp_configs = explorer.run()
        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime

        exp_info_map['aimFunc'] = explorer.statistics['aimFunc']
        exp_info_map['schedule_result_database'] = len(schedule_result_database)

        exp_info_map['hmp_size_list'] = explorer.statistics['hmp_size_list']
        exp_info_map['hmp_metric_best_list'] = explorer.statistics['hmp_metric_best_list']

        exp_info_map['history_f'] = explorer.history['f']
        exp_info_map['history_aimFunc'] = explorer.history['aimFunc']
        exp_info_map['gen'] = explorer.history['gen']
        exp_info_map['history_schedule_result_database'] = explorer.history['schedule_result_database']

        hmp_times = []
        hmp_energys = []
        #hmp_metics = []
        for hmp_config in hmp_configs:
            hmp_config_key = hmp_config.tostring()
            if hmp_config_key in schedule_result_database:
                f_val_array = schedule_result_database[hmp_config_key]
                if multi_obj_mode:
                    f_val = mean(f_val_array[:][0])
                    f_val2 = mean(f_val_array[:][1])
                else:
                    f_val = f_val_array[0][0]
                    f_val2 = f_val_array[0][1]
            else:
                print(f"error hmp_config should be evaluted")
                core_ids = []
                core_each_valid = []
                core_space_valid = []
                for each_id, each in enumerate(hmp_config):
                    if 0 < each:
                        core_ids.append(each_id)
                        core_each_valid.append(each)
                        core_space_valid.append(core_space[each_id])
                core_types_valid = ['Core-' + str(i) for i in range(len(core_ids))]
                core_counts = {f"{core_types_valid[i]}": core_each_valid[i] for i in range(len(core_ids))}

                f_val, f_val2 = schedule(program_bitmap, program_queue_ids,
                                         core_types=core_types_valid, core_counts=core_counts,
                                         core_vars=core_space_valid, use_eer=use_eer,
                                         models=None,
                                         multiobj_config=multiobj_config,
                                         )
            hmp_times.append(f_val)
            hmp_energys.append(f_val2)
            #hmp_metics.append(f_val_array)

    elif 'SA' in SOTA:
        from SA_model import SA_model
        multiobj_config = {}
        # multiobj_config['tag'] = SOTA + '_' + tag
        multiobj_config['exp_id'] = exp_info
        multiobj_config['sche_evaluation_index'] = sche_evaluation_index

        #surrogate_model_tag_hmp = "smoke_test2"
        #surrogate_model_config_hmp = get_surrogate_model(surrogate_model_tag_hmp)

        # multiobj_config['n_initial_points'] = N_SAMPLES_INIT
        # multiobj_config['n_generation_points'] = N_SAMPLES_ALL - multiobj_config['n_initial_points']
        multiobj_config['pop'] = N_SAMPLES_INIT_HMP #50
        #multiobj_config['gen'] = 250

        result_name = SOTA + '6_' + tag
        result_name += '_pop-' + str(multiobj_config['pop'])
        #result_name += '_gen-' + str(multiobj_config['gen'])
        multiobj_config['result_name'] = result_name

        #hmp_compose_space = Space(get_hmp_composement_space(core_space))
        #core_types = ['Core-' + str(i) for i in range(len(core_space))]
        obj_args = None, None, use_eer

        print(f"result_name={result_name}...")
        explorer = SA_model(#problem_space=hmp_compose_space,
                            #surrogate_model_tag=surrogate_model_tag_hmp,
                            #surrogate_model_config=surrogate_model_config_hmp,
                            core_space=core_space,
                            multiobj_config=multiobj_config,
                            program_queue_info=program_queue_info,
                            schedule_mode=schedule_mode,
                            obj_info=(1, schedule, obj_args),
                            )
        hmp_configs = explorer.run()
        #print(f"hmp_configs={hmp_configs}")
        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime

        exp_info_map['aimFunc'] = explorer.statistics['aimFunc']
        exp_info_map['schedule_result_database'] = len(schedule_result_database)

        exp_info_map['hmp_size_list'] = explorer.statistics['hmp_size_list']
        exp_info_map['hmp_metric_best_list'] = explorer.statistics['hmp_metric_best_list']

        exp_info_map['history_f'] = explorer.history['f']
        exp_info_map['history_aimFunc'] = explorer.history['aimFunc']
        exp_info_map['history_schedule_result_database'] = explorer.history['schedule_result_database']

        hmp_times = []
        hmp_energys = []
        for hmp_config in hmp_configs:
            hmp_config_key = hmp_config.tostring()
            if hmp_config_key in schedule_result_database:
                f_val, f_val2 = schedule_result_database[hmp_config_key]
            else:
                print(f"warning hmp_config={hmp_config} should be evaluted")
                core_ids = []
                core_each_valid = []
                core_space_valid = []
                for each_id, each in enumerate(hmp_config):
                    if 0 < each:
                        core_ids.append(each_id)
                        core_each_valid.append(each)
                        core_space_valid.append(core_space[each_id])
                core_types_valid = ['Core-' + str(i) for i in range(len(core_ids))]
                core_counts = {f"{core_types_valid[i]}": core_each_valid[i] for i in range(len(core_ids))}

                f_val, f_val2 = schedule(program_bitmap, program_queue_ids,
                                         core_types=core_types_valid, core_counts=core_counts,
                                         core_vars=core_space_valid, use_eer=use_eer,
                                         models=None,
                                         multiobj_config=multiobj_config,
                                         )
            hmp_times.append(f_val)
            hmp_energys.append(f_val2)

    elif 'BruteForce' == SOTA:
        result_name = SOTA + '_' + tag
        
        #if selection:
        #    core_space = design_point_selection(core_space, core_space_metrics, selection=selection)

        pareto_filter = 1
        if pareto_filter:
            result_name += '_pf'

        only_count_mode = False

        print(f"result_name={result_name}...")
        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime
        
        file_name = 'log_hmp' + host_tag + '/' + program_queue_name
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        file_name += '/' + result_name + '.txt'
        file = open(file_name, 'w')

        file.write(f"{program_bitmap} \n")
        file.write(f"schedule_mode = {schedule_mode} \n")
        for each_key in exp_info_map.keys():
            file.write(f"{each_key} #= {exp_info_map[each_key]} \n")
        file.write(f"core_space #= {exp_info_map['selected_core_space_size']} \n")
        file.write(f"{core_space} \n")

        #hmp_configs = get_all_hmp(core_space=core_space)
        hmp_configs_num = 0
        hmp_configs_num_pareto_filter = 0
        hmp_configs_num_list = []
        #core_area_space = [area_all[var_to_version(core_i)[:-1]] for core_i in core_space]
        #core_types = ['Core-' + str(i) for i in range(len(core_space))]
        hmp_space_args = [i for i in range(len(core_space))]

        efficiency_min = 999999

        file.write(f"hmp_configs = \n")
        for core_type_num in range(1, 1 + MAX_CORE_TYPES):
            print(f"core_type_num = {core_type_num} / {MAX_CORE_TYPES}")
            #design_space_permutation = itertools.combinations(hmp_space_args, r=core_type_num)
            for each_core_type_space in combinations(hmp_space_args, r=core_type_num):
                hmps = []
                #print(f"each_core_type_space={each_core_type_space}")
                if 1 == len(core_space):
                    each_core_type_space = np.asarray([each_core_type_space])[0]
                # print(f"each_core_type_space={each_core_type_space}")
                var_list_range2 = [[] for i in range(core_type_num)]
                area_each_core_type_space = [area_all[var_to_version(core_space[var_iter])[:-1]] for var_iter in each_core_type_space]
                for var_index, var_iter in enumerate(each_core_type_space):
                    num_max = 1 + max(0, int((AREA_CONSTRAIN - sum(area_each_core_type_space)) / area_each_core_type_space[var_index]))
                    var_list_range2[var_index] = [value for value in range(num_max, 0, -1)]
                hmp_space_args2 = (var_list_range2[i] for i in range(core_type_num))
                core_type_space_product2 = itertools.product(*hmp_space_args2)
                for each_hmp in core_type_space_product2:
                    if 1 == core_type_num:
                        each_hmp = np.asarray(each_hmp)
                    # print(f"each_hmp={each_hmp}")
                    #hmp = [0 for _ in range(len(core_space))]
                    #for each_core_type_index, each_core_num in zip(each_core_type_space, each_hmp):
                    #    hmp[each_core_type_index] = each_core_num
                    each_core_space = np.asarray(copy.deepcopy(core_space))[each_core_type_space]
                    if hmp_is_under_constrain(each_core_space, each_hmp, area_each_core_type_space):
                        if pareto_filter:
                            hmp_configs_num_pareto_filter_flag = False
                            for each in hmps:
                                if (np.asarray(each_hmp) <= np.asarray(each)).all():
                                    hmp_configs_num_pareto_filter_flag = True
                                    #print(f"each_hmp={each_hmp} filtered")
                                    break
                            if hmp_configs_num_pareto_filter_flag:
                                hmp_configs_num_pareto_filter += 1
                                continue
                        hmp_configs_num += 1
                        hmps.append(each_hmp)
                        if not only_count_mode:
                            core_types = ['Core-' + str(each_core_type_space[i]) for i in range(len(each_core_type_space))]
                            core_counts = {f"{core_types[i]}": each_hmp[i] for i in range(len(each_core_type_space))}
                            print(f"hmp_id={hmp_configs_num} {core_counts}")                            
                            hmp_time, hmp_energy = schedule(program_bitmap, program_queue_ids,
                                                            core_types=core_types, core_counts=core_counts,
                                                            core_vars=each_core_space, 
                                                            use_eer=use_eer,
                                                            multiobj_config={'exp_id':exp_info, 'sche_evaluation_index':sche_evaluation_index},
                                                            models=None)
                            efficiency = get_hmp_metrics(hmp_time, hmp_energy)[EVALUATION_INDEX]
                            if efficiency < efficiency_min:
                                efficiency_min = efficiency
                                best_hmp = core_counts
                                best_hmp_time = hmp_time
                                best_hmp_energy = hmp_energy
                                best_each_core_space = each_core_space
                                best_core_types = core_types
                            file.write(f"{core_counts} {hmp_time} {hmp_energy} \n")
            if not only_count_mode:
                file.write(f"core_type_num <= {core_type_num}: best_each_core_space = {best_each_core_space} \n")
                file.write(f"core_type_num <= {core_type_num}: best_hmp = {best_hmp} \n")
                file.write(f"core_type_num <= {core_type_num}: best_core_types = {best_core_types} \n")                            
            hmp_configs_num_list.append(hmp_configs_num-sum(hmp_configs_num_list))

        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime

        file.write(f"hmp_configs # = {hmp_configs_num} \n")
        file.write(f"hmp_configs_num_list = {hmp_configs_num_list} \n")
        if pareto_filter:
            file.write(f"hmp_configs_num_pareto_filter = {hmp_configs_num_pareto_filter} \n")
            file.write(f"hmp_configs_num_before_pareto_filter = {hmp_configs_num_pareto_filter} + {hmp_configs_num} => {hmp_configs_num+hmp_configs_num_pareto_filter} \n")
        file.write(f"time_used = {time_used} \n")
        if not only_count_mode:
            #file.write(f"best_each_core_space = {best_each_core_space} \n")
            file.write(f"best_hmp = {best_hmp} \n")
            file.write(f"best_core_types = {best_core_types} \n")
        else:
            best_hmp = []
            best_hmp_time = -1
            best_hmp_energy = -1
        file.close()

        return best_hmp, best_hmp_time, best_hmp_energy, time_used.total_seconds(), result_name

    if 'BruteForce' != SOTA:

        time_used_map['flow_all'] = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime
        time_used_real = time_used.total_seconds()
        if model_load:
            time_used_real += time_used_map['single_core_model']
        else:
            time_used_real -= time_used_map['register']

        file_name = 'log_hmp' + host_tag + '/' + program_queue_name
        core_space_database_filename = 'core_space_database' + '/' + program_queue_name
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        if not os.path.exists(core_space_database_filename):
            os.mkdir(core_space_database_filename)
        file_name += '/' + result_name
        if hmp_model_load:
            file_name += '_loaded'
        core_space_database_filename += '/' + result_name
        file = open(file_name + '.txt', 'w')
        file.write(f"{program_bitmap} \n")
        file.write(f"schedule_mode = {schedule_mode} \n")
        for each in exp_info_map.keys():
            file.write(f"{each} #= {exp_info_map[each]} \n")
        #file.write(f"origin_core_space_size #= {exp_info_map['origin_core_space_size']} \n")
        file.write(f"core_space #= {exp_info_map['selected_core_space_size']} \n")
        file.write(f"{core_space} \n")
        file.write(f"time_used = {time_used_real} \n")

        for each_key in time_used_map.keys():
            file.write(f"time_used_map[{each_key}] = {time_used_map[each_key]} \n")
        if hmp_configs is not None:
            file.write(f"hmp_configs # = {len(hmp_configs)} \n")
            file.write(f"hmp_configs = \n")
            if 0 == multi_obj_mode:
                for x, t, e in zip(hmp_configs, hmp_times, hmp_energys):
                    if type(x) == np.array:
                        x = x.tolist()
                    file.write(f"{x} {t} {e} \n")
            else:
                for x, t, e, m in zip(hmp_configs, hmp_times, hmp_energys, hmp_metrics):
                    if type(x) == np.array:
                        x = x.tolist()
                    file.write(f"{x} {t} {e} {m} \n")

            if 'HeterDSE' in SOTA:
                file.write(f"hmp_compose_space # = {len(explorer.opt_gp.X_all)} \n")
                if explorer.opt_gp.result_map['pareto_filter']:
                    file.write(f"hmp_configs_num_pareto_filter = {explorer.opt_gp.result_map['hmp_configs_num_pareto_filter']} \n")            
            elif 'SA' in SOTA or 'GA' in SOTA:
                for str_key in explorer.result_map.keys():
                    file.write(f"{str_key} = {explorer.result_map[str_key]} \n")
                if explorer.result_map['pareto_filter']:
                    file.write(f"hmp_configs_num_pareto_filter = {explorer.result_map['hmp_configs_num_pareto_filter']} \n")            


        file.close()

        if not selection_load:
            core_space_database_filename += '_core-space'
            core_space_file = open(core_space_database_filename + '.txt', 'w')
            core_space_file.write(f"time_used_map['selection'] = {time_used_map['selection']} \n")
            for each in core_space:
                if not list(each):
                    each = each.tolist()
                core_space_file.write(f"{each}\n")
            core_space_file.close()

        if hmp_configs is not None:
            hmp_efficiencys = [get_hmp_metrics(hmp_time, hmp_energy)[EVALUATION_INDEX] for hmp_time, hmp_energy in zip(hmp_times, hmp_energys)]
            # print(f"hmp_efficiencys={hmp_efficiencys}")
            best_hmp_id = np.argmin(hmp_efficiencys, axis=0)
        else:
            print(f"hmp_configs is None")
            return None, -1, -1, -1, result_name

        best_hmp = hmp_configs[best_hmp_id]
        best_core_ids = []
        var_each_valid = []
        for each_id, each in enumerate(best_hmp):
            if 0 < each:
                best_core_ids.append(each_id)
                var_each_valid.append(each)
        best_core_counts = {f'Core-{best_core_ids[i]}': var_each_valid[i] for i in range(len(var_each_valid))}

        return best_core_counts, hmp_times[best_hmp_id], hmp_energys[best_hmp_id], time_used_real, result_name

# if __name__ == '__main__':
# core_space = [var_YANQIHU, var_BOOMv2, var_ROCKET]
# program_queue_name, program_queue, program_bitmap = get_workloads_all(random_seed=0)
# file = open('log_hmp.txt', 'a')
# hmp_config, hmp_time, hmp_energy, hmp_schedule_time_used = hmp_evaluate(program_bitmap, core_space, tag='test', schedule_mode='naive', models=None)
# file.write(f"{core_space} {hmp_config} {hmp_time} {hmp_energy} {hmp_schedule_time_used} \n")
# file.close()
