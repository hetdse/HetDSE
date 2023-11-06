import copy
import numpy as np

from get_real_pareto_frontier import is_pareto_efficient_dumb
from simulation_metrics import delay_metric, energy_metric, metric_1_2, SIMULATOR_CYCLES_PER_SECOND_map
from config import EVALUATION_INDEX, get_hmp_metrics
from skopt_plot import rgb_to_hex


def compose_selection(core_space, benchmarks, selection, target_core_num, random_state=0, models=None):

    if len(core_space) <= target_core_num:
        return core_space

    #alg = 'clique'
    alg = selection.split('-')[1]

    points = []
    if 'rank2' == alg:
       points_value = []
       for x in core_space:
            point = np.zeros(len(benchmarks)*4)
            #for workload_id, workload_i in enumerate(program_queue_ids):
            for b_id, b in enumerate(benchmarks):
                est, est2 = models[b]
                for freq_str in SIMULATOR_CYCLES_PER_SECOND_map.keys():
                    freq = int(freq_str)
                    x_copy = copy.deepcopy(x).tolist()
                    x_copy.append(freq)
                    delay = est.predict([x_copy])[0]
                    energy = est2.predict([x_copy])[0]
                    point[b_id*4+freq] = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
            points_value.append(point)
       points_value = np.asarray(points_value)

       #core_space_pareto, points_value = is_pareto_efficient_dumb(core_space, points_value)
       #print(f"selection: size from {len(core_space)} to {len(core_space_pareto)}")
       #core_space = core_space_pareto
       #exit(1)

       ranks = np.argsort(points_value, axis=0)
       #print(f"points_value={points_value}")
       #print(f"ranks={ranks}")
       points = np.log2(ranks+1)
    elif 'prsort' == alg:
        for x in core_space:
            point = np.zeros(2)
            #for workload_id, workload_i in enumerate(program_queue_ids):
            for b_id, b in enumerate(benchmarks):
                est, est2 = models[b]
                for freq in SIMULATOR_CYCLES_PER_SECOND_map.keys():
                    x_copy = copy.deepcopy(x).tolist()
                    x_copy.append(int(freq))
                    delay = est.predict([x_copy])[0]
                    energy = est2.predict([x_copy])[0]
                    point[0] += delay
                    point[1] += energy
            points.append(point)
    elif 'sort2' == alg or 'sort4' == alg or 'sort5' == alg or 'sort6' == alg:
        for x in core_space:
            point = np.zeros(len(benchmarks)*4)
            #for workload_id, workload_i in enumerate(program_queue_ids):
            for b_id, b in enumerate(benchmarks):
                if models[b_id] is not None:
                    est, est2 = models[b]
                    for freq_str in SIMULATOR_CYCLES_PER_SECOND_map.keys():
                        freq = int(freq_str)
                        x_copy = copy.deepcopy(x).tolist()
                        x_copy.append(freq)
                        delay = est.predict([x_copy])[0]
                        energy = est2.predict([x_copy])[0]
                        point[b_id*4+freq] = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
            point_min = [min(point[b_id*4:(b_id+1)*4]) for b_id in range(len(benchmarks))]
            points.append(point_min)
    else:
        for x in core_space:
            point = np.zeros(len(benchmarks)*4)
            #for workload_id, workload_i in enumerate(program_queue_ids):
            for b_id, b in enumerate(benchmarks):
                est, est2 = models[b]
                for freq_str in SIMULATOR_CYCLES_PER_SECOND_map.keys():
                    freq = int(freq_str)
                    x_copy = copy.deepcopy(x).tolist()
                    x_copy.append(freq)
                    delay = est.predict([x_copy])[0]
                    energy = est2.predict([x_copy])[0]
                    point[b_id*4+freq] = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
                #point[b_id*4:(b_id+1)*4] /= point[b_id*4]
            points.append(point)
    points = np.asarray(points)
    points_raw = copy.deepcopy(points)
    
    if 'sort4' == alg or 'sort5' == alg:
        core_space_pareto, points, index = is_pareto_efficient_dumb(core_space, points, return_index=1)
        print(f"selection: size from {len(core_space)} to {len(core_space_pareto)}")
        core_space = core_space_pareto
        points_raw = points_raw[index]

    # BIPS^3/W = (0.01 / delay) ^ 3 / (energy / delay) = 1e-6 / delay^2 / energy

    if 1:
        from sklearn.decomposition import PCA
        if 'sort3' == alg or 'sort4' == alg or 'sort5' == alg:
            n_components = 1
        else:
            n_components = 2
        pca = PCA(n_components=n_components)
        points = pca.fit_transform(points)
        print(f"explained_variance_ratio_={pca.explained_variance_ratio_}")
        print(f"pca.get_params={pca.get_params()}")

    samples = []
    x_selecteds = []

    if 'rank' == alg:
        #print(f"points_raw={points_raw}")
        metrics_agg = np.sum(points_raw, axis=1)
        x_selecteds = np.argsort(metrics_agg)[:target_core_num]
        samples = core_space[x_selecteds]
    elif 'sort5' == alg or 'sort6' == alg:
        from pyclustering.utils.metric import type_metric, distance_metric

        #feature_weight = [1 for _ in range(len(points[0]))]
        metric = distance_metric(type_metric.USER_DEFINED,
                                 func=lambda point1, point2: np.sum(np.square(point1 - point2)))

        from pyclustering.cluster.kmeans import kmeans
        X_index = np.arange(0, len(points))
        import random
        random.shuffle(X_index)
        start_centers = np.asarray(points)[X_index[:target_core_num]]
        alg_instance = kmeans(points, start_centers, metric=metric)
        alg_instance.process()
        clusters = alg_instance.get_clusters()

        for each_cluster in clusters:
            metrics = []
            for each_point in each_cluster:
                metrics.append(sum(points_raw[each_point]))
            x_selected = np.argmin(metrics)
            samples.append(core_space[each_cluster[x_selected]])
            x_selecteds.append(x_selected)
    elif 'sort4' == alg:
        metrics_agg = np.sum(points, axis=1)
        x_selecteds = np.argsort(metrics_agg)[:target_core_num]
        samples = core_space[x_selecteds]
        print(f"x_selecteds={x_selecteds}")
    elif 'sort' == alg or 'sort2' == alg or 'sort3' == alg:
        metrics_agg = np.sum(points_raw, axis=1)
        x_selecteds = np.argsort(metrics_agg)[:target_core_num]
        samples = core_space[x_selecteds]
        print(f"x_selecteds={x_selecteds}")
        #exit(0)
    elif 'prsort' == alg:
        from Pareto import Pareto
        pareto = Pareto(pop_obj=points[:, 0], pop_obj2=points[:, 1])
        pareto.fast_non_dominate_sort()
        for i, f in enumerate(pareto.f):
            if len(x_selecteds) < target_core_num:
                for each_point in f:
                    x_selecteds.append(each_point)
            else:
                break
        samples = core_space[x_selecteds]
    else:
        if 'kmeans' == alg:
            from pyclustering.utils.metric import type_metric, distance_metric

            feature_weight = [1 for _ in range(len(points[0]))]
            metric = distance_metric(type_metric.USER_DEFINED,
                                     func=lambda point1, point2: np.sum(np.square(point1 - point2) * feature_weight))

            from pyclustering.cluster.kmeans import kmeans
            X_index = np.arange(0, len(points))
            import random
            random.shuffle(X_index)
            start_centers = np.asarray(points)[X_index[:target_core_num]]
            alg_instance = kmeans(points, start_centers, metric=metric)
        elif 'clique' == alg:
            from pyclustering.cluster.clique import clique, clique_visualizer
            alg_instance = clique(data=points, amount_intervals=20, density_threshold=0)
        elif 'birch' == alg:
            from pyclustering.cluster.birch import birch
            alg_instance = birch(data=points, number_clusters=target_core_num)

        # run cluster analysis and obtain results
        alg_instance.process()
        clusters = alg_instance.get_clusters()

        for each_cluster in clusters:
            metrics = []
            for each_point in each_cluster:
                metrics.append(sum(points_raw[each_point]))
            x_selected = np.argmin(metrics)
            samples.append(core_space[x_selected])
            x_selecteds.append(x_selected)

    if 0:
        visualizer_filename = 'fig/hmp/design_point_selection_' + selection + alg + '.png'
        if 'clique' == alg:
            noise = alg_instance.get_noise()
            cells = alg_instance.get_cells()
            clique_visualizer.show_grid(cells, points)
            clique_visualizer.show_clusters(points, clusters, noise)
            clique_visualizer.save(visualizer_filename)
        elif 'birch' == alg:
            from pyclustering.cluster import cluster_visualizer, cluster_visualizer_multidim
            visualizer = cluster_visualizer_multidim()
            visualizer.append_clusters(clusters, points)
            visualizer.show()
            visualizer.save(visualizer_filename)

    #exit(0)

    if 0:
        # colors = ['red', 'green', 'blue', 'black']
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 4))
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

        #for each in points:
        if 0:
            if False: #'sort' == alg:
                plt.scatter(points[:, 0], points[:, 1], s=6, c=metrics_agg, cmap='Blues_r')
                plt.colorbar()
            else:
                plt.scatter(points[:, 0], points[:, 1], s=6, c='gray')

        if 0:
            for each in x_selecteds:
                plt.scatter(points[each][0], points[each][1], s=3, c=rgb_to_hex(227, 102, 19), marker='x')

        if 0:
            for cluster_i, cluster in enumerate(alg_instance._kmeans__clusters):
                plt.scatter(points[cluster][:, 0], points[cluster][:, 1], label=str(cluster_i), s=2)
                #ax.set_xlabel(multiobj_mode_x_label, fontsize=fontsize, font=font, labelpad=10)
                #ax.set_ylabel(multiobj_mode_y_label, fontsize=fontsize, font=font, labelpad=10)
                # plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)
                # plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)
                # plt.set_yticklabels(['%.1f'%(i/4) for i in range(0,5)], fontsize=fontsize, font=font)
                # plt.xlabel(var_list[0])
                # plt.ylabel(var_list[1])
        from hmp_plot import read_log_hmp_file
        final_result, best_id = read_log_hmp_file(hostname_tag='-BF',
                                   program_queue_name='set_n23_id0',
                                   result_name='HeterDSE_pc_m-AdaGBRT-mm1-init1_v0_MRF_diff_model_ei2_ucb3v12-0.05_v4_N100_EER__AC0_EI2_EX0_sel-Hetsel-sort-30',
                                   evaluation_index=2,
                                   top_avg_mode=0,
                                   return_best=1,
                                )
        final_result_random, best_id_random = read_log_hmp_file(hostname_tag='-BF',
                                   program_queue_name='set_n23_id0',
                                   result_name='HeterDSE_pc_m-AdaGBRT-mm1-init1_v0_MRF_diff_model_ei2_ucb3v12-0.05_v4_N100_EER__AC0_EI2_EX0_sel-Random-30',
                                   evaluation_index=2,
                                   top_avg_mode=0,
                                   return_best=1,
                                )

        #for each_hmp_str in final_result['hmp_configs']:
        final_result_hmp_configs = [81,82,64,85,50,62,52,60,47,59,68,66,48,83,61,77,86,71,70,31,80,32,79,43,44,56,57,53,55,42]
        final_result_hmp_configs_random = [38,27,17,3,44,57,81,22,30,25,62,68,23,72,59,16,58,29,2,54,8,37,64,69,76,43,82,52,86,73]

        print(f"{selection}")
        plt.scatter(points[final_result_hmp_configs][0], points[final_result_hmp_configs][1], s=3, c='orange', marker='x', label=selection)
        for each_core_idx_iter, each_core_num in enumerate(eval(final_result['hmp_configs'][best_id])):
            if each_core_num:
                each_core_idx = final_result_hmp_configs[each_core_idx_iter]
                plt.scatter(points[each_core_idx][0], points[each_core_idx][1], s=4, c='red', marker='p')
                print(f"[{each_core_idx_iter}]: {points[each_core_idx][0]},{points[each_core_idx][1]}")

        print(f"random")
        plt.scatter(points[final_result_hmp_configs_random,0], points[final_result_hmp_configs_random,1], s=3, c='lightblue', marker='x', label='random')
        for each_core_idx_iter, each_core_num in enumerate(eval(final_result_random['hmp_configs'][best_id])):
            if each_core_num:
                each_core_idx = final_result_hmp_configs_random[each_core_idx_iter]
                plt.scatter(points[each_core_idx][0], points[each_core_idx][1], s=4, c='blue', marker='^')
                print(f"[{each_core_idx_iter}]: {points[each_core_idx][0]},{points[each_core_idx][1]}")

        label_fontsize = 10
        if 1:
            labelss = plt.legend(fontsize=90, loc='upper center',
                                 #bbox_to_anchor=(x_shift, y_shift),
                                 prop={'size': 15}, shadow=False,
                                 # ncol=labels_num,
                                 #ncol=int(labels_num / 2 + 0.5),
                                 )
            font2 = font
            font2['weight'] = 'bold'
            font2['size'] = label_fontsize
            [label.set_fontproperties(font2) for label in labelss.get_texts()]

        plt.tight_layout()
        plt.savefig('fig/hmp/design_point_selection_' + selection + '.pdf')
        plt.savefig('fig/hmp/design_point_selection_' + selection + '.png')
        exit(0)

    # print(f"samples={samples}")
    return samples