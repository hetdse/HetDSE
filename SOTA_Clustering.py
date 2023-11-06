import copy
import numpy as np

from simulation_metrics import delay_metric, energy_metric, metric_1_2
from config import EVALUATION_INDEX, get_hmp_metrics


class SOTA_Clustering():

    def __init__(self, core_space, benchmarks, selection):
        self.core_space = core_space
        self.benchmarks = benchmarks
        self.selection = selection

    def get_Clustering(self, target_core_num, random_state=0, models=None):
        if len(self.core_space) <= target_core_num:
            return self.core_space
        points = []
        for x in self.core_space:
            if 'Clustering4' == self.selection:
                point = np.zeros(len(self.benchmarks)*4)
                #for workload_id, workload_i in enumerate(program_queue_ids):
                for b_id, b in enumerate(self.benchmarks):
                    est, est2 = models[b_id]
                    for freq in range(4):
                        x_copy = copy.deepcopy(x).tolist()
                        x_copy.append(freq)
                        delay = est.predict([x_copy])[0]
                        energy = est2.predict([x_copy])[0]
                        point[b_id*4+freq] = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
                    point[b_id*4:(b_id+1)*4] /= point[b_id*4]
            elif 'Clustering3' == self.selection:
                point = np.zeros(len(self.benchmarks)*4)
                #for workload_id, workload_i in enumerate(program_queue_ids):
                for b_id, b in enumerate(self.benchmarks):
                    est, est2 = models[b_id]
                    for freq in range(4):
                        x_copy = copy.deepcopy(x).tolist()
                        x_copy.append(freq)
                        delay = est.predict([x_copy])[0]
                        energy = est2.predict([x_copy])[0]
                        point[b_id*4+freq] = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
                    point[b_id*4:(b_id+1)*4] /= point[b_id*4]
            elif 'Clustering2' == self.selection:
                x_copy.append(x[0])
                point = []
                for b in self.benchmarks:
                    delay, energy = metric_1_2(x_copy, b)
                    efficiency = get_hmp_metrics(delay, energy)[EVALUATION_INDEX]
                    point.append(efficiency)
            else:
                x_copy = copy.deepcopy(x).tolist() 
                x_copy.append(x[0])
                point = [(delay_metric(x_copy, b) ** 2 / energy_metric(x_copy, b)) for b in self.benchmarks]                
            points.append(point)
        points = np.asarray(points)
        # BIPS^3/W = (0.01 / delay) ^ 3 / (energy / delay) = 1e-6 / delay^2 / energy
        #Y_copy = np.asarray(copy.deepcopy(Y))
        #points = Y_copy[:0] ** 2 / Y_copy[:1]

        if 'Clustering4' == self.selection:
            from sklearn.decomposition import PCA  # 加载PCA算法包
            pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
            points = pca.fit_transform(points)  # 对样本进行降维
            print(f"explained_variance_ratio_={pca.explained_variance_ratio_}")

        '''
        if 'kmeansy-scale' == selection:
            points = np.asarray(copy.deepcopy(Y))
        elif 'kmeansy' == selection:
            points = np.asarray(copy.deepcopy(Y))
        elif 'kmeansx' == selection:
            points = np.asarray(copy.deepcopy(X))
        else:
            print(f"get_by_kmeans no def selection={selection}")
            exit(0)
        '''
        from pyclustering.utils.metric import type_metric, distance_metric

        feature_weight = [1 for _ in range(len(points[0]))]
        user_function = lambda point1, point2: np.sum(np.square(point1 - point2) * feature_weight)
        metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

        samples = []
        x_selecteds = []

        from pyclustering.cluster.kmeans import kmeans
        X_index = np.arange(0, len(points))
        import random
        random.shuffle(X_index)
        start_centers = np.asarray(points)[X_index[:target_core_num]]
        kmeans_instance = kmeans(points, start_centers, metric=metric)

        # run cluster analysis and obtain results
        kmeans_instance.process()
        clusters = kmeans_instance.get_centers()

        for each_center in clusters:
            x_distance = []
            for x_id in range(len(points)):
                x_distance.append([x_id, np.sum(np.square(points[x_id] - each_center) * feature_weight)])
            x_distance_sorted = np.asarray(sorted(x_distance, key=lambda obj_values: obj_values[1]))
            # print(f"x_distance_sorted={x_distance_sorted}")
            equal_flag = (np.abs(x_distance_sorted[:, 1] - x_distance_sorted[0, 1])) < 1e-4
            equal_index = np.argmin(equal_flag)
            if 0 == equal_index:
                x_selected = int(x_distance_sorted[0][0])
            else:
                #print(f"equal_flag={equal_flag}, equal_index={equal_index}")
                x_selected = int(x_distance_sorted[int(random.randint(0, equal_index))][0])
            # print(f"x_distance_sorted={x_distance_sorted[:10]}")
            # x_selected = np.argmin(x_distance)
            samples.append(self.core_space[x_selected])
            x_selecteds.append(x_selected)
            # samples.append(X[x_selected].tolist())

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

            for each in points:
                plt.scatter(each[0], each[1], c='gray', s=2)

            for each in x_selecteds:
                plt.scatter(points[each][0], points[each][1], s=2, marker='*')

            if 0:
                for cluster_i, cluster in enumerate(kmeans_instance._kmeans__clusters):
                    plt.scatter(points[cluster][:, 0], points[cluster][:, 1], label=str(cluster_i), s=2)
                    #ax.set_xlabel(multiobj_mode_x_label, fontsize=fontsize, font=font, labelpad=10)
                    #ax.set_ylabel(multiobj_mode_y_label, fontsize=fontsize, font=font, labelpad=10)
                    # plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)
                    # plt.set_yticks([str(i) for i in range(0,5)], fontsize=fontsize, font=font)
                    # plt.set_yticklabels(['%.1f'%(i/4) for i in range(0,5)], fontsize=fontsize, font=font)
                    # plt.xlabel(var_list[0])
                    # plt.ylabel(var_list[1])

            label_fontsize = 10
            if 0:
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

            plt.savefig('fig/hmp/design_point_selection_kmeans_' + self.selection + '.png')
            exit(0)

        # print(f"samples={samples}")
        return samples