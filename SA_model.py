import copy
import math
import random

import numpy as np
#from scipy import spatial
#import sys
#from sko.SA import SAFast

from config import EVALUATION_INDEX, get_hmp_metrics, schedule_result_database, MAX_CORE_TYPES, N_SAMPLES_ALL_HMP
from hmp_evaluate import get_all_hmp
from simulation_metrics import AREA_CONSTRAIN, var_to_version, area_all, metric_1_2, hmp_is_under_constrain


#Simulated Annealing，SA
class SA_model():

    def __init__(self, core_space, multiobj_config, program_queue_info, schedule_mode, obj_info):
        self.core_space = core_space
        self.program_queue_name, self.program_queue, self.program_bitmap, self.program_queue_ids = program_queue_info
        self.num_var = len(core_space)
        self.multiobj_config = multiobj_config
        self.schedule_mode = schedule_mode
        self.obj_func_num, self.obj_func, self.obj_args = obj_info
        _, _, self.use_eer = self.obj_args
        self.core_area_space = [area_all[var_to_version(var_iter)[:-1]] for var_iter in self.core_space]
        self.statistics = {}
        self.statistics['aimFunc'] = 0
        self.statistics['schedule_result_database'] = 0
        self.statistics['hmp_metric_best'] = 9999
        self.statistics['hmp_size_list'] = []
        self.statistics['hmp_metric_best_list'] = []
        self.statistics['neighbours_count'] = 0
        self.upper_bound = []
        for var_index, var_area in enumerate(self.core_area_space):
            max_num = int(AREA_CONSTRAIN / var_area)
            self.upper_bound.append(max_num)
        #lower=np.zeros(self.num_var)
        # 100->1 with 0.9, 44 iterations, 800 sche
        # 5->1 with 0.9, 16 iterations, 454 sche
        self.T_max = 300
        self.T_min = 1
        self.alpha = 0.8 # temperature down coefficient

        self.T = self.T_max

        import random
        #random_seed = 0
        self.hmp_all, self.result_map = get_all_hmp(core_space)
        self.hmp_all = np.asarray(self.hmp_all)
        self.pop = min(multiobj_config['pop'], len(self.hmp_all))
        case_ids = np.arange(0, len(self.hmp_all))
        random.seed(multiobj_config['exp_id'])
        random.shuffle(case_ids)
        self.x = self.hmp_all[case_ids[:self.pop]]
        self.best_pop = self.x
        self.y = [self.aimFunc(x) for x in self.x]
        self.neighbours = {}
        self.arch_weights = []
        self.set_arch_weight()
        #print(f"arch_weights={self.arch_weights}")
        self.history = {'f': [], 'aimFunc': [], 'schedule_result_database': []}

    def set_arch_weight(self):
        for x in self.core_space:
            weight = 0
            for program in self.program_queue:
                x_copy = copy.deepcopy(x).tolist()
                x_copy.append(x[0])
                f_val, f_val2 = metric_1_2(x_copy, program)
                weight += get_hmp_metrics(delay=f_val, energy=f_val2)[EVALUATION_INDEX]
            self.arch_weights.append(weight)

    def aimFunc(self, var_each):
        #print(f"SA: var_each={var_each}")
        area_sum = np.sum(self.core_area_space * np.abs(np.asarray(copy.deepcopy(var_each))))
        var_each_valid_slice = 0 < np.asarray(var_each)
        var_each_valid = var_each[var_each_valid_slice]
        core_counts_flag = len(var_each_valid) - 1e-1 - MAX_CORE_TYPES
        # print(f"aimFunc: area_sum={area_sum}")
        if (0 < sum(var_each)) and (area_sum < AREA_CONSTRAIN) and (core_counts_flag < 0):
            var_each_key = var_each.tostring()
            if var_each_key in schedule_result_database:
                f_val, f_val2 = schedule_result_database[var_each_key]
                metric = get_hmp_metrics(delay=f_val, energy=f_val2)[EVALUATION_INDEX]
            else:
                #print(f"aimFunc var_each={var_each}")
                core_space = np.asarray(copy.deepcopy(self.core_space))[var_each_valid_slice]
                core_types = ['Core-' + str(i) for i in range(len(core_space))]
                core_counts = {f"{core_types[i]}": int(var_each_valid[i]) for i in range(len(core_space))}
                f_val, f_val2 = self.obj_func(self.program_bitmap,
                                              self.program_queue_ids,
                                              core_types=core_types, core_counts=core_counts,
                                              core_vars=core_space, use_eer=self.use_eer,
                                              models=None,
                                              multiobj_config={'exp_id': self.multiobj_config['exp_id'],
                                                        'sche_evaluation_index': self.multiobj_config['sche_evaluation_index']},
                                              )
                schedule_result_database[var_each_key] = (f_val, f_val2)
                self.statistics['schedule_result_database'] += 1
                self.statistics['hmp_size_list'].append(self.statistics['schedule_result_database'])
                metric = get_hmp_metrics(delay=f_val, energy=f_val2)[EVALUATION_INDEX]
                #print(f"{self.statistics['schedule_result_database']} = {core_space} {core_counts} {self.program_bitmap} {f_val}, {f_val2} {metric}")
                self.statistics['hmp_metric_best'] = min(self.statistics['hmp_metric_best'], metric)
                self.statistics['hmp_metric_best_list'].append(self.statistics['hmp_metric_best'])
            self.statistics['aimFunc'] += 1
        else:
            metric = area_sum + 1000
        return metric

    def get_neighbours(self, x):
        x_key = x.tostring()
        if x_key in self.neighbours:
            neighbours, neighbours_weight = self.neighbours[x_key]
        else:
            neighbours = []
            neighbours_weight = []
            for each_x in self.hmp_all:
                x_diff = (each_x - x)
                if not (each_x == x).all():
                    minus_one = np.count_nonzero(x_diff == -1)
                    plus_one = np.count_nonzero(x_diff == 1)
                    #print(f"neighbour_flag: minus_one = {minus_one}, plus_one = {plus_one}")
                    neighbour_flag = (1 == minus_one) and (1 == plus_one) and ((len(x_diff)-2) == np.count_nonzero(x_diff == 0))
                    if neighbour_flag:
                        if hmp_is_under_constrain(self.core_space, each_x, self.core_area_space):
                            #print(f"neighbour_flag: x = {x}, each_x={each_x}")
                            neighbours.append(each_x)
                            neighbour_weight = sum(x_diff * self.arch_weights)
                            neighbours_weight.append(neighbour_weight)
            self.neighbours[x_key] = (neighbours, neighbours_weight)
            self.statistics['neighbours_count'] += len(neighbours)
        return neighbours, neighbours_weight

    def roulette_wheel_method(self, di):
        di = np.asarray(di)        
        if 1:
            di_pow = pow(math.e, di)
            prob = di_pow / sum(di_pow)
        else:
            di_sort_index = np.argsort(-di)
            di_head = di[di_sort_index]
            di_prob_1 = di_head / sum(di_head)
            di_pow = pow(di_prob_1, 0.5)
            #di_pow = pow(di_prob_1, 1.0/power_num)
            prob = di_pow / sum(di_pow)
            #di_sort_index[-1]
        #print(f"weight prob={prob*100}")
        di_acc = 0
        for di_choose_iter, di_choose_index in enumerate(prob):
            di_acc += prob[di_choose_iter]
            if random.random() < di_acc:
                #di_max_index = di_choose_index
                return di_choose_iter
        return len(prob)-1

    def generate_new(self, x):  # 扰动产生新解的过程
        #print(f"generate_new x={x}")
        neighbours, neighbours_weight = self.get_neighbours(x)
        if len(neighbours) < 1:
            return x
        x_new_id = self.roulette_wheel_method(neighbours_weight)
        x_new = neighbours[x_new_id]
        #x_new = random.choice(neighbours)
        #while True:
            #x_new = random.choice(self.hmp_all)
        #    break
            #x_new = x + self.T * (random() - random())
            #if (0 <= x_new).all() and (x_new <= self.upper_bound).all():
            #    break # 重复得到新解，直到产生的新解满足约束条件
        return x_new

    def Metrospolis(self, f, f_new):
        if f_new < f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random.random() < p:
                return 1
            else:
                return 0

    def best(self):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.pop):
            f = self.aimFunc(self.x[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        #count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        ft_min = 9999
        while self.T > self.T_min:

            # 内循环迭代
            for i in range(self.pop):
                f = self.y[i]  # f为迭代一次后的值
                x_new = self.generate_new(self.x[i])
                #print(f"T={self.T}: [{i}] / {len(self.y)} x={x_new} ")
                #print(f"all={self.hmp_all}")
                f_new = self.aimFunc(x_new)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    self.x[i] = x_new  # 如果接受新值，则保存新值的x
                    self.y[i] = f_new
                end_contition = N_SAMPLES_ALL_HMP <= self.statistics['schedule_result_database']
                if end_contition:
                    print(f"reach N_SAMPLES_ALL_HMP={N_SAMPLES_ALL_HMP}")
                    break
            # 迭代L次记录在该温度下最优解
            ft, _ = self.best()
            print(f"T={self.T} {ft}")
            self.history['f'].append(ft)
            if ft < ft_min:
                self.best_pop = self.x
            ft_min = min(ft_min, ft)                
            self.history['aimFunc'].append(self.statistics['aimFunc'])
            self.history['schedule_result_database'].append(self.statistics['schedule_result_database'])
            #self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            #self.iter_count += 1
            if end_contition:
                break

        print(f"neighbours_count = {self.statistics['neighbours_count']}")
        print(f"hmp_configs_num #= {len(self.hmp_all)}")
        print(f"schedule_result_database #= {self.statistics['schedule_result_database']}")
        print(f"iteration_count #= {len(self.history['aimFunc'])}")
        return self.best_pop
        #f_best, idx = self.best()
        #print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")
