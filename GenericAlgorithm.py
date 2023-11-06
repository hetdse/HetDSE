# -*- coding: utf-8 -*-
import copy

import numpy as np
import geatpy as ea
import torch
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning

from config import EVALUATION_INDEX, get_hmp_metrics, schedule_result_database, MAX_CORE_TYPES, multi_obj_mode, \
    N_SAMPLES_ALL_HMP
from get_real_pareto_frontier import is_pareto_efficient_dumb
from simulation_metrics import var_to_version, area_all, AREA_CONSTRAIN


class GenericAlgorithm(ea.Problem): # 继承Problem父类
 
    def __init__(self, var_space, obj_info, multiobj_config, program_bitmap, program_queue_ids):
        self.multiobj_config = multiobj_config
        name = multiobj_config['result_name'] # 初始化name(函数名称，可以随意设置)
        self.var_space = var_space
        self.program_bitmap = program_bitmap
        self.program_queue_ids = program_queue_ids
        #self.obj_info = obj_info
        self.obj_func_num, self.obj_func, self.obj_args = obj_info
        _, _, self.use_eer = self.obj_args
        M = 1 + multi_obj_mode # 初始化M(目标维数)
        maxormins = [1 for _ in range(M)] # 初始化maxormins(目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标)
        Dim = len(var_space) # 初始化Dim(决策变量维数)
        varTypes = [1] * Dim # 这是一个list,初始化varTypes(决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的)
        lb = [0 for _ in range(len(var_space))]  # 决策变量下界
        lbin = [1 for _ in range(len(var_space))]  # 决策变量下边界(0表示不包含该变量的下边界，1表示包含)
        ubin = [1 for _ in range(len(var_space))]  # 决策变量上边界(0表示不包含该变量的上边界，1表示包含)
        ub = []
        self.core_area_space = [area_all[var_to_version(var_iter)[:-1]] for var_iter in var_space]
        for var_index, var_area in enumerate(self.core_area_space):
            max_num = int(AREA_CONSTRAIN / var_area)
            ub.append(max_num) # 决策变量上界
        self.statistics = {}
        self.statistics['aimFunc'] = 0
        self.statistics['schedule_result_database_num'] = 0
        self.statistics['hmp_metric_best'] = 9999
        self.statistics['hmp_size_list'] = []
        self.statistics['hmp_metric_best_list'] = []
        self.Xi = None
        self.Yi = None
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
     
    def aimFunc(self, pop):
        Vars = pop.Phen
        objV_metrics = []
        CV = []
        #print(f"aimFunc: Vars={Vars}")
        for var_each in Vars:
            area_sum = np.sum(self.core_area_space * np.asarray(copy.deepcopy(var_each)))
            var_each_valid_slice = 0 < np.asarray(var_each)
            var_each_valid = var_each[var_each_valid_slice]
            core_counts_flag = len(var_each_valid) - 1e-1 - MAX_CORE_TYPES
            #print(f"aimFunc: var_each={var_each}")
            if (self.statistics['schedule_result_database_num'] < N_SAMPLES_ALL_HMP) and \
                    ((0 < sum(var_each)) and (area_sum < AREA_CONSTRAIN) and (core_counts_flag < 0)):
                var_each_key = var_each.tostring()
                if var_each_key in schedule_result_database:
                    f_val_array = schedule_result_database[var_each_key]
                    metrics = []
                    for each_f_val in f_val_array:
                        metrics.append(get_hmp_metrics(delay=each_f_val[0], energy=each_f_val[1])[EVALUATION_INDEX])
                    #if 0 == multi_obj_mode:
                    #    metrics = metrics[0]
                else:
                    core_space = np.asarray(copy.deepcopy(self.var_space))[var_each_valid_slice]
                    core_types = ['Core-' + str(i) for i in range(len(core_space))]
                    core_counts = {f"{core_types[i]}": var_each_valid[i] for i in range(len(core_space))}
                    #print(f"aimFunc core_counts={core_counts}")
                    f_val, f_val2 = self.obj_func(self.program_bitmap, self.program_queue_ids,
                                             core_types=core_types, core_counts=core_counts,
                                             core_vars=core_space, use_eer=self.use_eer,
                                             models=None,
                                             multiobj_config={'exp_id': self.multiobj_config['exp_id'],
                                                              'sche_evaluation_index':self.multiobj_config['sche_evaluation_index']},
                                             )
                    if multi_obj_mode:
                        f_val_2, f_val2_2 = self.obj_func(self.program_bitmap, core_types=core_types, core_counts=core_counts,
                                                 core_vars=core_space, use_eer=False,
                                                 models=None,
                                                 multiobj_config={'exp_id': self.multiobj_config['exp_id']},
                                                 )
                        f_val_array = [[f_val, f_val2], [f_val_2, f_val2_2]]
                    else:
                        f_val_array = [[f_val, f_val2]]
                    schedule_result_database[var_each_key] = f_val_array

                    metrics = []
                    for each_f_val in f_val_array:
                        metrics.append(get_hmp_metrics(delay=each_f_val[0], energy=each_f_val[1])[EVALUATION_INDEX])
                    #if 0 == multi_obj_mode:
                    #    metrics = metrics[0]
                    #print(f"{self.statistics['schedule_result_database_num']} = {core_space} {core_counts} {self.program_bitmap} {each_f_val[0]}, {each_f_val[1]} {metrics[0]}")

                    if multi_obj_mode:
                        if self.Xi is not None:
                            self.Xi = np.append(self.Xi, [var_each], axis=0)
                            self.Yi = np.append(self.Yi, [metrics], axis=0)
                        else:
                            self.Xi = np.asarray([var_each])
                            self.Yi = np.asarray([metrics])
                        frontier_x, frontier_y = is_pareto_efficient_dumb(self.Xi, self.Yi)
                        hypervolume = DominatedPartitioning(
                            ref_point=torch.Tensor([-100, -100]),
                            Y=0 - torch.Tensor(frontier_y)
                        ).compute_hypervolume().item()
                        self.statistics['hmp_metric_best'] = min(self.statistics['hmp_metric_best'], hypervolume)
                    else:
                        self.statistics['hmp_metric_best'] = min(self.statistics['hmp_metric_best'], metrics[0])

                    self.statistics['schedule_result_database_num'] += 1
                    self.statistics['hmp_size_list'].append(self.statistics['schedule_result_database_num'])
                    self.statistics['hmp_metric_best_list'].append(self.statistics['hmp_metric_best'])

                #objV_metrics.append(metrics)
                self.statistics['aimFunc'] += 1
            else:
                metrics = [999 + area_sum for _ in range(1 + multi_obj_mode)]
            objV_metrics.append(metrics)
            CV.append([-area_sum, area_sum - AREA_CONSTRAIN, core_counts_flag])

        pop.ObjV = np.asarray(objV_metrics)
        pop.CV = np.asarray(CV)
        return self.statistics['aimFunc'], self.statistics['schedule_result_database_num']

