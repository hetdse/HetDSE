import os

import numpy as np

from GenericAlgorithm import GenericAlgorithm
import geatpy as ea

from hmp_evaluate import get_all_hmp
from schedule_function import schedule
from config import host_tag, N_SAMPLES_ALL_HMP


class GA_explore():

    def __init__(self, problem_space, surrogate_model_tag, surrogate_model_config, multiobj_config,
                 program_queue_info, program_queue_ids,
                 schedule_mode, obj_info):
        # threading.Thread.__init__(self)
        self.problem_space = problem_space
        self.surrogate_model_tag = surrogate_model_tag
        self.surrogate_model_config = surrogate_model_config
        self.program_queue_info = program_queue_info
        self.program_queue_name, self.program_queue, self.program_bitmap, self.program_queue_ids = program_queue_info
        self.schedule_mode = schedule_mode
        self.obj_info = obj_info
        self.obj_func_num, self.obj_func, self.obj_args = obj_info
        self.core_types, self.core_space, self.use_eer = self.obj_args
        self.multiobj_config = multiobj_config
        self.num = multiobj_config['exp_id']
        self.statistics = {}
        #self.n_sample_all = self.multiobj_config['n_initial_points'] + self.multiobj_config['n_generation_points']
        self.hmp_all, self.result_map = get_all_hmp(self.core_space)
        """================================实例化问题对象==========================="""
        self.problem = GenericAlgorithm(var_space=self.core_space, obj_info=obj_info, multiobj_config=multiobj_config, program_bitmap=self.program_bitmap, program_queue_ids=program_queue_ids)
        self.history = {'f': [], 'aimFunc': [], 'schedule_result_database': []}

    def run(self):
        """==================================种群设置==============================="""
        Encoding = 'RI'  # 编码方式
        NIND = self.multiobj_config['pop']  # 种群规模
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges, self.problem.borders)  # 创建区域描述器

        self.hmp_all = np.asarray(self.hmp_all)
        case_ids = np.arange(0, len(self.hmp_all))
        import random
        #random_seed = 0
        random.seed(self.num)
        random.shuffle(case_ids)
        prepared_pop = self.hmp_all[case_ids[:NIND]]
        #print(f"prepared_pop={prepared_pop}")
        if len(self.hmp_all) < NIND:
            print(f'GA_explore.py hmp_all len={len(self.hmp_all)} < pop={NIND}, exit')
            exit(0)

        population = ea.Population(Encoding, Field, NIND, prepared_pop)  # 实例化种群对象(此时种群还没被初始化，仅仅是完成种群对象的实例化)
        """================================算法参数设置============================="""
        myAlgorithm = ea.soea_DE_rand_1_L_templet(self.problem, population)
        myAlgorithm.MAXGEN = self.multiobj_config['gen']  # 最大进化代数
        myAlgorithm.schedule_result_database_num_max = N_SAMPLES_ALL_HMP
        myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
        myAlgorithm.recOper.XOVR = 0.7  # 重组概率
        myAlgorithm.drawing = 0
        myAlgorithm.logTras = 1
        """===========================调用算法模板进行种群进化======================="""
        run_result = myAlgorithm.run()  # 执行算法模板
        self.statistics = self.problem.statistics
        #print(run_result)
        [BestIndi, obj_trace] = run_result
        #population.save()  # 把最后一代种群的信息保存到文件中
        # 输出结果
        best_gen = np.argmin(self.problem.maxormins * obj_trace.ObjV)  # 记录最优种群个体是在哪一代
        best_ObjV = obj_trace.ObjV[best_gen]

        self.history['f'] = myAlgorithm.log['f_opt']
        self.history['aimFunc'] = myAlgorithm.log['aimFunc_num']
        self.history['gen'] = myAlgorithm.log['gen']
        self.history['schedule_result_database'] = myAlgorithm.log['schedule_result_database']

        file_name = 'log_hmp' + host_tag + '/' + self.program_queue_name
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        file_name += '/' + self.multiobj_config['result_name'] + '_log.txt'

        file = open(file_name, 'w')
        file.write(f"BestIndi = {obj_trace.Chrom[best_gen]} \n")
        file.write('best_ObjV: %s \n' % (best_ObjV))
        file.write(f"obj_trace = {obj_trace} \n ")
        file.write(f"aimFunc #= {self.problem.statistics['aimFunc']} \n ")
        #file.write(f"schedule_result_database #= {self.problem.statistics['schedule_result_database']} \n ")
        if 'f_opt' in myAlgorithm.log:
            file.write(f"f_opt = {myAlgorithm.log['f_opt']} \n ")
        file.close()

        legal_index = (obj_trace.CV < 0).all(axis=1)
        hmp_configs = obj_trace.Chrom[legal_index]
        return hmp_configs

