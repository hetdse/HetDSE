import sys
import time
from datetime import datetime
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch

import threading

import matplotlib.pyplot as plt

import skopt
from config import *
#from sklearn_DKL_GP import Sklearn_DKL_GP
# from skopt.plots import plot_gaussian_process, plot_evaluations, plot_convergence, plot_objective, plot_regret
from skopt import Optimizer, OptimizerMultiobj, Space
from skopt.space import Integer


from simulation_metrics import metric_1, metric_2, metric_3, evaluate_ADRS
from get_model import *

#transform = 'normalize'
transform = 'identity'
if 2304 == N_SPACE_SIZE:
    single_core_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache")])
elif 9216==N_SPACE_SIZE:
    single_core_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="decode_width")])
elif 36864==N_SPACE_SIZE:
    single_core_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="decode_width"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="rob")])
elif 147456 == N_SPACE_SIZE:
    single_core_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="decode_width"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="rob"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="freq")])
else:
    print(f"no def for N_SPACE_SIZE={N_SPACE_SIZE}")


def to_named_params(results, search_space):
    params = results.x
    param_dict = {}
    params_list = [(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = item[1]
    return param_dict


# class Problem_Model(threading.Thread):
class Problem_Model():

    def __init__(self, problem_space, surrogate_model_tag, surrogate_model_config,
                 multiobj_config, program_queue_info, schedule_mode, obj_info, model_load, tag):
        # threading.Thread.__init__(self)
        self.problem_space = problem_space
        self.surrogate_model_tag = surrogate_model_tag
        self.surrogate_model_config = surrogate_model_config
        base_estimator, base_estimator2, self.surrogate_model_dict = self.surrogate_model_config
        self.program_queue_info = program_queue_info
        self.program_queue_name, self.program_queue, self.program_bitmap, self.program_queue_ids = program_queue_info
        self.schedule_mode = schedule_mode
        self.obj_info = obj_info
        self.obj_func_num, self.obj_func, self.obj_args = obj_info
        self.core_space = None
        if self.obj_args:
            self.core_types, self.core_space, self.use_eer = self.obj_args
        self.multiobj_config = multiobj_config
        self.num = 0 #multiobj_config['exp_id']
        self.n_sample_all = self.multiobj_config['n_initial_points'] + self.multiobj_config['n_generation_points']

        self.reset_result()
        self.statistics = {}
        self.statistics['aimFunc'] = 0

        if self.schedule_mode:
            surrogate_model_tag_singlecore = self.surrogate_model_dict['tag']
            model_data_filename = 'model_data/' + self.program_queue_name + '_' + surrogate_model_tag_singlecore + '_' + tag
            self.dump_filename = model_data_filename
        else:
            surrogate_model_tag_hmp = self.surrogate_model_dict['tag']
            model_data_filename = 'model_data/' + self.program_queue[0] + '_' + surrogate_model_tag_hmp
            self.dump_filename = model_data_filename
        if 0 < self.multiobj_config['exp_id']:
            self.dump_filename += "-exp" + str(self.multiobj_config['exp_id'])
        self.dump_filename += '.skoptdump'
        if self.schedule_mode:        
            print(f"dump_filename={self.dump_filename}")

    def run_from_model(self):
        result = skopt.load(self.dump_filename)
        return result

    def run(self):
        if self.schedule_mode:
            real_pareto_data = None
        else:
            from get_real_pareto_frontier import get_pareto_optimality_from_file_ax_interface
            real_pareto_data = get_pareto_optimality_from_file_ax_interface(self.program_queue[0], multiobj_mode=multiobj_mode)
        # print(f"real_pareto={real_pareto}")
        surrogate_model_real_tag = self.surrogate_model_dict["tag"]
        if self.schedule_mode is None:
            self.surrogate_model_dict["tag"] += self.program_queue[0]
        self.surrogate_model_dict["tag"] += "-exp" + str(self.multiobj_config['exp_id'])

        result_filename_prefix = "log"
        #if 2304 != N_SPACE_SIZE:
        result_filename_prefix += "_" + str(N_SPACE_SIZE)
        result_filename_prefix +=  '/' + self.program_queue_name
        if not os.path.exists(result_filename_prefix):
            os.mkdir(result_filename_prefix)
        result_filename_prefix += '/' + self.multiobj_config['tag']
        if not os.path.exists(result_filename_prefix):
            os.mkdir(result_filename_prefix)
        if self.schedule_mode:
            result_filename_prefix +=  "/" + self.schedule_mode + '_' + surrogate_model_real_tag
            if not os.path.exists(result_filename_prefix):
                os.mkdir(result_filename_prefix)
        else:
            result_filename_prefix +=  "/" + surrogate_model_real_tag + '_' + self.program_queue[0]
            if not os.path.exists(result_filename_prefix):
                os.mkdir(result_filename_prefix)
        result_filename_prefix +=  "/exp-" + str(self.multiobj_config['exp_id'])

        np.random.seed(1234+self.multiobj_config['exp_id'])
        obj_func_num, obj_func, obj_args = self.obj_info
        _, core_space, use_eer = obj_args
        self.multiobj_config['use_eer'] = use_eer

        self.opt_gp = OptimizerMultiobj(
            dimensions=self.problem_space,
            surrogate_model_info = get_surrogate_model(self.surrogate_model_tag),
            n_random_starts=None,
            n_initial_points=self.multiobj_config['n_initial_points'],
            initial_point_generator=self.surrogate_model_dict['initial_point_generator'],
            n_jobs=-1,
            acq_func=self.surrogate_model_dict['acq_func'],
            acq_optimizer="full",  # "auto",
            random_state=self.multiobj_config['exp_id'],
            model_queue_size=1,
            acq_func_kwargs=None,  # {"xi": 0.000001, "kappa": 0.001} #favor exploitaton
            acq_optimizer_kwargs={"n_points": 10},
            real_pareto_data=real_pareto_data,
            n_generation_points=self.multiobj_config['n_generation_points'],
            multiobj_config=self.multiobj_config,
            mape_line_analysis=mape_line_analysis,
            program_queue_info=self.program_queue_info,
            schedule_mode=self.schedule_mode,
            core_space=self.core_space
        )

        startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
        last_time = 0
        #time_used_dataset_acc = 0
        self.time_register_acc = 0
        log_file = open(result_filename_prefix + ".log", "w")
        if self.surrogate_model_dict['all_workloads_per_iter_mode'] and (self.n_samples_all % len(self.program_queue)):
            print(f"N_SAMPLES_ALL({self.n_samples_all}) should be multiple of len(program_queue)={len(self.program_queue)}")

        ITER_MAX = self.multiobj_config['n_initial_points'] + self.multiobj_config['n_generation_points']
        for iter in range(ITER_MAX):
            startTime2 = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
            next_x, next_case_id = self.opt_gp.ask()
            if self.multiobj_config['sche_explore']:
                sche = next_x[-1]
                next_hmp_x = next_x[: -1]
            else:
                sche = None
                next_hmp_x = next_x
            #start_dataset_time = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
            #print(f"main iter_id={iter} next_x={next_x}")
            if 1 == obj_func_num:
                var_each_key = ''.join([str(s) for s in next_x])
                if var_each_key in schedule_result_database:
                    f_val, f_val2 = schedule_result_database[var_each_key]
                else:
                    if 'models_all_workloads' in self.multiobj_config:
                        if self.multiobj_config['metric_pred_for_schedule']:
                            models = self.multiobj_config['models_all_workloads']
                        else:
                            models = None
                    else:
                        models = None

                    core_ids = []
                    core_each_valid = []
                    core_space_valid = []
                    for each_id, each in enumerate(next_hmp_x):
                        if 0 < each:
                            core_ids.append(each_id)
                            core_each_valid.append(each)
                            core_space_valid.append(core_space[each_id])
                    core_types_valid = ['Core-' + str(i) for i in range(len(core_ids))]
                    core_counts = {f"Core-{i}": core_each_valid[i] for i in range(len(core_ids))}
                    #print(f"main iter_id={iter} core_counts={core_counts}")
                    f_val, f_val2 = obj_func(program_queue=self.program_bitmap, 
                                             program_queue_ids=self.program_queue_ids,
                                             core_types=core_types_valid, core_counts=core_counts,
                                             core_vars=core_space_valid, use_eer=use_eer,
                                             models=models,
                                             multiobj_config=self.multiobj_config,
                                             sche=sche,
                                             )
                    schedule_result_database[var_each_key] = (f_val, f_val2)
                self.statistics['aimFunc'] += 1
            else:
                metric_1, metric_2 = obj_func
                f_val = metric_1(next_x, case_names[next_case_id])
                f_val2 = metric_2(next_x, case_names[next_case_id])
            #time_used_dataset = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - start_dataset_time
            #time_used_dataset_acc += time_used_dataset.total_seconds()
            #print(f"time_used_dataset_acc={time_used_dataset_acc}")
            #print(f"before tell {time.localtime()}")
            if self.surrogate_model_dict['multi_est']:
                res = self.opt_gp.tell(next_x, f_val, f_val2)
            else:
                f_value_t = get_hmp_metrics(delay=f_val, energy=f_val2)[EVALUATION_INDEX]
                res = self.opt_gp.tell(next_x, f_value_t, 0)
            #print(f"after tell {time.localtime()}")
            if mape_line_analysis and (self.surrogate_model_dict['model_is_iterative'] is False):
                # model is not iterative, here should only count in the time of this iteration
                time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                          "%Y-%m-%d %H:%M:%S") - startTime2
            else:
                time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                          "%Y-%m-%d %H:%M:%S") - startTime
            #start_time_io = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")                
            #if self.surrogate_model_dict['model_is_iterative'] and print_info and (self.multiobj_config['n_initial_points'] < (iter + 2)):
                #print(f"sample={iter + 1} ADRS= {res.fun} [{time_used}]")
                #print(f"sample={iter + 1} metric= {res.result_array_sample_map['metric_ucb']} [{time_used}]")
            log_file.write(
                f"sample= {iter + 1} HV= {res.result_array_sample_map['HV']} "
                f"predict_error= {res.predict_error} predict_error_unlabelled_mape= {res.predict_error_unlabelled_mape} "
                #f"non_uniformity= {res.non_uniformity} "
                f"coverage = {res.coverage} "
                f"time_used= {time_used}\n")
            #time_io_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"%Y-%m-%d %H:%M:%S") - start_time_io
            #time_used_io_acc += time_io_used.total_seconds()

            '''
            if init_effect_analysis and (self.multiobj_config['n_initial_points'] == (iter + 1)):
                global result_init_alg
                result_init_alg[self.num] = res.fun
                if 0 == self.num and "random" != surrogate_model_dict['initial_point_generator'] and "sobol" != surrogate_model_dict['initial_point_generator']:
                    save_result_init_effect(surrogate_model_tag_real=surrogate_model_tag, surrogate_model_dict=surrogate_model_dict)
            '''

            if (self.multiobj_config['n_initial_points'] < (iter + 2)):
                if mape_line_analysis:
                    #self.result_array_sample_map['ADRS'][iter + 1][self.num] = res.fun
                    self.result_array_sample_map['Time'][iter + 1][self.num] = time_used.total_seconds()
                    self.result_array_sample_map['metric_ucb'][iter + 1][self.num] = res.result_array_sample_map['metric_ucb']                    
                    self.result_array_sample_map['HV'][iter + 1][self.num] = res.result_array_sample_map['HV']
                    self.result_array_sample_map['IGD'][iter + 1][self.num] = res.result_array_sample_map['IGD']
                    '''
                    # print(f"predict_error={res.predict_error}")
                    result_mape_array1_samples[iter + 1][self.num] = res.predict_error[0]
                    result_mape_array2_samples[iter + 1][self.num] = res.predict_error[1]
                    result_unlabelled_mape_array1_samples[iter + 1][self.num] = 0#res.predict_error_unlabelled_mape[0]
                    result_unlabelled_mape_array2_samples[iter + 1][self.num] = 0#res.predict_error_unlabelled_mape[1]
                    result_rsquare1_samples[iter + 1][self.num] = 0#res.Rsquare[0]
                    result_rsquare2_samples[iter + 1][self.num] = 0#res.Rsquare[1]
                    result_non_uniformity_samples[iter + 1][self.num] = opt_gp.statistics['non_uniformity_explore']
                    result_coverage[iter + 1][self.num] = res.coverage
                    result_hv[iter + 1][self.num] = opt_gp.statistics['hv'][-1]
                    result_hv_acq_stuck[iter + 1][self.num] = opt_gp.statistics['hv_acq_stuck']
                    result_hv_last_iter[iter + 1][self.num] = opt_gp.statistics['hv_last_iter']
                    '''

                    for each in self.result_time_breakdown_over_iterations:
                        self.result_array_sample_map[each][iter + 1][self.num] = res.result_array_sample_map[each]
                self.time_register_acc += res.result_array_sample_map['time_register']
                #print(f"time_register_acc= {time_register_acc}")
                if mape_line_analysis:
                  self.result_array_sample_map['time_valid'][iter + 1][self.num] = self.result_array_samples['Time'][iter + 1][self.num] - self.time_register_acc
                else:
                  self.result_array_sample_map['time_valid'][iter + 1][self.num] = time_used.total_seconds()
                #result_array_sample_map['time_io_acc'][iter + 1][self.num] = time_used_io

        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime
        print(result_filename_prefix)
        final_result = self.opt_gp.get_result()
        final_result.used_map['register'] = self.time_register_acc
        final_result.used_map['current_pareto_y'] = self.opt_gp.current_pareto_y
        final_result.used_map['opt_gp'] = self.opt_gp
        final_result.result_array_sample_map = self.result_array_sample_map
        print(f"core_space_mask final= {np.count_nonzero(self.opt_gp.core_space_mask) / len(self.opt_gp.core_space_mask)}")
        #global result_array
        #global result_time_array
        #global result_mape_array1, result_mape_array2
        #global result_unlabelled_mape_array1, result_unlabelled_mape_array2
        #global result_rsquare1, result_rsquare2
        #self.result_array['ADRS'][self.num] = final_result.fun
        #self.result_array['IGD'][self.num] = self.result_array_sample_map['IGD'][-1][self.num]
        if self.schedule_mode is None:
            self.result_array['Time'][self.num] = self.result_array_sample_map['time_valid'][-1][self.num]
            self.result_array['MAPE1'][self.num] = final_result.predict_error[0]
            self.result_array['MAPE2'][self.num] = final_result.predict_error[1]        
            self.result_array['Unlabelled_MAPE1'][self.num] = final_result.predict_error_unlabelled_mape[0]
            self.result_array['Unlabelled_MAPE2'][self.num] = final_result.predict_error_unlabelled_mape[1]
            self.result_array['Rsquare1'][self.num] = final_result.Rsquare[0]
            self.result_array['Rsquare2'][self.num] = final_result.Rsquare[1]
        '''
        result_hv_last[self.num] = opt_gp.statistics['hv'][-1]       
        result_uniexplore[self.num] = opt_gp.statistics['non_uniformity_explore']
        '''
        self.statistics['last_mean'] = self.opt_gp.result_array_sample_map['last_mean']
        self.statistics['last_ucb'] = self.opt_gp.result_array_sample_map['last_ucb']

        log_file.write(f"time_used={time_used}\n")
        log_file.write(f"result={final_result}")
        #print(f"ADRS= {result_array[self.num]}")
        print(f"time_used={time_used}")

        '''
        print(f"[OUT] semi_train = {self.opt_gp.statistics['semi_train_accumulation']} "
              f"ucb = {opt_gp.statistics['ucb']} "
              f"cv_ranking_beta= {opt_gp.statistics['cv_ranking_beta']} "
              f"non_uniformity_explore = {opt_gp.statistics['non_uniformity_explore']} "
              f"igd = {res.igd} "              
              f"init_sample_time_used= {opt_gp.statistics['init_sample_time_used']} \n")
        log_file.write(f"[OUT] semi_train = {opt_gp.statistics['semi_train_accumulation']} "
              f"ucb = {opt_gp.statistics['ucb']} "
              f"cv_ranking_beta= {opt_gp.statistics['cv_ranking_beta']} "
              f"non_uniformity_explore = {opt_gp.statistics['non_uniformity_explore']} "
              f"init_sample_time_used= {opt_gp.statistics['init_sample_time_used']} \n")
        #print(f"[OUT] non_uniformity real / last = {opt_gp.real_pareto_data_non_uniformity} / {opt_gp.non_uniformitys[-1]}")
        #log_file.write(f"[OUT] non_uniformity real / last = {opt_gp.real_pareto_data_non_uniformity} / {opt_gp.non_uniformitys[-1]} \n")
        '''
        log_file.close()
        if self.schedule_mode is None:
            self.save_result(surrogate_model_tag_real=self.surrogate_model_dict['tag'], program_queue_name=self.program_queue_name)

        save_feature_importances(result_filename_prefix, self.opt_gp.feature_importances)
        if False:
            fig = plt.figure()
            fig.suptitle(surrogate_model)
            plot_convergence(self.opt_gp.get_result())
            plt.plot()
            plt.show()
        if False:
            plot_objective(self.opt_gp.get_result(), n_points=10)
            # plot_regret(opt_gp.get_result())
            plt.show()

        if self.schedule_mode is None:
            skopt.dump(final_result, self.dump_filename, store_objective=True)
        return final_result


    def save_line_file(self, case_name, surrogate_model_tag_real, result_array_samples, data_name):
        if self.schedule_mode is None:
            result_filename_prefix = ''
            # if 2304 != N_SPACE_SIZE:
            result_filename_prefix += str(self.n_sample_all) + "_"
            result_mape_line_file = open(
                "log_summary_mape_line/" + result_filename_prefix + case_name + "_" + data_name + ".txt", "a")
            result_mape_line_file.write("%-40s %2d " % (surrogate_model_tag_real, len(result_array_samples),))
            for sample_iter in range(self.multiobj_config['n_initial_points'], self.n_sample_all + 1):
                # print(f"result_array_samples[{sample_iter}]={result_array_samples[sample_iter]}")
                result_mape_line_file.write(
                    "%-10f %-10f " % (result_array_samples[sample_iter].mean(), ci(result_array_samples[sample_iter]),))
            result_mape_line_file.write('\n')
            result_mape_line_file.close()


    def reset_result(self):
        self.result_array = {}
        self.evaluation_metrics = ['Time', 'metric_ucb', 'HV', 'IGD', 'ADRS', 'MAPE1', 'MAPE2', 'Unlabelled_MAPE1', 'Unlabelled_MAPE2', 'Rsquare1', 'Rsquare2', 'uniexplore', 'HV_last']
        for each in self.evaluation_metrics:
            self.result_array[each] = np.zeros(n_experiment)

        self.result_array_sample_map = {}
        if mape_line_analysis:
            self.evaluation_metrics_over_iterations = evaluation_metrics + ['non_uniformity', 'coverage', 'HV', 'HV_acq_stuck']
            for each in self.evaluation_metrics_over_iterations:
                self.result_array_sample_map[each] = np.zeros([self.n_sample_all + 1, n_experiment])

        self.result_time_breakdown_over_iterations = ['time_train', 'time_infer', 'time_explore', 'time_pareto',
                                                      'time_hv', 'time_filter', 'filter_ratio', 'time_pred_pareto',
                                                      'time_register', 'time_hmp', 'time_valid',
                                                      ]
        for each in self.result_time_breakdown_over_iterations:
            self.result_array_sample_map[each] = np.zeros([self.n_sample_all + 1, n_experiment])
        # result_array_sample_map['time_io_acc'] = np.zeros([self.n_samples_all + 1, n_experiment])
        # result_array_sample_map['time_dataset_acc'] = np.zeros([self.n_samples_all + 1, n_experiment])

        #if init_effect_analysis:
        #    result_init_alg = np.zeros(n_experiment)


    def save_result(self, surrogate_model_tag_real, program_queue_name):
        result_filename_prefix = ''
        # if 2304 != N_SPACE_SIZE:
        result_filename_prefix += str(N_SPACE_SIZE) + "_"
        result_filename_prefix += self.multiobj_config['tag'] + "_" 
        if self.schedule_mode:
            result_filename_prefix += self.schedule_mode + "_"
        else:
            result_filename_prefix += 'pred' + "_"
        result_filename_prefix += program_queue_name + "_"
        result_summary_file = open('log_summary' + host_tag + '/' + result_filename_prefix + "-summary.txt", "a")
        result_summary_file.write(
            "%-60s %-15s %-2d %-10f %-10f %-10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %10f %7f %7f %7f %7f \n" \
            % (surrogate_model_tag_real,
               hostname, len(self.result_array['Time']),
               self.result_array['ADRS'].mean(), ci(self.result_array['ADRS']),
               self.result_array['Time'].mean(), ci(self.result_array['Time']),
               self.result_array['MAPE1'].mean(), ci(self.result_array['MAPE1']),
               self.result_array['MAPE2'].mean(), ci(self.result_array['MAPE2']),               
               self.result_array['Unlabelled_MAPE1'].mean(), ci(self.result_array['Unlabelled_MAPE1']),
               self.result_array['Unlabelled_MAPE2'].mean(), ci(self.result_array['Unlabelled_MAPE2']),
               self.result_array['Rsquare1'].mean(), ci(self.result_array['Rsquare1']),
               self.result_array['Rsquare2'].mean(), ci(self.result_array['Rsquare2']),
               self.result_array['uniexplore'].mean(), ci(self.result_array['uniexplore']),
               self.result_array['HV_last'].mean(), ci(self.result_array['HV_last']),
               self.result_array['IGD'].mean(), ci(self.result_array['IGD']),
               )
        )
        result_summary_file.close()

        if mape_line_analysis:
            print(f"mape_line: surrogate_model= {surrogate_model_tag_real}")
            for each in self.evaluation_metrics_over_iterations:
                self.save_line_file(program_queue_name, surrogate_model_tag_real, self.result_array_sample_map[each], data_name=each)
            #print(f"hv_acq_stuck= {np.mean(result_hv_acq_stuck[-1, :])}")

        for each in self.result_time_breakdown_over_iterations:
            self.save_line_file(program_queue_name, surrogate_model_tag_real, self.result_array_sample_map[each], data_name=each)

    '''
    def save_result_init_effect(self, surrogate_model_tag_real, surrogate_model_dict):
        if init_effect_analysis:
            global result_init_alg
            result_init_alg_index = result_init_alg > 0
            result_init_alg_valid = result_init_alg[result_init_alg_index]
            print(
                f"init_algo= {surrogate_model_dict['initial_point_generator']} result= {result_init_alg_valid.mean()} ci= {ci(result_init_alg_valid)}")
            result_init_alg_summary_file = open("log_summary" + "/init_algo_summary" + '_' + str(N_SPACE_SIZE) + ".txt", "a")
            result_init_alg_summary_file.write(
                "%-15s %-10f %-10f %2d %-10s %-40s \n" % (surrogate_model_dict['initial_point_generator'],
                                                          result_init_alg_valid.mean(), ci(result_init_alg_valid),
                                                          len(result_init_alg_valid),
                                                          'x',
                                                          surrogate_model_tag_real),
            )
            result_init_alg_summary_file.close()
    '''

def save_feature_importances(result_filename_prefix, feature_importances):
    if feature_importances is not None:
        feature_importances_file = open("log_summary/feature_importances.txt", "a")
        feature_importances_file.write(result_filename_prefix + " ")
        for each in feature_importances[0]:
            feature_importances_file.write(str(each) + " ")
        for each in feature_importances[1]:
            feature_importances_file.write(str(each) + " ")
        feature_importances_file.write("\n")
        feature_importances_file.close()


def ci(y):
    # 95% for 1.96
    return 1.96 * y.std(axis=0) / np.sqrt(len(y))


'''
def multi_exp_problem_model(surrogate_model_tag):
    reset_result()
    experiment_range = range(n_experiment)
    if exp_id is not None:
        experiment_range = range(exp_id, exp_id + 1)
    program_queue_name = ''
    for thread_i in experiment_range:
        program_queue_info = get_workloads_all(random_seed=thread_i)
        program_queue_name, program_queue, program_bitmap, program_queue_ids = program_queue_info
        surrogate_model_config = get_surrogate_model(surrogate_model_tag)
        if 0 == thread_i or (exp_id is not None and exp_id == thread_i):
            _, _, surrogate_model_tag_real, surrogate_model_dict = surrogate_model_config
            print("running " + program_queue_name + " " + surrogate_model_tag_real)
        #problem_model = Problem_Model(surrogate_model_tag, surrogate_model_config, thread_i, program_queue_info)
        #problem_model.run()

    # filter failed cases
    valid_index = result_array > 0
    self.result_array['Time'] = self.result_array['Time'][valid_index]
    print(f"surrogate_model= {surrogate_model_tag_real} result= {self.result_array['Time'].mean()} ci= {ci(self.result_array['Time'])}")
    #save_result(surrogate_model_tag_real=surrogate_model_tag_real, program_queue_name=program_queue_name)
    #save_result_init_effect(surrogate_model_tag_real=surrogate_model_tag_real,
                            #surrogate_model_dict=surrogate_model_dict)


from config import surrogate_model_tag_list
if __name__ == '__main__':
    for surrogate_model_tag in surrogate_model_tag_list:
        multi_exp_problem_model(surrogate_model_tag)
'''