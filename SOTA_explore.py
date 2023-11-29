import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from config import *
from get_real_pareto_frontier import is_pareto_efficient_dumb
from simulation_metrics import *
from hmp_evaluate import hmp_evaluate, get_all_hmp, design_point_selection


def get_core_space(core_space_tag, program_queue=None, selection=None, random_seed=0):
    core_space_metrics = None
    if 'C_1' == core_space_tag:
        core_config_space = [var_BOOMv2]
        core_space = [x[:-1] for x in core_config_space]
    elif 'C_2' == core_space_tag:
        core_config_space = [var_BOOMv2, var_ROCKET]
        core_space = [x[:-1] for x in core_config_space]
    elif 'C_3' == core_space_tag:
        core_config_space = [var_YANQIHU, var_BOOMv2, var_ROCKET]
        core_space = [x[:-1] for x in core_config_space]
    elif 'C_5' == core_space_tag:
        core_config_space = [var_YANQIHU, var_BOOMv2, var_ROCKET] # ...
        core_space = [x[:-1] for x in core_config_space]        
    elif 'C_pareto' == core_space_tag:
        core_space = []
        core_space_metrics0 = []
        core_space_metrics1 = []
        #core_space_all = get_all_design_point()
        from get_real_pareto_frontier import get_pareto_optimality_from_file_ax_interface
        for each_program in program_queue:
            real_pareto_data = get_pareto_optimality_from_file_ax_interface(each_program, multiobj_mode=multiobj_mode)
            pareto_x = [x[:-1] for x in real_pareto_data[2]]
            if 'kmeansy-scale' == selection:
                pareto_y = [y / max(real_pareto_data[0]) for y in real_pareto_data[0]]
                pareto_y2 = [y / max(real_pareto_data[1]) for y in real_pareto_data[1]]
            else:
                pareto_y = real_pareto_data[0]
                pareto_y2 = real_pareto_data[1]
            core_space += pareto_x
            core_space_metrics0 += pareto_y
            core_space_metrics1 += pareto_y2
            '''
            if core_space is not None:
                core_space = np.append(core_space, pareto_x, axis=0)
                core_space_metrics0 += real_pareto_data[0] #np.append(core_space_metrics0, [0], axis=0)
                core_space_metrics1 += real_pareto_data[1] #np.append(core_space_metrics1, real_pareto_data[1], axis=0)
                if 'kmeansy-scale' == selection:
                    real_pareto_data[0]
            else:
                core_space = pareto_x
                core_space_metrics0 = real_pareto_data[0]
                core_space_metrics1 = real_pareto_data[1]
            '''
        print(f"get_core_space: core_space size = {len(core_space)}")
        core_space, unique_index = np.unique(core_space, return_index=True, axis=0)
        core_space_metrics = [[y0, y1] for y0, y1 in zip(core_space_metrics0, core_space_metrics1)]
        core_space_metrics = np.asarray(core_space_metrics)[unique_index]
        print(f"get_core_space: core_space size after unique = {len(core_space)}")
    elif 'C_random_pareto' in core_space_tag:
        core_space_all = get_all_design_point()
        slice_ids = np.arange(len(core_space_all))
        random.seed(random_seed)
        random.shuffle(slice_ids)
        samples_num = int(core_space_tag.split('-')[1])
        core_space_random = np.asarray(core_space_all)[slice_ids[:samples_num]]
        core_space = []
        core_space_metrics = []
        for each_program in program_queue:
            core_space_random_metrics = [[*metric_1_2(each_x, each_program)] for each_x in core_space_random]
            pareto_optimal_sets, pareto_frontier_y = is_pareto_efficient_dumb(core_space_random, np.asarray(core_space_random_metrics))
            core_space += [x[:-1] for x in pareto_optimal_sets]
            core_space_metrics += pareto_frontier_y.tolist()
        print(f"get_core_space: core_space size = {len(core_space)}")
        core_space, unique_index = np.unique(core_space, return_index=True, axis=0)
        core_space_metrics = np.asarray(core_space_metrics)[unique_index]
        print(f"get_core_space: core_space size after unique = {len(core_space)}")
    elif '' == core_space_tag:
        core_space = None
    else:
        print(f"no def core_space_tag={core_space_tag}")
        exit(0)
    return core_space, core_space_metrics


if __name__ == '__main__':
    workload_set_id_list = [0] #[0, 1, 2, 3, 4]
    if 2 < len(sys.argv):
        workload_set_id_list = [int(sys.argv[2])] # for batch running, argv[2] is workload_set_id
        #workload_set_id_list = [-int(sys.argv[2])] # for model_data generation argv[2] is case_id
    for workload_set_id in workload_set_id_list:
        program_queue_info = get_workloads_all(random_seed=workload_set_id)
        program_queue_name, program_queue, program_bitmap, program_queue_ids = program_queue_info
        #print(f"program_queue_name={program_queue_name}")

        if 1 < len(sys.argv):
            if 'BF' == sys.argv[1]:
                SOTA = 'BruteForce'
            else:
                SOTA = sys.argv[1]
        else:
            SOTA = 'BruteForce'
            #SOTA = 'GA'
            #SOTA = 'SA'
            #SOTA = 'SA_pc'
            #SOTA = 'HeterDSE'
            #SOTA = 'HeterDSE_pc'
            #SOTA = 'HeterDSE_pc_m'
            #SOTA = 'HeterDSE_m'
            #SOTA = 'HeterDSE_awm'
            #SOTA = 'HeterDSE_tf'
            #SOTA = 'HeterDSE_pc_tsb'

        selection = None
        #for selection in ['Random', 'Hetsel-birch']:
        #for selection in ['Random']:
        #for selection in ['Hetsel-kmeans']:
        #for selection in ['Hetsel-clique']:
        #for selection in ['Random','Hetsel-sort']:
        #for selection in ['Hetsel-sort5']:
        for selection in [None]:
        #for selection in ['Hetsel-rank2']:
            #core_space_tag = 'C_1'
            #core_space_tag = 'C_2'
            #core_space_tag = 'C_3'
            ##core_space_tag = 'C_5'
            core_space_tag = 'C_pareto'
            #core_space_tag = 'C_random_pareto-100' # LUCIE TACO-2016
            target_core_num = 3

            if 'HeterDSE' in SOTA and '_pc' in SOTA:
                core_space_tag = ''
                #selection = 'kmeansx'
                #selection = 'kmeansy-scale'
                #selection = 'LUCIE'
                #selection = 'Clustering3'
                #selection = 'Random'
                target_core_num = 30
            elif '_pareto' in core_space_tag:
                #core_space_tag = ''
                #selection = None
                #selection = 'kmeans'
                #selection = 'kmeansy'
                #selection = 'kmeansy-scale'
                #selection = 'LUCIE'
                selection = 'Clustering'
                target_core_num = 3

            #if 5 != EVALUATION_INDEX:
            #    sche_evaluation_index_list = [0,EVALUATION_INDEX,5]
            #else:
            #    sche_evaluation_index_list = [0,5]
            #for sche_evaluation_index in [EVALUATION_INDEX]:
            #for sche_evaluation_index in [0]:
            #sche_evaluation_index_list = [-1,0,2,3,5,7,8]
            sche_evaluation_index_list = [0]
            for sche_evaluation_index in sche_evaluation_index_list:
                if 0 == multi_obj_mode:
                    if -1 == sche_evaluation_index:
                        schedule_mode = 'Naive'
                    else:
                        schedule_mode = 'EER'
                        if 0 < sche_evaluation_index:
                            schedule_mode += '-' + str(sche_evaluation_index)
                elif 1 == multi_obj_mode:
                    schedule_mode = 'EER-Naive'
                    print(f"not support now")
                    exit(0)

                #exp_info = 0 #workload_set_id if -1 < workload_set_id else 0
                #exp_info_list = [0] #range(1)
                if len(program_queue) < 2:
                    exp_info_list = [0]
                else:
                    if 5 < len(sys.argv):
                        exp_info_list = [int(sys.argv[5])]
                    else:
                        #exp_info_list = [0, 1, 2, 3, 4]
                        exp_info_list = [0]                        
                for exp_info in exp_info_list:
                    core_space, core_space_metrics = get_core_space(core_space_tag,
                                                                    program_queue=program_queue,
                                                                    selection=selection,
                                                                    random_seed=exp_info)

                    tag = schedule_mode + '_' + core_space_tag
                    #if 0 < AREA_CONSTRAIN_VERSION:
                    tag += '_AC' + str(AREA_CONSTRAIN_VERSION)
                    #if 2 != EVALUATION_INDEX:
                    tag += '_EI' + str(EVALUATION_INDEX)
                    #if 0 < exp_info:
                    tag += '_EX' + str(exp_info)

                    hmp_config, hmp_time, hmp_energy, hmp_schedule_time_used, result_name = \
                        hmp_evaluate(program_queue_info=program_queue_info,
                                     core_space=core_space,
                                     core_space_metrics=core_space_metrics,
                                     tag=tag,
                                     SOTA=SOTA,
                                     exp_info=exp_info,
                                     selection=selection,
                                     target_core_num=target_core_num,
                                     schedule_mode=schedule_mode,
                                     sche_evaluation_index=sche_evaluation_index,
                                     )

                    hmp_config = np.asarray(hmp_config)
                    print(f"hmp_schedule_time_used = {hmp_schedule_time_used}")

                    file_name = 'log_summary' + host_tag + '/' + program_queue_name
                    if not os.path.exists(file_name):
                        os.mkdir(file_name)
                    file_name += '/' + schedule_mode + '.txt'
                    if os.path.exists(file_name):
                        file = open(file_name, 'a')
                    else:
                        file = open(file_name, 'a')
                        file.write(f'%100s %10s %10s %10s %10s %15s \n' % ('name', 'hmp_time','hmp_energy', 'ED','time_used','hostname'))

                    file.write(f'%100s %10.5f %10.5f %10.5f %10d %15s \n' % (result_name, hmp_time, hmp_energy, hmp_time*hmp_energy, hmp_schedule_time_used, hostname))
                    #file.write(f"program_queue_name = {program_queue_name} \n")
                    #file.write(f"core_space_tag = {core_space_tag} \n")
                    file.close()
                    print(f"result_name={result_name}")
