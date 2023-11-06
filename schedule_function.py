import copy
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

import numpy as np

from config import case_names, get_hmp_metrics
from simulation_metrics import delay_metric, energy_metric, power_metric, metric_1_2

'''
test_mode = False
if not test_mode:
    from simulation_metrics import *
'''

def schedule(program_queue, program_queue_ids, core_types, core_counts, core_vars, use_eer=False, multiobj_config=None, models=None, sche=None):
    # init config
    program_sequence = program_queue
    real_freq = [1e9, 1.5e9, 2e9, 3e9]
    frequency_range = [x for x in range(len(real_freq))]
    instruction_count = 10000000
    if sche is None:
        sche_evaluation_index = multiobj_config['sche_evaluation_index']
    else:
        sche_evaluation_index = sche

    random.seed(multiobj_config['exp_id'])

    # EER is proposed in paper "21-TC-HFEE" Fairness-Aware Energy Efficient Scheduling on Heterogeneous Multi-Core Processors

    # get performance metrics from dataset
    def read_data(program_id, core_id, frequency):
        program = program_id #program_queue_ids[program_id]
        core_var_copy = copy.deepcopy(core_vars[core_id])
        core_var_copy = np.append(core_var_copy, [frequency], axis=0)
        #print(f"core_vars={core_vars} core_id={core_id} core_version={core_var}")
        if models:
            if multiobj_config['aggr_workloads_models']:
                est, est2 = models
                predic_instance = core_var_copy.copy()
                predic_instance = np.append(predic_instance, [program], axis=0)
                processing_time = est.predict([predic_instance])[0]
                energy_consumption = est2.predict([predic_instance])[0]
            else:
                est, est2 = models[program]
                predic_instance = copy.deepcopy(core_var_copy)
                processing_time = est.predict([predic_instance])[0]
                energy_consumption = est2.predict([predic_instance])[0]
            from math import isnan
            if isnan(processing_time) or processing_time < 0 or isnan(energy_consumption) or energy_consumption < 0:
                print(f"[error] model predict: {core_var_copy} => {processing_time, energy_consumption}")
                processing_time = 999
                energy_consumption = 999
        else:
            case_name = case_names[program]
            processing_time, energy_consumption = metric_1_2(core_var_copy, case_name) # delay, energy
            #print(f"core_vars={core_vars}")
            #print(f"{case_name} {core_vars[core_id]} {core_id} {processing_time} {energy_consumption}")
            #exit(0)
            #processing_time = delay_metric(core_var_copy, case_name) # random.randint(1, 10) // (freq_factor)
            #power = power_metric(core_var_copy, case_name) # W = J/second
            #energy_consumption = power * processing_time # random.uniform(min_pow, max_pow) * (freq_factor)

        return energy_consumption, processing_time

    # test, random the metrics
    def read_data_from_dataset(program, core_id, frequency):
        core_performance = [4.5, 3, 1.5]
        core_power = [4.5, 3, 1.5]
        freq_factor = 1 + frequency * 0.25

        #instruction_count = 1e6 #core_performance[core_id] * 1e9 * freq_factor
        energy_consumption = core_power[core_id] * freq_factor
        processing_time = ((program + 1) / (core_performance[core_id] * freq_factor)) + 20
        return energy_consumption, processing_time

    def calculate_eer(program, core_id, freq_id):
        '''
        if test_mode:
            energy, delay = read_data_from_dataset(program, core_id, freq_id)
        else:
        '''
        # for all metrics: smaller is better
        energy, delay = read_data(program, core_id, freq_id)
        if 0 < sche_evaluation_index:
            EER_i = get_hmp_metrics(delay, energy)[sche_evaluation_index]
        else:
            #EER_i = IPC / power, negative that smaller is better
            EER_i = - instruction_count / real_freq[freq_id] / energy
        '''
        core_id_big = 0
        core_id_little = len(core_types) - 1
        eer_i = []
        for core_id in [core_id_big, core_id_little]:
            freq_id = len(real_freq) - 1
            if test_mode:
                inst, energy, delay = read_data_from_dataset(program, core_id, freq_id)
            else:
                inst, energy, delay = read_data(program, core_id, freq_id)
            #delay = CPI * 10000000 / frequency
            #energy = power * delay

            #CPI = delay / 10000000 * frequency
            #power = energy / delay
            # eer_i = IPC / power = 1 / CPI / power
            eer_i.append(10000000 / real_freq[freq_id] / energy)

        EER_i = eer_i[0] / eer_i[1]
        '''
        return EER_i

    def assign_core(remaining_program_sequence, core_type=None, use_eer=False, eer_results=None):
        if use_eer:
            min_eer = 999
            min_eer_program = -1
            min_eer_frequency = -1

            for i, program in enumerate(remaining_program_sequence):
                if program == 1:
                    for freq, eer in eer_results[core_type].items():
                        if eer[i] < min_eer:
                            min_eer = eer[i]
                            min_eer_program = i
                            min_eer_frequency = freq

            if min_eer_program != -1:
                remaining_program_sequence[min_eer_program] = 0

            return min_eer_program, min_eer_frequency, None
        else:
            chosen_program = -1
            chosen_frequency = None

            for i, program in enumerate(remaining_program_sequence):
                if program > 0:
                    chosen_program = i
                    remaining_program_sequence[chosen_program] = 0
                    chosen_frequency = random.choice(frequency_range)
                    break

            return chosen_program, chosen_frequency, None

    def execute_program(core_type, core_id, program, freq, processing_time, total_processing_time,
                        total_power_consumption, total_instructions, total_cycles):
        #print(f"  Core: {core_type}-{core_id} is now assigned to program {program} with frequency {freq} and processing time {processing_time:.2f}")
        total_processing_time[f"{core_type}-{core_id}"] += processing_time_results[core_type][freq][program]
        total_power_consumption[f"{core_type}-{core_id}"] += power_consumption_results[core_type][freq][program]
        total_instructions[f"{core_type}-{core_id}"] += instruction_count #instruction_results[core_type][freq][program]
        total_cycles[f"{core_type}-{core_id}"] += processing_time_results[core_type][freq][program] * real_freq[freq]

        num_iterations = 100
        for _ in range(num_iterations):
            current_total_time = total_processing_time[f"{core_type}-{core_id}"]
            min_total_time = min(total_processing_time.values())
            if (current_total_time == min_total_time) or (sum(remaining_program_sequence) == 0):
                idle_cores_queue.put((core_type, core_id))
                break
            time.sleep(0.001)
        return core_type, core_id

    #start_time = time.time()
    # core_states = {core_type: "idle" for core_type in core_counts}

    eer_results = {core: {freq: [] for freq in frequency_range} for core in core_types}
    processing_time_results = {core: {freq: [] for freq in frequency_range} for core in core_types}
    power_consumption_results = {core: {freq: [] for freq in frequency_range} for core in core_types}
    #instruction_results = {core: {freq: [] for freq in frequency_range} for core in core_types}

    for program_id, program in enumerate(program_sequence):
        for core_id, core in enumerate(core_types):
            for freq in frequency_range:
                if 1 == program:
                    '''
                    if not test_mode:
                        energy_consumption, processing_time = read_data(program_id, core_id, freq)
                    else:
                        energy_consumption, processing_time = read_data_from_dataset(program_id, core_id, freq)                    
                    '''
                    energy_consumption, processing_time = read_data(program_id, core_id, freq)
                    eer = calculate_eer(program_id, core_id, freq)
                else:
                    eer = -1
                    processing_time =  0
                    energy_consumption = 0
                    #instruction_count = 0
                eer_results[core][freq].append(eer)
                processing_time_results[core][freq].append(processing_time)
                power_consumption_results[core][freq].append(energy_consumption)
                #instruction_results[core][freq].append(instruction_count)

    remaining_program_sequence = program_sequence.copy()
    # working_cores = []
    # core_ids = {"big": 0, "middle": 0, "small": 0}

    total_processing_time = {f"{core_type}-{core_id}": 0.0 for core_type in core_types for core_id in
                               range(core_counts[core_type])}
    #print(f"total_processing_time = {total_processing_time}")
    total_power_consumption = {f"{core_type}-{core_id}": 0.0 for core_type in core_types for core_id in
                               range(core_counts[core_type])}
    total_instructions = {f"{core_type}-{core_id}": 0 for core_type in core_types for core_id in
                          range(core_counts[core_type])}
    total_cycles = {f"{core_type}-{core_id}": 0 for core_type in core_types for core_id in range(core_counts[core_type])}

    idle_cores_queue = Queue()
    for core_type, core_count in core_counts.items():
        for i in range(core_count):
            idle_cores_queue.put((core_type, i))

    #print(f"eer={eer_results}")
    # first time assign
    assigned_cores = []
    while (not idle_cores_queue.empty()) and sum(remaining_program_sequence) > 0:
        core_type, core_id = idle_cores_queue.get()
        program, freq, _ = assign_core(remaining_program_sequence, core_type=core_type, use_eer=use_eer, eer_results=eer_results)
        if program != -1:
            assigned_cores.append((core_type, core_id, program, freq, processing_time_results[core_type][freq][program]))
        else:
            idle_cores_queue.put((core_type, core_id))

    # following execution and assignment
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(execute_program, *core, total_processing_time, total_power_consumption, total_instructions, total_cycles)
            for core in assigned_cores
        ]
        assigned_cores.clear()

        while futures:
            done_future = None
            for f in futures:
                if f.done():
                    done_future = f
                    break

            if done_future:
                futures.remove(done_future)
                core_type, core_id = done_future.result()

                program, freq, _ = assign_core(remaining_program_sequence, core_type, use_eer=use_eer, eer_results=eer_results)
                if program != -1:
                    new_future = executor.submit(execute_program, core_type, core_id, program, freq,
                                                 processing_time_results[core_type][freq][program],
                                                 total_processing_time, total_power_consumption, total_instructions,
                                                 total_cycles)
                    futures.append(new_future)

                #print(f"program={program} core_type={core_type}, core_id={core_id}")

    '''
    total_cpi = {f"{core_type}-{core_id}": 0.0 for core_type in core_types for core_id in range(core_counts[core_type])}
    for core in total_processing_time.keys():
        cpi = total_cycles[core] / total_instructions[core] if total_instructions[core] != 0 else 0
        total_cpi[f"{core}"] = cpi
    '''
    
    #total_processing_time_sorted = sorted(total_processing_time.values())
    #min_id = np.nonzero(total_processing_time_sorted)[0][0]
    #sum_processing_time = total_processing_time_sorted[min_id]
    sum_processing_time = max(total_processing_time.values())
    sum_power_consumption = sum(total_power_consumption.values())

    #end_time = time.time()
    #total_runtime = end_time - start_time
    #print(f"\ntotal_runtime: {total_runtime:.2f} sec")

    return sum_processing_time, sum_power_consumption


if __name__ == "__main__":
    program_queue = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    core_types = ["big", "middle", "small"]
    core_counts = {"big": 3, "middle": 2, "small": 1}
    if test_mode:
        core_vars = None
    else:
        core_vars = [var_ROCKET[:-1], var_BOOMv2[:-1], var_YANQIHU[:-1]]

    print("\nNo.1:Running EER-based scheduling strategy:")
    eer_core_time, eer_core_power = schedule(program_queue, core_types, core_counts, core_vars=core_vars, use_eer=True)
    print(f"eer_core_time={eer_core_time}, eer_core_power={eer_core_power}")

    if 0:
        print("\nprocessing_time:")
        for core, processing_time in eer_core_time.items():
            print(f"{core}: {processing_time:.2f}")

        print("\npower_consumption:")
        for core, power_consumption in eer_core_power.items():
            print(f"{core}: {power_consumption:.2f}")

        print("\nCPI:")
        for core, cpi_value in eer_core_cpi.items():
            print(f"{core}: {cpi_value:.2f}")

        eer_total_power = sum(eer_core_power.values())
        print("\ntotal_power:", eer_total_power)

    print("\nNo.2:Running basic scheduling strategy:")
    basic_core_time, basic_core_power = schedule(program_queue, core_types, core_counts, core_vars=core_vars, use_eer=False)
    print(f"eer_core_time={basic_core_time}, eer_core_power={basic_core_power}")

    if 0:
        print("\nprocessing_time:")
        for core, processing_time in basic_core_time.items():
            print(f"{core}: {processing_time:.2f}")

        print("\npower_consumption:")
        for core, power_consumption in basic_core_power.items():
            print(f"{core}: {power_consumption:.2f}")
        
        print("\nCPI:")
        for core, cpi_value in basic_core_cpi.items():
            print(f"{core}: {cpi_value:.2f}")

        basic_total_power = sum(basic_core_power.values())
        print("\ntotal_power:", basic_total_power)
