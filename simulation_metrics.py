import copy
import os
import numpy as np

from config import *
from specific_design_microarchiteture import *

max_cpi = 99.99
max_power = 99.99


def CPI_metric(x: np.ndarray, case_name):
    version_to_match = var_to_version(x)
    metric = 99.99  # max
    for case in metrics_cases[case_name]:
        if case['version'] == version_to_match:
            metric = case['CPI']
            break
    # print(f"call CPIMetric x: {x}, y: {metric}, version: {version_to_match}")
    if 99.98 < metric:
        print(f"missing: call CPIMetric x: {x}, version: {version_to_match}")
        exit(1)
    elif metric < 0.2:
        print(f"missing: call CPIMetric x: {x}, version: {version_to_match}")
        exit(1)
    return metric


def delay_metric(x: np.ndarray, case_name):
    CPI = CPI_metric(x, case_name)
    if 147456 == N_SPACE_SIZE:
        freq_id = 10
    else:
        freq_id = 0
    delay = CPI * 10000000 / SIMULATOR_CYCLES_PER_SECOND_map[str(x[freq_id])] # second
    return delay


def power_metric(x: np.ndarray, case_name):
    version_to_match = var_to_version(x)
    metric = 99.99  # max
    for case in metrics_cases[case_name]:
        if case['version'] == version_to_match:
            metric = case['Power']
            break
    # print(f"call PowerMetric x: {x}, y: {metric}, version: {version_to_match}")
    if 99.98 < metric:
        print(f"missing: call PowerMetric x: {x}, version: {version_to_match}")
        exit(1)
    elif metric < 0.2:
        print(f"missing: call PowerMetric x: {x}, version: {version_to_match} case_name={case_name}")
        exit(1)
    return metric


def energy_metric(x: np.ndarray, case_name):
    power = power_metric(x, case_name) # W = J/second
    delay = delay_metric(x, case_name) # second
    energy = power * delay # J
    return energy


def area_metric(x: np.ndarray):
    version_to_match = var_to_version(x)
    metric = area_all[version_to_match]
    if 99.98 < metric:
        print(f"missing: call area_metric x: {x}, version: {version_to_match}")
        exit(1)
    elif metric < 0.2:
        print(f"missing: call area_metric x: {x}, version: {version_to_match}")
        exit(1)
    return metric


def metric_1(x: np.ndarray, case_name):
    if MULTIOBJ_DELAYXENERGY == multiobj_mode:
        return delay_metric(x, case_name)
    else:
        return CPI_metric(x, case_name)


def CPI_to_metric_1(CPI, x):
    if 147456 == N_SPACE_SIZE:
        freq_id = 10
    else:
        freq_id = 0
    delay = CPI * 10000000 / SIMULATOR_CYCLES_PER_SECOND_map[str(x[freq_id])] # second
    return delay


def metric_2(x: np.ndarray, case_name):
    if MULTIOBJ_DELAYXENERGY == multiobj_mode:
        return energy_metric(x, case_name)
    else:
        return power_metric(x, case_name)


def metric_1_2(x: np.ndarray, case_name):
    if MULTIOBJ_DELAYXENERGY == multiobj_mode:
        CPI = CPI_metric(x, case_name)
        if 147456 == N_SPACE_SIZE:
            freq_id = 10
        else:
            freq_id = 0
        delay = CPI * 10000000 / SIMULATOR_CYCLES_PER_SECOND_map[str(x[freq_id])]  # second

        power = power_metric(x, case_name)  # W = J/second
        energy = power * delay  # J
        #print(f"{case_name} {x} {CPI} {power}")
        return delay, energy
    else:
        return CPI_metric(x, case_name), power_metric(x, case_name)


def metric_3(x: np.ndarray):
    return area_metric(x)


def get_dataset(case_name, multiobj_mode):
    raw_data_file = open(data_set_file + "/" + case_name + '.txt', 'r')
    version_list = []
    y = []
    for each_line in raw_data_file:
        each_line_str = each_line.split(' ')
        version = each_line_str[0]
        if 30 < len(version):
            cpi = float(each_line_str[1])
            power = float(each_line_str[3])
            version_list.append(version)
            if MULTIOBJ_DELAYXENERGY == multiobj_mode:
                metric_1 = cpi * 10000000 / SIMULATOR_CYCLES_PER_SECOND_map[version[version_map_id['Frequency'][0]]] # second
                metric_2 = power * metric_1 # energy in J
            else:
                metric_1 = cpi
                metric_2 = power
            y.append([metric_1, metric_2])
    raw_data_file.close()
    return version_list, y


def get_ref_point():
    return [max_cpi + 1, max_power + 1]


def get_max_hv():
    return (max_cpi + 1) * (max_power + 1)


SIMULATOR_CYCLES_PER_SECOND_map = {
    '0': 1000000000,
    '1': 1500000000,
    '2': 2000000000,  # 2GHz
    '3': 3000000000,
}

IFQ_SIZE_map = {
    '0': 8,
    '1': 16,
}

DECODEQ_SIZE_map = {
    '0': 8,
    '1': 16,
}
FETCH_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 8,
    '3': 16,
}
DECODE_WIDTH_map = {
    '0': 2,
    '1': 3,
    '2': 4,
    '3': 5,
}

DISPATCH_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 5,
    '3': 6,
}

COMMIT_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 6,
    '3': 8,
}

PHY_GPR_NUM_map = {
    '0': 40,
    '1': 64,
    '2': 128,
    '3': 180,
}

PHY_FGPR_NUM_map = {
    '0': 40,
    '1': 64,
    '2': 128,
    '3': 180,
}

GPR_WRITEBACK_WIDTH_map = {
    '0': 2,
    '1': 4,
}

FGPR_WRITEBACK_WIDTH_map = {
    '0': 2,
    '1': 4,
}

RUU_SIZE_MAX_map = {
    '0': 32,
    '1': 64,
    '2': 128,
    '3': 256,
}

INT_BP_map = {
    '0': 1,
    '1': 2,
}

INT_ALU_map = {
    '0': 1,
    '1': 2,
}

INT_MULT_map = {
    '0': 1,
    '1': 2,
}

INT_MULT_OP_LAT_map = {
    '0': 2,
    '1': 4
}

INT_MULT_ISSUE_LAT_map = {
    '0': 4,
    '1': 1,
}

INT_DIV_OP_LAT_map = {
    '0': 8,
    '1': 16,
}

INT_DIV_ISSUE_LAT_map = {
    '0': 16,
    '1': 1,
}

FP_ALU_map = {
    '0': 1,
    '1': 2,
}

FP_ALU_MULT_map = {
    '0': 1,
    '1': 2,
}

FP_MULT_DIV_map = {
    '0': 1,
    '1': 2,
}

FP_ALU_MULT_DIV_map = {
    '0': 0,
    '1': 1,
}

FP_MULT_OP_LAT_map = {
    '0': 2,
    '1': 4,
}

FP_MULT_ISSUE_LAT_map = {
    '0': 4,
    '1': 1,
}

FP_DIV_OP_LAT_map = {
    '0': 8,
    '1': 16,
}

FP_DIV_ISSUE_LAT_map = {
    '0': 16,
    '1': 1,
}
'''
FP_SQRT_OP_LAT_map = {
'0' : 4,
'1' : 1,
}

FP_SQRT_ISSUE_LAT_map = {
'0' : 4,
'1' : 1,
}
'''

LOAD_PORT_WIDTH_map = {
    '0': 1,
    '1': 2,
}

STORE_PORT_WIDTH_map = {
    '0': 1,
    '1': 2,
}

LOAD_STORE_PORT_WIDTH_map = {
    '0': 0,
    '1': 2,
}

LOAD_QUEUE_SIZE_map = {
    '0': 10,
    '1': 30,
    '2': 60,
    '3': 90,
}

STORE_QUEUE_SIZE_map = {
    '0': 10,
    '1': 30,
    '2': 60,
    '3': 90,
}

BPRED_map = {
    '0': 'gShare',
    '1': 'tage'
}

RAS_SIZE_map = {
    '0': 8,
    '1': 16,
}

L1_ICACHE_SET_map = {
    '0': 64,
    '1': 128,
    '2': 256,
}

L1_ICACHE_ASSOC_map = {
    '0': 2,
    '1': 4,
    '2': 8,
}

L1_DCACHE_SET_map = {
    '0': 64,
    '1': 128,
    '2': 256,
}

L1_DCACHE_ASSOC_map = {
    '0': 2,
    '1': 4,
    '2': 8,
}

L1_DCACHE_WRITEBACK_map = {
    '0': 0,
    '1': 1
}

L2_CACHE_SET_map = {
    '0': 128,
    '1': 1024,
}

L2_CACHE_ASSOC_map = {
    '0': 4,
    '1': 8,
}

LLC_map = {
    '0': 2,
    # '1' : 3,
}

version_map_id = {
    'IFQ_SIZE': [0, len(IFQ_SIZE_map)],
    'DECODEQ_SIZE': [1, len(DECODEQ_SIZE_map)],
    'FETCH_WIDTH': [2, len(FETCH_WIDTH_map)],
    'DECODE_WIDTH': [3, len(DECODE_WIDTH_map)],
    'DISPATCH_WIDTH': [4, len(DISPATCH_WIDTH_map)],
    'COMMIT_WIDTH': [5, len(COMMIT_WIDTH_map)],
    'PHY_GPR_NUM': [6, len(PHY_GPR_NUM_map)],
    'PHY_FGPR_NUM': [7, len(PHY_FGPR_NUM_map)],
    'GPR_WRITEBACK_WIDTH': [8, len(GPR_WRITEBACK_WIDTH_map)],
    'FGPR_WRITEBACK_WIDTH': [9, len(FGPR_WRITEBACK_WIDTH_map)],
    'RUU_SIZE_MAX': [10, len(RUU_SIZE_MAX_map)],
    'INT_BP': [11, len(INT_BP_map)],
    'INT_ALU': [12, len(INT_ALU_map)],
    'INT_MULT': [13, len(INT_MULT_map)],
    'INT_MULT_OP_LAT': [14, len(INT_MULT_OP_LAT_map)],
    'INT_MULT_ISSUE_LAT': [15, len(INT_MULT_ISSUE_LAT_map)],
    'INT_DIV_OP_LAT': [16, len(INT_DIV_OP_LAT_map)],
    'INT_DIV_ISSUE_LAT': [17, len(INT_DIV_ISSUE_LAT_map)],
    'FP_ALU': [18, len(FP_ALU_map)],
    'FP_ALU_MULT': [19, len(FP_ALU_MULT_map)],
    'FP_MULT_DIV': [20, len(FP_MULT_DIV_map)],
    'FP_ALU_MULT_DIV': [21, len(FP_ALU_MULT_DIV_map)],
    'FP_MULT_OP_LAT': [22, len(FP_MULT_OP_LAT_map)],
    'FP_MULT_ISSUE_LAT': [23, len(FP_MULT_ISSUE_LAT_map)],
    'FP_DIV_OP_LAT': [24, len(FP_DIV_OP_LAT_map)],
    'FP_DIV_ISSUE_LAT': [25, len(FP_DIV_ISSUE_LAT_map)],
    # 'FP_SQRT_OP_LAT': 25,
    # 'FP_SQRT_ISSUE_LAT': 26,
    'LOAD_PORT_WIDTH': [26, len(LOAD_PORT_WIDTH_map)],
    'STORE_PORT_WIDTH': [27, len(STORE_PORT_WIDTH_map)],
    'LOAD_STORE_PORT_WIDTH': [28, len(LOAD_STORE_PORT_WIDTH_map)],
    'LOAD_QUEUE_SIZE': [29, len(LOAD_QUEUE_SIZE_map)],
    'STORE_QUEUE_SIZE': [30, len(STORE_QUEUE_SIZE_map)],
    'BPRED': [31, len(BPRED_map)],
    'RAS_SIZE': [32, len(RAS_SIZE_map)],
    'L1_ICACHE_SET': [33, len(L1_ICACHE_SET_map)],
    'L1_ICACHE_ASSOC': [34, len(L1_ICACHE_ASSOC_map)],
    'L1_DCACHE_SET': [35, len(L1_DCACHE_SET_map)],
    'L1_DCACHE_ASSOC': [36, len(L1_DCACHE_ASSOC_map)],
    'L1_DCACHE_WRITEBACK': [37, len(L1_DCACHE_WRITEBACK_map)],
    'L2_CACHE_SET': [38, len(L2_CACHE_SET_map)],
    'L2_CACHE_ASSOC': [39, len(L2_CACHE_ASSOC_map)],
    'LLC': [40, len(LLC_map)],
}

if 147456 == N_SPACE_SIZE:
    version_map_id['Frequency'] = [41, len(SIMULATOR_CYCLES_PER_SECOND_map)]
    version_map_id['max'] = 42
else:
    version_map_id['max'] = 41
    version_map_id['Frequency'] = version_map_id['DISPATCH_WIDTH']

DEF_FREQ = 0
DEF_IFQ = 1
DEF_DECODEQ = 2
DEF_FETCH_WIDTH = 3
DEF_DECODE_WIDTH = 4
DEF_DISPATCH_WIDTH = 5
DEF_COMMIT_WIDHT = 6
DEF_GPR = 7
DEF_FGPR = 8
DEF_GPR_WRITEBACK = 9
DEF_FGPR_WRITEBACK = 10
DEF_RUU_SIZE_MAX = 11
DEF_INT_BP = 12
DEF_FP_ALU = 19
DEF_LOAD_PORT_WIDTH = 29

DEF_MULTI_BTB = 35
DEF_BPRED = 36
DEF_L0_ICACHE = 38
DEF_EXECUTE_RECOVER = 39
DEF_RAW_LOAD_PRED = 40
DEF_PREFETCH_INST = 41
DEF_PREFETCH_DATA = 42
DEF_L1_ICACHE_SET = 43
DEF_L1_DCACHE_SET = 45
DEF_L2_CACHE_SET = 48


# 49

def gen_version_choose(DISPATCH_WIDTH_index, exe_int, exe_fp, lsq, dcache, icache, bp, l2cache, DECODE_WIDTH_index=-1,
                       RUU_SIZE_index=-1, freq_index=-1):
    version = ['0' for i in range(int(version_map_id['max']))]
    # DISPATCH_WIDTH = DISPATCH_WIDTH_map[version[version_map_id['DISPATCH_WIDTH'][0]]]
    version[version_map_id['DISPATCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['IFQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['FETCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    if -1 < DECODE_WIDTH_index:
        version[version_map_id['DECODE_WIDTH'][0]] = str(DECODE_WIDTH_index)
    else:
        version[version_map_id['DECODE_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['DECODEQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['COMMIT_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['PHY_GPR_NUM'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['PHY_FGPR_NUM'][0]] = str(DISPATCH_WIDTH_index)

    version[version_map_id['GPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['FGPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    if -1 < RUU_SIZE_index:
        version[version_map_id['RUU_SIZE_MAX'][0]] = str(RUU_SIZE_index)
    else:
        version[version_map_id['RUU_SIZE_MAX'][0]] = str(DISPATCH_WIDTH_index)

    if -1 < exe_int:
        version[version_map_id['INT_BP'][0]] = str(exe_int)
        version[version_map_id['INT_ALU'][0]] = str(exe_int)
        version[version_map_id['INT_MULT'][0]] = str(exe_int)
        version[version_map_id['INT_MULT_OP_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_DIV_OP_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(exe_int)
    else:
        version[version_map_id['INT_BP'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_ALU'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < exe_fp:
        version[version_map_id['FP_ALU'][0]] = str(exe_fp)
        version[version_map_id['FP_ALU_MULT'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_DIV'][0]] = str(exe_fp)
        version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_OP_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_DIV_OP_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(exe_fp)
    else:
        version[version_map_id['FP_ALU'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_ALU_MULT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < lsq:
        version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(lsq)
        version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(lsq)
    else:
        version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)
        version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)

    if -1 < bp:
        version[version_map_id['BPRED'][0]] = str(bp)
        version[version_map_id['RAS_SIZE'][0]] = str(bp)
    else:
        version[version_map_id['BPRED'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['RAS_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < icache:
        version[version_map_id['L1_ICACHE_SET'][0]] = str(icache)
        version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(icache)
    else:
        version[version_map_id['L1_ICACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))

    if -1 < dcache:
        version[version_map_id['L1_DCACHE_SET'][0]] = str(dcache)
        version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(dcache)
        version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int(dcache / 2))
    else:
        version[version_map_id['L1_DCACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int((DISPATCH_WIDTH_index) / 2))

    if -1 < l2cache:
        version[version_map_id['L2_CACHE_SET'][0]] = str(l2cache)
        version[version_map_id['L2_CACHE_ASSOC'][0]] = str(l2cache)
    else:
        version[version_map_id['L2_CACHE_SET'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['L2_CACHE_ASSOC'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    version[version_map_id['LLC'][0]] = '0'

    if -1 < freq_index:
        version[version_map_id['Frequency'][0]] = str(freq_index)
    else:
        version[version_map_id['Frequency'][0]] = str(DISPATCH_WIDTH_index)

    version_str = ''
    for version_iter in version:
        version_str += version_iter
    return version_str


var_list = [
    "DispatchWidth"
    , "ExeInt"
    , "ExeFP"
    , "LSQ"
    , "Dcache"
    , "Icache"
    , "BP"
    , "L2cache"
]
if 9216 == N_SPACE_SIZE:
    var_list.append("DecodeWidth")
elif 36864 == N_SPACE_SIZE:
    var_list.append("DecodeWidth")
    var_list.append("RUU_SIZE")
elif 147456 == N_SPACE_SIZE:
    var_list.append("DecodeWidth")
    var_list.append("RUU_SIZE")
    var_list.append("Frequency")

var_list_index = [
    version_map_id['DISPATCH_WIDTH'][0],
    version_map_id['INT_ALU'][0],
    version_map_id['FP_ALU'][0],
    version_map_id['LOAD_QUEUE_SIZE'][0],
    version_map_id['L1_DCACHE_SET'][0],
    version_map_id['L1_ICACHE_SET'][0],
    version_map_id['BPRED'][0],
    version_map_id['L2_CACHE_SET'][0],
]
if 9216 == N_SPACE_SIZE:
    var_list_index.append(version_map_id['DECODE_WIDTH'][0])
elif 36864 == N_SPACE_SIZE:
    var_list_index.append(version_map_id['DECODE_WIDTH'][0])
    var_list_index.append(version_map_id['RUU_SIZE_MAX'][0])
elif 147456 == N_SPACE_SIZE:
    var_list_index.append(version_map_id['DECODE_WIDTH'][0])
    var_list_index.append(version_map_id['RUU_SIZE_MAX'][0])
    var_list_index.append(version_map_id['Frequency'][0])


range_list = [
    len(DISPATCH_WIDTH_map)
    , len(INT_ALU_map)
    , len(FP_ALU_map)
    , len(LOAD_QUEUE_SIZE_map)
    , len(L1_DCACHE_SET_map)
    , len(L1_ICACHE_SET_map)
    , len(BPRED_map)
    , len(L2_CACHE_SET_map)
]
if 9216 == N_SPACE_SIZE:
    range_list.append(len(DECODE_WIDTH_map))
elif 36864 == N_SPACE_SIZE:
    range_list.append(len(DECODE_WIDTH_map))
    range_list.append(len(RUU_SIZE_MAX_map))
elif 147456 == N_SPACE_SIZE:
    range_list.append(len(DECODE_WIDTH_map))
    range_list.append(len(RUU_SIZE_MAX_map))
    range_list.append(len(SIMULATOR_CYCLES_PER_SECOND_map))    


def get_orthogonal_array():
    parameters = []
    for index in range_list:
        parameters.append([i for i in range(index)])

    from allpairspy import AllPairs
    samples = []
    for i, pairs in enumerate(AllPairs(parameters)):
        #print("orthogonal_array case id {:2d}: {}".format(i, pairs))
        samples.append(pairs)
    return samples


def get_orthogonal_array_v2():
    from OAT import OAT
    from collections import OrderedDict
    oat = OAT()
    parameters = []
    for param_name, index in zip(var_list, range_list):
        parameters.append((param_name, [i for i in range(index)]))
    case1 = OrderedDict(parameters)
    oa = oat.genSets(case1, mode=1, num=0)
    for i, pairs in enumerate(oa):
        print("orthogonal_array_v2 OAT case id {:2d}: {}".format(i, pairs))
    return oa


def get_sensitivity_data(case_name):
    # import os
    # root = os.getcwd()
    # print(f"get_sensitivity_data current path={root}")
    filename = './cpu_model_data_boom/' + case_name + '.txt'
    sensitivity = []
    try:
        file = open(filename, "r")
        for each_line_str in file:
            each_line_str_arr = each_line_str.split(' ')
            each_line_array = [str_try for str_try in each_line_str_arr if str_try != '']
            cpi = float(each_line_array[1])
            sensitivity.append(cpi)
        file.close()
    except:
        print(f"get_sensitivity_data: filename={filename} reading failed")
        exit(1)
    return sensitivity


def get_feature_weight(case_name):
    sensitivity = get_sensitivity_data(case_name=case_name)
    # sensitivity = [2, 1]
    feature_weight = [sensitivity[0] / sensitivity[7],  # DISPATCH_WIDTH_map
                      sensitivity[0] / sensitivity[6],  # INT_ALU_map
                      sensitivity[0] / sensitivity[6],  # FP_ALU_map
                      sensitivity[0] / sensitivity[5],  # LOAD_QUEUE_SIZE_map
                      sensitivity[0] / sensitivity[5],  # L1_DCACHE_SET_map
                      sensitivity[0] / sensitivity[4],  # L1_ICACHE_SET_map
                      sensitivity[0] / sensitivity[2],  # BPRED_map
                      sensitivity[0] / sensitivity[5],  # L2_CACHE_SET_map
                      ]
    if 9216 <= N_SPACE_SIZE:
        feature_weight.append(sensitivity[0] / sensitivity[1])  # DECODE_WIDTH_map
    if 36864 <= N_SPACE_SIZE:
        feature_weight.append(sensitivity[0] / sensitivity[7])  # RUU_SIZE_MAX_map

    feature_weight = np.asarray(feature_weight) / np.asarray(range_list)

    return feature_weight


def get_cluster_matrix(case_name):
    feature_weight = get_feature_weight(case_name=case_name)
    X = get_all_design_point()
    distance_matrix = np.zeros((len(X), len(X)))
    for X_id, x in enumerate(X):
        for X_id2, x_2 in enumerate(X):
            if X_id == X_id2:
                distance_matrix[X_id][X_id2] = 0
            else:
                distance_matrix[X_id][X_id2] = - np.sum(np.square(x - x_2) * feature_weight)
                # print(f"distance_matrix={distance_matrix}")
    return distance_matrix, feature_weight

'''
def get_init_by_kmeans_sklearn(n_clusters=0, random_state=0):
    samples = []
    from sklearn.cluster import k_means
    kmeans = k_means(X=X, n_clusters=n_clusters, random_state=random_state)
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, sample_weight=sample_weight).fit(X)
    for each_center in kmeans.cluster_centers_:
        x_distance = []
        for x in X:
            x_distance.append(np.sum(np.square(x - each_center)))
        x_selected = np.argmin(x_distance)
        samples.append(X[x_selected].tolist())

    if 0:
        colors = ['red', 'green', 'blue']
        import matplotlib.pyplot as plt
        for i, cluster in enumerate(kmeans.labels_):
            plt.scatter(X[i][0], X[i][1], color=colors[cluster])
    # print(f"samples={samples}")
    return samples
'''

def get_init_by_kmeans(n_clusters=0, random_state=0):
    if False:
        samples = get_kmeans_samples_from_file()
        return samples, -1, -1, -1

    import time
    from datetime import datetime
    startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
    # feature_weight = get_feature_weight(case_name=case_name)
    feature_weight_map = get_anova_from_file()
    feature_weight = np.asarray(feature_weight_map['CPI'][case_name][:len(var_list)])
    feature_weight = np.log(feature_weight)
    feature_weight[feature_weight < 0] = 0 #1e-1
    # distance_matrix, feature_weight = get_cluster_matrix(case_name=case_name)
    print(f"feature_weight={feature_weight} len={len(feature_weight)}")

    from pyclustering.cluster.kmeans import kmeans
    from pyclustering.utils.metric import type_metric, distance_metric

    user_function = lambda point1, point2: np.sum(np.square(point1 - point2) * feature_weight)
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    samples = []
    X = get_all_design_point()
    # create K-Means algorithm with specific distance metric
    X_index = np.arange(0, len(X))
    import random
    random.shuffle(X_index)
    start_centers = np.asarray(X)[X_index[:n_clusters]]
    kmeans_instance = kmeans(X, start_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    # clusters = kmeans_instance.get_clusters()
    clusters = kmeans_instance.get_centers()

    for each_center in clusters:
        x_distance = []
        for x_id in range(len(X)):
            x_distance.append([x_id, np.sum(np.square(X[x_id] - each_center) * feature_weight)])
        x_distance_sorted = np.asarray(sorted(x_distance, key=lambda obj_values: obj_values[1]))
        #print(f"x_distance_sorted={x_distance_sorted}")
        equal_flag = (np.abs(x_distance_sorted[:,1] - x_distance_sorted[0,1])) < 1e-4
        equal_index = np.argmin(equal_flag)
        #print(f"equal_flag={equal_flag}, equal_index={equal_index}")
        x_selected = int(x_distance_sorted[int(random.randint(0,equal_index))][0])
        #print(f"x_distance_sorted={x_distance_sorted[:10]}")
        #x_selected = np.argmin(x_distance)
        samples.append(X[x_selected].tolist())

    time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - startTime

    ss = -1
    # labels = kmeans_instance.predict(X)
    # from metric_function import evaluate_cluster
    # ss = evaluate_cluster(X=-distance_matrix, labels=labels)
    return samples, time_used, ss, feature_weight


def get_init_by_distance_cluster(n_clusters=0, random_state=0):
    samples = []
    distance_matrix, feature_weight = get_cluster_matrix(case_name=case_name)
    x_selecte_map = np.zeros(len(X))
    # from sklearn.cluster import AgglomerativeClustering
    # cluster_algo = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete') # average, ['ward', 'complete', 'average', 'single']
    # cluster_labels = cluster_algo.fit_predict(X=distance_matrix)
    from sklearn.cluster import AffinityPropagation
    cluster_algo = AffinityPropagation(affinity='precomputed', preference=-1).fit(distance_matrix)
    print(f"cluster_centers_indices_={len(cluster_algo.cluster_centers_indices_)}")
    for each_center in cluster_algo.cluster_centers_indices_:
        x_distance = []
        for x_id, x in enumerate(X):
            if x_selecte_map[x_id]:
                x_distance = 9999
            else:
                x_distance.append(np.sum(np.square(x - X[each_center]) * feature_weight))
        x_selected = np.argmin(x_distance)
        samples.append(X[x_selected].tolist())
        x_selecte_map[x_selected] = True
    if len(samples) < n_clusters:
        x_distance2 = []
        for x in X:
            x_distance_t = []
            for each_center in cluster_algo.cluster_centers_indices_:
                x_distance_t.append(np.sum(np.square(x - X[each_center]) * feature_weight))
            print(f"x_distance_t={x_distance_t}")
            x_distance2.append(min(x_distance_t))
        x_select_index = np.argsort(x_distance2)
        for each in x_select_index:
            if not x_selecte_map[x_select_index]:
                samples.append(X[x_select_index].tolist())
                if n_clusters <= len(samples):
                    break

    if 0:
        colors = ['red', 'green', 'blue']
        import matplotlib.pyplot as plt
        for i, cluster in enumerate(kmeans.labels_):
            plt.scatter(X[i][0], X[i][1], color=colors[cluster])
    # print(f"get_init_by_distance_cluster samples={samples}")
    print(f"get_init_by_distance_cluster len(samples)={len(samples)}")
    return samples


# for ax SearchSpace
def get_search_space():
    parameters_vec = []
    for index in range(len(var_list)):
        # parameters_vec.append(ChoiceParameter(name=var_list[index], parameter_type=ParameterType.STRING, values=[str(i) for i in range(range_list[index])]))
        parameters_vec.append(RangeParameter(name=var_list[index], lower=0, upper=range_list[index] - 1,
                                             parameter_type=ParameterType.INT))

    search_space = SearchSpace(
        parameters=parameters_vec,
    )
    return search_space


def get_var_list():
    return var_list
    # return ["x"+str(i) for i in range(1, 1 + length(var_list)]


def get_var_list_index():
    return var_list_index


def var_to_version(x):
    if 11 == len(x):
        version = gen_version_choose(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])
    elif 10 == len(x):
        version = gen_version_choose(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    elif 9 == len(x):
        version = gen_version_choose(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])
    else:
        version = gen_version_choose(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])

    return version


# Euclidean distance
def distance_f(real_pareto_optimal, learned_pareto_optimal):
    # print(f"real_pareto_optimal={real_pareto_optimal}")
    # print(f"learned_pareto_optimal={learned_pareto_optimal}")
    # distance = np.sqrt(np.sum(np.square(np.array(real_pareto_optimal) - np.array(learned_pareto_optimal))))
    pareto_diff = (np.array(real_pareto_optimal) - np.array(learned_pareto_optimal))
    # try:
    #    if 2 < len(pareto_diff):
    #        pareto_diff = pareto_diff / range_list
    # except:
    #    print(f"Error: pareto_diff={pareto_diff}, range_list={range_list}")
    distance = np.sqrt(np.sum(np.square(pareto_diff)))
    return distance


def distance_f2(real_pareto_optimal, learned_pareto_optimal):
    # print(f"real_pareto_optimal={real_pareto_optimal}")
    # print(f"learned_pareto_optimal={learned_pareto_optimal}")
    # distance = np.sqrt(np.sum(np.square(np.array(real_pareto_optimal) - np.array(learned_pareto_optimal))))
    pareto_diff = (np.array(real_pareto_optimal) / np.array(learned_pareto_optimal) - 1)
    # try:
    #    if 2 < len(pareto_diff):
    #        pareto_diff = pareto_diff / range_list
    # except:
    #    print(f"Error: pareto_diff={pareto_diff}, range_list={range_list}")
    distance = np.max(0, pareto_diff[0], pareto_diff[1])
    return distance


def evaluate_ADRS(real_pareto_optimal_sets, learned_pareto_optimal_sets, coverage=False):
    ADRS = 0
    coverage_num = 0
    # print("real=", real_pareto_optimal_sets)
    # print("leared=", learned_pareto_optimal_sets)
    for real_pareto_optimal in real_pareto_optimal_sets:
        distances = []
        for learned_pareto_optimal in learned_pareto_optimal_sets:
            distances.append(distance_f(real_pareto_optimal, learned_pareto_optimal))
        min_distance = min(distances)
        ADRS += min_distance
        if coverage:
            if min_distance < 0.001:
                coverage_num += 1
    ADRS /= len(real_pareto_optimal_sets)
    coverage_percent = coverage_num / len(real_pareto_optimal_sets)
    if coverage:
        return ADRS, coverage_percent
    else:
        return ADRS


def evaluate_ADRS_IGD(real_pareto_optimal_data, learned_pareto_optimal_data, coverage=False):
    real_pareto_optimal_y, real_pareto_optimal_y2, real_pareto_optimal_sets = real_pareto_optimal_data
    learned_pareto_optimal_sets, learned_pareto_optimal_frontier = learned_pareto_optimal_data
    # IGD (Inverted Generational Distance)
    ADRS = 0
    IGD = 0    
    coverage_num = 0
    # print("real=", real_pareto_optimal_sets)
    # print("leared=", learned_pareto_optimal_sets)
    for real_pareto_optimal in real_pareto_optimal_sets:
        distances = []
        for learned_pareto_optimal in learned_pareto_optimal_sets:
            distances.append(distance_f(real_pareto_optimal, learned_pareto_optimal))
        min_distance = min(distances)
        ADRS += min_distance
        if coverage:
            if min_distance < 0.001:
                coverage_num += 1
    for real_pareto_optimal_iter_y, real_pareto_optimal_iter_y2 in zip(real_pareto_optimal_y, real_pareto_optimal_y2):
        distances = []
        for learned_pareto_optimal in learned_pareto_optimal_frontier:
            distances.append(distance_f([real_pareto_optimal_iter_y, real_pareto_optimal_iter_y2], learned_pareto_optimal))
        min_distance = min(distances)
        IGD += min_distance**2

    ADRS /= len(real_pareto_optimal_sets)
    IGD /= len(learned_pareto_optimal_sets)
    IGD *= 1000000
    coverage_percent = coverage_num / len(real_pareto_optimal_sets)
    if coverage:
        #print(f"evaluate_ADRS_IGD IGD={IGD}")
        return ADRS, IGD, coverage_percent
    else:
        return ADRS, IGD

def get_di(learned_pareto_optimal_sets_y_unsort):
    di = []
    learned_pareto_optimal_sets_y_sort = copy.deepcopy(learned_pareto_optimal_sets_y_unsort)
    learned_pareto_optimal_sets_y_sort = np.asarray(sorted(learned_pareto_optimal_sets_y_sort, key=(lambda x: [x[0]])))
    learned_pareto_optimal_sets_y_sort_trans = np.zeros(np.shape(learned_pareto_optimal_sets_y_sort)[0:2])
    learned_pareto_optimal_sets_y_sort_trans[:, 0] = learned_pareto_optimal_sets_y_sort[:, 0] / max(
        learned_pareto_optimal_sets_y_sort[:, 0])
    learned_pareto_optimal_sets_y_sort_trans[:, 1] = learned_pareto_optimal_sets_y_sort[:, 1] / max(
        learned_pareto_optimal_sets_y_sort[:, 1])
    # learned_pareto_optimal_sets_y_sort_trans[:, 2] = learned_pareto_optimal_sets_y_sort[:, 2]
    for i in range(len(learned_pareto_optimal_sets_y_sort) - 1):
        di.append(distance_f(learned_pareto_optimal_sets_y_sort_trans[i, 0:2],
                             learned_pareto_optimal_sets_y_sort_trans[i + 1, 0:2]))
    return di, learned_pareto_optimal_sets_y_sort


def evaluate_non_uniformity(learned_pareto_optimal_sets_y_unsort):
    di, _ = get_di(learned_pareto_optimal_sets_y_unsort)
    non_uniformity = sum(np.abs(di - np.mean(di))) / (np.sqrt(2) * (len(learned_pareto_optimal_sets_y_unsort) - 1))
    return non_uniformity



def get_all_design_point():
    X = []
    import itertools
    if 2304 == N_SPACE_SIZE:
        var_list_range = [[] for i in range(8)]
    elif 9216 == N_SPACE_SIZE:
        var_list_range = [[] for i in range(9)]
    elif 36864 == N_SPACE_SIZE:
        var_list_range = [[] for i in range(10)]
    elif 147456 == N_SPACE_SIZE:
        var_list_range = [[] for i in range(11)]        
    for var_index, var_iter in enumerate(range_list):
        var_list_range[var_index] = [value for value in range(var_iter)]
        # print(f"var_list[{var_list[var_index]}] # = {var_iter}")
    if 147456 == N_SPACE_SIZE:
        design_space_product = itertools.product(var_list_range[0], var_list_range[1], var_list_range[2],
                                                 var_list_range[3], var_list_range[4], var_list_range[5],
                                                 var_list_range[6], var_list_range[7], var_list_range[8],
                                                 var_list_range[9], var_list_range[10])        
    elif 36864 == N_SPACE_SIZE:
        design_space_product = itertools.product(var_list_range[0], var_list_range[1], var_list_range[2],
                                                 var_list_range[3], var_list_range[4], var_list_range[5],
                                                 var_list_range[6], var_list_range[7], var_list_range[8],
                                                 var_list_range[9])
    elif 9216 == N_SPACE_SIZE:
        design_space_product = itertools.product(var_list_range[0], var_list_range[1], var_list_range[2],
                                                 var_list_range[3], var_list_range[4], var_list_range[5],
                                                 var_list_range[6], var_list_range[7], var_list_range[8])
    else:
        design_space_product = itertools.product(var_list_range[0], var_list_range[1], var_list_range[2],
                                                 var_list_range[3], var_list_range[4], var_list_range[5],
                                                 var_list_range[6], var_list_range[7])
    for each in design_space_product:
        X.append(list(each))
    # print(f"X_length={len(X)}")
    return X


def get_anova(case_name_config):
    all_design_point = get_all_design_point()
    metric = 'CPI'
    #metric = 'Power'
    if 'CPI' == metric:
        all_y = [CPI_metric(x) for x in all_design_point]
    else:
        all_y = [power_metric(x) for x in all_design_point]
    all_y_mean = np.mean(np.array(all_y))
    F = np.zeros(len(range_list))
    n = len(all_y)
    for factor_index in range(len(range_list)):
        k = range_list[factor_index]
        SST_k = np.zeros(k)
        SSE_k = np.zeros(k)
        y_k_mean = np.zeros(k)
        for x, y in zip(all_design_point, all_y):
            SST_k[x[factor_index]] += (y - all_y_mean) ** 2
            y_k_mean[x[factor_index]] += y
            # print(f"SST_k[x[factor_index]] = {SST_k[x[factor_index]]}")
        y_k_mean /= (n / k)
        for x, y in zip(all_design_point, all_y):
            SSE_k[x[factor_index]] += (y - y_k_mean[x[factor_index]]) ** 2
            # print(f"SSE_k[x[factor_index]] = {SSE_k[x[factor_index]]}")
        SST = sum(SST_k)
        SSE = sum(SSE_k)
        SSB = SST - SSE
        F[factor_index] = (SSB / (k - 1)) / (SSE / (n - k))
        print(f"SSB={SSB}, SSE={SSE}")
        import scipy.stats as ss
        pval = ss.f.cdf(F[factor_index], k - 1, n - k)
        print(
            f"metric={metric}, factor_index={factor_index}, var_list={var_list[factor_index]}, F={F[factor_index]}, pval={pval}")
    print(f"{case_name_config} {F}")
    file = open('anova.txt', 'a')
    file.write(f"{case_name_config}")
    for f in F:
        file.write(" %-10.2f" % (f))
    file.write(f" \n")
    file.close()
    return F


def get_anova_from_file():
    sensitivity_map = {'CPI': {}, 'Power': {}}
    file = open('anova.txt', 'r')
    for each_line_str in file:
        if '#' != each_line_str[0]:
            each_line_str_arr = each_line_str.split(' ')
            # print(f"each_line_str_arr={each_line_str_arr}")
            each_line_array = [float(str_try) for str_try in each_line_str_arr[2:-1] if str_try != '']
            sensitivity = each_line_array
            sensitivity_map[each_line_str_arr[0]][each_line_str_arr[1]] = sensitivity
    file.close()
    return sensitivity_map


def get_specific_micro_data():
    specific_micro_data = {}
    for each_case in case_names:
        specific_micro_data[each_case] = {'CPI': {}, 'Power': {}}
        file = open("data_all_simpoint_" + str(N_SPACE_SIZE) + "/cpu_model_data_boom/" + each_case + '.txt', 'r')
        for each_line_str in file:
            each_line_str_arr = each_line_str.split(' ')
            specific_micro_data[each_case]['CPI'][each_line_str_arr[0]] = float(each_line_str_arr[1])
            specific_micro_data[each_case]['Power'][each_line_str_arr[0]] = float(each_line_str_arr[3])
        file.close()
    return specific_micro_data


def get_kmeans_samples_from_file():
    samples = []
    file = open(str(N_SPACE_SIZE) + '_kmeans_log.txt', 'r')
    for each_line_str in file:
        each_line_str_arr = each_line_str.split(' ')
        # print(f"each_line_str_arr={each_line_str_arr}")
        each_line_array = [int(str_try) for str_try in each_line_str_arr if str_try != '' and str_try != '\n']
        sample = each_line_array
        samples.append(sample)
    file.close()
    # print(f"samples={samples}")
    return samples


def filter_data_records(case_name):
    config_file = open(data_set_file + '/' + case_name + '-filtered.txt', "w", encoding='utf-8')
    all_design_point = get_all_design_point()
    print(f"all_design_point = {len(all_design_point)} ")
    metrics_all = read_metrics(data_set_file + '/', case_name)
    found_count = 0
    for design_point in all_design_point:
        version_to_match = var_to_version(design_point)
        found = False
        for case in metrics_all:
            if case['version'] == version_to_match:
                config_file.write(version_to_match + ' ' + str(case['CPI']) + ' ' + str(case['bpred']) + ' ' + str(
                    case['Power']) + ' 0 \n')
                found = True
                found_count += 1
                break
        # if not found:
        # print(f"not found version_to_match={version_to_match}")
    config_file.close()
    print(f"found_count = {found_count}/{len(all_design_point)}")
    return


def merge_data_records(case_name):
    config_file1 = open('../ExplorDSE/data_all_simpoint_36864' + '/' + case_name + '.txt', "r", encoding='utf-8')
    config_file2 = open('../ExplorDSE/data_all_simpoint_110592' + '/' + case_name + '.txt', "r", encoding='utf-8')
    final_config_file = open('data_all_simpoint_147456' + '/' + case_name + '.txt', "w", encoding='utf-8')
    for design_point in config_file1:
        strs = design_point.split(' ')
        fix_version = strs[0]+strs[0][version_map_id['DISPATCH_WIDTH'][0]]
        final_config_file.write(fix_version+' ' + strs[1] + ' ' + strs[2] + ' ' + strs[3] + ' ' + strs[4] + ' \n')
    config_file1.close()
    for design_point in config_file2:
        final_config_file.write(design_point)
    config_file2.close()
    final_config_file.close()


def get_gmean_case():
    program_queue_name, program_queue, program_bitmap, program_queue_ids = get_workloads_all(random_seed=0)
    metric_cases_1 = np.zeros((len(program_queue), N_SPACE_SIZE))
    metric_cases_2 = np.zeros((len(program_queue), N_SPACE_SIZE))
    all_design_point = get_all_design_point()
    for case_name_id, case_name in enumerate(program_queue):
        print(f"dealing {case_name_id+1} / {len(program_queue)}")
        metric_cases_1[case_name_id] = [CPI_metric(x, case_name) for x in all_design_point]
        metric_cases_2[case_name_id] = [power_metric(x, case_name) for x in all_design_point]
    #gmean_case = np.mean(gmean_cases)
    from scipy.stats import gmean
    gmean_1 = gmean(metric_cases_1)
    gmean_2 = gmean(metric_cases_2)
    config_file = open('data_all_simpoint_' + str(N_SPACE_SIZE) + '/' + program_queue_name + '.txt', "w", encoding='utf-8')
    for x, metric_1, metric_2 in zip(all_design_point, gmean_1, gmean_2):
        config_file.write(str(var_to_version(x)) + ' ' + str(metric_1) + ' 0 ' + str(metric_2) + ' 0 \n')
    config_file.close()

    #from get_real_pareto_frontier import cal_pareto_optimality_to_file
    #print(f"cal_pareto_optimality_to_file {program_queue_name}")
    #cal_pareto_optimality_to_file(case_name=program_queue_name, multiobj_mode=multiobj_mode)


def read_metrics(config_dir, case_name):
    metrics = []

    metrics_list_max = 0
    for config_dir_name in os.listdir(config_dir):
        if case_name not in config_dir_name:
            continue
        print(f"open {config_dir + config_dir_name}")
        config_file = open(config_dir + config_dir_name, "r", encoding='utf-8')
        for each_line in config_file:
            metrics_list = None
            try:
                metrics_list = [i for i in each_line.split(" ")]
            except:
                print(config_dir_name + ' failed')

            if 0 == metrics_list_max:
                metrics_list_max = len(metrics_list)
            elif len(metrics_list) != metrics_list_max:
                print(config_dir_name + ' len != ' + str(metrics_list_max))
                # exit(1)
            cpi = float(metrics_list[1])
            power = float(metrics_list[3])
            if cpi < 0.2:
                print(f"version={metrics_list[0]} cpi={cpi} < 0.2")
                #exit(1)
            if power < 0.2:
                print(
                    f"version={metrics_list[0]} each_line={each_line}, metrics_list={metrics_list}, power={power} < 0.2")
                #exit(1)
            metrics += [
                {'name': config_dir_name.split('.')[0]
                    , 'version': metrics_list[0]
                    , 'CPI': cpi
                    , 'bpred': float(metrics_list[2])
                    , 'Power': power
                    , 'Area': 0.0}]
            global max_cpi, max_power
            max_cpi = max(max_cpi, float(metrics_list[1]))
            max_power = max(max_power, float(metrics_list[3]))
        config_file.close()
    return metrics


data_set_file = 'data_all_simpoint'
#if 2304 != N_SPACE_SIZE:
data_set_file += "_" + str(N_SPACE_SIZE)
#metrics_all = read_metrics(data_set_file + '/', case_name_config)
# print(f"metrics_all={metrics_all}")


def read_metrics_cases(config_dir=data_set_file):
    metrics_cases = {}
    #for case_name_id, case_name_txt in enumerate(os.listdir(config_dir)):
    #    case_name = case_name_txt[:-4]
    for case_name_id, case_name in enumerate(case_names):
        metrics = read_metrics(config_dir, case_name)
        metrics_cases[case_name] = metrics
    return metrics_cases


def read_area_file():
    area_all = {}
    area_range = [9999, 0]
    area_file = open('area_core_' + '147456' + '.txt', 'r')
    for each_area in area_file:
        each_area_strs = each_area.split(" ")
        version = each_area_strs[0][:-1]
        #if len('00131111003111111111111111000110011221000') == len(version):
        #    version += version[version_map_id['DISPATCH_WIDTH'][0]]
        #version_to_var
        area = float(each_area_strs[1])
        if (version in area_all) and (area != area_all[version]):
            print(f"error area for version={version} not match")
        else:
            area_all[version] = area
        area_range = [min(area_range[0], area), max(area_range[1], area)]
    area_file.close()
    return area_all, area_range


metrics_cases = read_metrics_cases(data_set_file + '/')
area_all, area_range = read_area_file()


AREA_ROCKET = area_all[var_to_version(var_ROCKET)[:-1]]
AREA_BOOMV2 = area_all[var_to_version(var_BOOMv2)[:-1]]
AREA_YANQIHU = area_all[var_to_version(var_YANQIHU)[:-1]]

AREA_CONSTRAIN_VERSION = 0

if 1 == AREA_CONSTRAIN_VERSION:
    AREA_CONSTRAIN = 5 * AREA_BOOMV2
elif 2 == AREA_CONSTRAIN_VERSION:
    AREA_CONSTRAIN = 6 * AREA_BOOMV2    
else:
    AREA_CONSTRAIN = 4 * AREA_BOOMV2
AREA_CONSTRAIN += 1e-5 # for floating-point little error


def hmp_is_under_constrain(core_space, hmp, core_area_space=None):
    if core_area_space is None:
        core_area_space = [area_all[var_to_version(core_i)[:-1]] for core_i in core_space]
    area_sum = np.sum(core_area_space * np.asarray(copy.deepcopy(hmp)))
    #print(f"area_sum={area_sum} AREA_CONSTRAIN={AREA_CONSTRAIN}")
    core_space_slice = np.asarray(hmp)[0 < np.asarray(hmp)]
    if (0 < area_sum) and (area_sum < AREA_CONSTRAIN) and (len(core_space_slice) <= MAX_CORE_TYPES):
        return True
    else:
        return False


if __name__ == '__main__':
    # filter_data_records(case_name='505.1-refrate-1')
    #merge_data_records(case_name=case_name_config)
    #F = get_anova(case_name_config)
    #get_orthogonal_array_v2()

    #get_gmean_case()

    print(f"area_range={area_range}")
    print(f"AREA_BOOMV2={AREA_BOOMV2}, AREA_YANQIHU={AREA_YANQIHU}, AREA_CONSTRAIN=4*AREA_BOOMV2={AREA_CONSTRAIN}")
