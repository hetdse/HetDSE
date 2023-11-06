# spec06-int * 2, fp *2
# case_name = "403.2-ref-1"
# case_name = "437.1-ref-10"
# case_name = "471.1-ref-1"
# case_name = "453.1-ref-16"

# spec17-int * 9
# case_name = "500.1-refrate-1"
# case_name = "502.1-refrate-1" ### not ready
# case_name = "505.1-refrate-1"
# case_name = "523.1-refrate-1"
# case_name = "525.1-refrate-1"
# case_name = "531.1-refrate-1"
# case_name = "541.1-refrate-1"
# case_name = "548.1-refrate-1"
# case_name = "557.1-refrate-1"
# spec17-fp * 13
# case_name = "503.1-refrate-1"
# case_name = "507.1-refrate-1"
# case_name = "508.1-refrate-1"
# case_name = "510.1-refrate-1"
# case_name = "511.1-refrate-1"
# case_name = "519.1-refrate-1"
# case_name = "521.1-refrate-1"
# case_name = "526.1-refrate-1"
# case_name = "527.1-refrate-1"
# case_name = "538.1-refrate-1"
# case_name = "544.1-refrate-1"
# case_name = "549.1-refrate-1"
# case_name = "554.1-refrate-1"
import copy
import random

import numpy as np

np.random.seed(0)

import socket
hostname = socket.getfqdn(socket.gethostname())
print(f"hostname={hostname}")
smoke_test = False
host_tag = ''
if 'BF-202106011024' in hostname \
    or 'DESKTOP-A4P9S3E' in hostname \
    or 'SC-202005121725' in hostname :
    host_tag = '-BF'
elif 'gtl' in hostname:
    host_tag = '-gtl'

case_names = [
    "500.1-refrate-1",
    "502.2-refrate-1",
    "503.1-refrate-1",
    "505.1-refrate-1",
    "507.1-refrate-1",  # not coverage, last_predict has high effect
    "508.1-refrate-1",
    "510.1-refrate-1",
    "511.1-refrate-1",
    "519.1-refrate-1",  # pareto is straight strange
    "520.1-refrate-1",
    "521.1-refrate-1",  # pareto is straight strange
    "523.1-refrate-1",  # pareto non-uniform interval is easy to stuck
    "525.1-refrate-1",
    "526.1-refrate-1",
    "527.1-refrate-1",
    "531.1-refrate-1",  # pareto is straight strange
    "538.1-refrate-1",
    "541.1-refrate-1",
    "544.1-refrate-1",
    "548.1-refrate-1",
    "549.1-refrate-1",
    "554.1-refrate-1",
    "557.1-refrate-1",
    #'gmean',
]

case_names2 = [
    "500.1-refrate-1",
    "502.2-refrate-1",
    "503.1-refrate-1",
    "505.1-refrate-1",
    "507.1-refrate-1",  # not coverage, last_predict has high effect
    "508.1-refrate-1",
    "510.1-refrate-1",
    "511.1-refrate-1",
    "519.1-refrate-1",  # pareto is straight strange
    "520.1-refrate-1",
    "521.1-refrate-1",  # pareto is straight strange
    #"523.1-refrate-1",  # pareto non-uniform interval is easy to stuck
    "525.1-refrate-1",
    "526.1-refrate-1",
    "527.1-refrate-1",
    "531.1-refrate-1",  # pareto is straight strange
    "538.1-refrate-1",
    "541.1-refrate-1",
    "544.1-refrate-1",
    "548.1-refrate-1",
    #"549.1-refrate-1",
    #"554.1-refrate-1",
    "557.1-refrate-1",
]

import sys
if 3 < len(sys.argv):
    N_DOMAIN = int(sys.argv[3])
else:
    N_DOMAIN = 1
    #N_DOMAIN = 10
    # case_names = ["523.1-refrate-1", '549.1-refrate-1']
    #N_DOMAIN = len(case_names)


if 'BF-202106011024' == hostname:
    #N_DOMAIN = 1 #len(case_names)
    if 1 == N_DOMAIN:
        case_names = ["549.1-refrate-1"]
if 'SC-202005121725' == hostname or (1 == N_DOMAIN):
     #len(case_names)
    if 1 == N_DOMAIN:
        case_names = ["549.1-refrate-1"]
if 'DESKTOP-A4P9S3E' == hostname:
    N_DOMAIN = 3 #len(case_names)
    if 1 == N_DOMAIN:
        case_names = ["549.1-refrate-1"]
'''
elif 'LGM-' in hostname:
    case_names = [
        "523.1-refrate-1",
        #"549.1-refrate-1",
        #"554.1-refrate-1",
    ]
    N_DOMAIN = len(case_names)
'''

if smoke_test:
    case_names = [
        "523.1-refrate-1",  # pareto non-uniform interval is easy to stuck
        #"549.1-refrate-1",
        #"554.1-refrate-1",
    ]
    N_DOMAIN = len(case_names)

case_names_str = [
    'perlbench',  # "500.1-refrate-1",
    'gcc',  # '# "502.2-refrate-1"
    'bwaves',  # "503.1-refrate-1",
    'mcf',  # "505.1-refrate-1",
    'cactuBSSN',  # "507.1-refrate-1",
    'namd',  # "508.1-refrate-1",
    'parest',  # '"510.1-refrate-1",
    'povray',  # "511.1-refrate-1",
    'lbm',  # "519.1-refrate-1",
    'omnetpp',  # "520.1-refrate-1",
    'wrf',  # "521.1-refrate-1",
    'xalancbmk',  # '"523.1-refrate-1",
    'x264',  # '"525.1-refrate-1",
    'blender',  # "526.1-refrate-1",
    'cam4',  # "527.1-refrate-1",
    'deepsjeng',  # '"531.1-refrate-1",
    'imagick',  # "538.1-refrate-1",
    'leela',  # '"541.1-refrate-1",
    'nab',  # "544.1-refrate-1",
    'exchange2',  # '"548.1-refrate-1",
    'fotonik3d',  # "549.1-refrate-1",
    'roms',  # "554.1-refrate-1",
    'xz',  # '"557.1-refrate-1",
]


def get_domain_id(case_name):
    for domain_id, case_name_iter in enumerate(case_names):
        if case_name_iter == case_name:
            return domain_id
    print(f"get_domain_id: no {case_name}")
    exit(1)
    return -1

'''
import sys
if 1 < len(sys.argv):
    if '502' == sys.argv[1]:
        case_name_config = sys.argv[1] + ".2-refrate-1"
    else:
        case_name_config = sys.argv[1] + ".1-refrate-1"
else:
    #case_name_config = "502.2-refrate-1"
    #case_name_config = "503.1-refrate-1"
    #case_name_config = "519.1-refrate-2"
    case_name_config = "557.1-refrate-1"
    #case_name_config = "gmean"
'''

mape_line_analysis = True
plot_pareto = False

'''
if 0:
    if 3 < len(sys.argv):
        exp_id = int(sys.argv[3])
        mape_line_analysis = True
    else:
        exp_id = None
else:
    if 3 < len(sys.argv):
        case_range = int(sys.argv[3])
    else:
        case_range = None
    exp_id = None
'''

def get_SRC_DOMAIN_ENCODE_LIST_STR(src_domain_list):
    SRC_DOMAIN_ENCODE_LIST_STR = ''
    for domain_iter, case_name_iter in enumerate(src_domain_list):
        domain_id = get_domain_id(case_name_iter)
        SRC_DOMAIN_ENCODE_LIST_STR += "%02d" % (domain_id)
    return SRC_DOMAIN_ENCODE_LIST_STR


def get_domain_encode_map(case_name, src_domain_list):
    domain_encode_map = np.ones(len(case_names), dtype=int) * len(case_names)  # init
    for domain_iter, case_name_iter in enumerate(src_domain_list):
        domain_id = get_domain_id(case_name_iter)
        #print(f"{case_name_iter} -> {domain_id}")
        domain_encode_map[domain_id] = domain_iter
    domain_encode_map[get_domain_id(case_name)] = N_SRC_DOMAIN
    return domain_encode_map


def get_domain_encode_list(src_domain_list):
    SRC_DOMAIN_ENCODE_LIST = []
    for case_name_iter in src_domain_list:
        domain_id = get_domain_id(case_name_iter)
        SRC_DOMAIN_ENCODE_LIST.append(domain_id)
    return SRC_DOMAIN_ENCODE_LIST


def shuffle(target):
    for change in range(len(target) - 1, 0, -1):
        lower = random.randint(0, change)
        target[lower], target[change] = target[change], target[lower]


def get_src_domain_list(case_name, random_seed):
    src_domain_list = copy.deepcopy(case_names)
    src_domain_list.remove(case_name)
    random.seed(random_seed)
    random.shuffle(src_domain_list)
    src_domain_list = src_domain_list[:N_SRC_DOMAIN]
    return src_domain_list


def get_workloads_all(random_seed=0, n_domain=0):
    if 0 == n_domain:
        n_domain = N_DOMAIN
    if -1 < random_seed:
      #all_list = copy.deepcopy(case_names)
      case_ids = np.arange(0, len(case_names))
      random.seed(random_seed)
      random.shuffle(case_ids)
      program_queue_ids = case_ids[:n_domain]
      #program_queue = [21, 22]
    else:
      program_queue_ids = [-1-random_seed]
    program_queue = np.asarray(case_names)[program_queue_ids]
    program_queue_name = 'set_n' + str(n_domain) + '_id' + str(random_seed)
    program_bitmap = [0 for _ in range(len(case_names))]
    for program_id in program_queue_ids:
        program_bitmap[program_id] = 1
    print(f"program_queue_name={program_queue_name}, program_queue={program_queue}")
    return program_queue_name, program_queue, program_bitmap, program_queue_ids
  

MULTIOBJ_CPIXPOWER = 0
MULTIOBJ_CPIXAREA = 1
MULTIOBJ_DELAYXENERGY = 2

multiobj_mode = MULTIOBJ_DELAYXENERGY
if MULTIOBJ_DELAYXENERGY == multiobj_mode:
    multiobj_mode_x_label = 'Delay'
    multiobj_mode_y_label = 'Energy'
else:
    multiobj_mode_x_label = 'CPI'
    multiobj_mode_y_label = 'Power'


#metric_name = "CPI"
#metric_name = "Power"

DOMAIN_ENCODE_ID = 0

#N_SPACE_SIZE = 2304  #  8 uarch parameters
#N_SPACE_SIZE = 9216  #  9 uarch parameters
#N_SPACE_SIZE = 36864 # 10 uarch parameters
N_SPACE_SIZE = 147456 # 11 uarch parameters

if smoke_test:
    N_SAMPLES_INIT = 2
elif 147456 == N_SPACE_SIZE:
    N_SAMPLES_INIT = 26
elif 36864 == N_SPACE_SIZE:
    N_SAMPLES_INIT = 34
elif 9216 == N_SPACE_SIZE:
    N_SAMPLES_INIT = 25 #28
else:
    N_SAMPLES_INIT = 24 #23

N_SAMPLES_ALL = N_SAMPLES_INIT+1 if smoke_test else 100

N_SAMPLES_INIT_HMP = 20
N_SAMPLES_ALL_HMP = 100
#N_SAMPLES_ALL_HMP = 575

#N_SAMPLES_INIT_HMP = 31771
#N_SAMPLES_ALL_HMP = N_SAMPLES_ALL_HMP

MAX_CORE_TYPES = 3

EVALUATION_INDEX = 2 # energy*delay
#EVALUATION_INDEX = 3 # delay*energy^2
#EVALUATION_INDEX = 5 # delay^2*energy
#EVALUATION_INDEX = 7 # BIPS/W
#EVALUATION_INDEX = 8 # BIPS^3/W
if 4 < len(sys.argv):
    EVALUATION_INDEX = int(sys.argv[4])
print(f"EVALUATION_INDEX={EVALUATION_INDEX}")

##########EVALUATION_INDEX = 4 # BIPS/W ##########
######EVALUATION_INDEX = 6 # BIPS^3/W ##########

def get_evaluation_factor(evaluation_index):
    if 7 == evaluation_index or 8 == evaluation_index:
        evaluation_factor = -1
    else:
        evaluation_factor = 1
    return evaluation_factor

EVALUATION_FACTOR = get_evaluation_factor(EVALUATION_INDEX)

schedule_result_database = {}
multi_obj_mode = 0 # GA\SA not support multi-obj for now

# BIPS/W = (0.01 / delay) / (energy / delay) = 1e-2 / energy
# BIPS^3/W = (0.01 / delay)^3 / (energy / delay) = 1e-6 / energy / delay^2
def get_hmp_metrics(delay, energy):
    return [delay,
            energy,
            delay * energy,
            delay * (energy ** 2),
            1e-2 / energy,
            energy * (delay ** 2),
            1e-6 / energy / (delay ** 2),
            -1e-2 / energy,
            -1e-6 / energy / (delay ** 2),            
            ]

#N_SRC_DOMAIN_TRAIN = N_SAMPLES_ALL
N_SRC_DOMAIN_TRAIN = 500 #500 #200
N_SRC_DOMAIN = 3#len(SRC_DOMAIN_LIST)

n_experiment = 1
print_info = True
mape_line_analysis = False

'''
if smoke_test:
    surrogate_model_tag_list = ["smoke_test2"]
    n_experiment = 1
elif "SC-202005121725" == hostname:
    # bookpad
    surrogate_model_tag_list = ["AdaGBRT"] #["ActBoost", "GBRT"]
    mape_line_analysis = True
    n_experiment = 1
    print_info = True
elif 2 < len(sys.argv):
    # desktop
    surrogate_model_tag_list = [
        #"ASPLOS06",
        "BOOM-Explorer",
        #"AdaBoost_MLP",
        #"ActBoost",
        # "CatBoostRegressor",
        # "LGBMRegressor",
        # "XGBoost",
        #"GBRT",
        #"M5P",
        #"GBRT-orh",
        #"AdaGBRT-no-iter",
        #"AdaGBRT-fi",
        #"AdaGBRT",
        #"AdaGBRT-cv",
        #"SVR_Matern",
        #"SemiBoost",
        #"BagGBRT",
    ]
    n_experiment = 1
    print_info = False
else:
    surrogate_model_tag_list = ["ASPLOS06"]
    #mape_line_analysis = True
    n_experiment = 1
    print_info = False
'''    