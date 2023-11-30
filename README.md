# HetDSE

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [License](#license)

## Background
HetDSE is a heterogeneous multi-core CPU design space exploration framework.

## Install

This project uses torch, skopt, math, and several basic python package. Go check them out if you don't have them locally installed. See requirements.txt

## Usage

```sh
# run main() in SOTA_explore.py
# default is Clustering
$ python SOTA_explore.py
'''

'''
# more settings in SOTA_explore.py
$ python SOTA_explore.py BruteForce 0 23
# running args
# arg1: SOTA name of {BruteForce, GA, SA, HeterDSE_pc_m}
# arg2: workload_set_id
# arg3: case_num, workload set size
# hyper-args
# 1) sche_evaluation_index_list: resouce management id of {-1(RM-Naive),0(RM-EER),2(RM-ED),3(RM-EED),5(RM-EDD),7(RM-BIPS/W),8(RM-BIPS^3/W)}
# 2) selection: core selection method of {LUCIE, Clustering}
# 3) target_core_num: core selected num
# 4) EVALUATION_INDEX: HMP-RM evaluation metric id of {2(ED),3(EED),5(EDD),7(BIPS/W),8(BIPS^3/W)}
```

## Maintainers
[@hetdse](https://github.com/hetdse/HetDSE).

## License
[MIT](license)
