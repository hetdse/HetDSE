'''
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
'''
from config import N_SPACE_SIZE

var_ROCKET = [
   0, #"DispatchWidth"
   0, #"ExeInt"
   0, #"ExeFP"
   0, #"LSQ"
   1, #"Dcache"
   1, #"Icache"
   0, #"BP"
   1, # "L2cache"
]

if 147456 == N_SPACE_SIZE:
   var_ROCKET.append(0)  # "DecodeWidth"
   var_ROCKET.append(0)  # "RUU_SIZE"
   var_ROCKET.append(0)  # "Frequency"

var_BOOMv2 = [
   1, #"DispatchWidth" 4
   1, #"ExeInt"
   0, #"ExeFP"
   0, #"LSQ"
   1, #"Dcache"
   1, #"Icache"
   0, #"BP"
   1, # "L2cache"
]

if 147456 == N_SPACE_SIZE:
   var_BOOMv2.append(1)  # "DecodeWidth" 4
   var_BOOMv2.append(2)  #  "RUU_SIZE"128
   var_BOOMv2.append(0)  # "Frequency"

var_YANQIHU = [
   3, #"DispatchWidth" 6
   1, #"ExeInt"
   1, #"ExeFP"
   2, #"LSQ"
   1, #"Dcache"
   1, #"Icache"
   1, #"BP"
   1, # "L2cache"
]

if 147456 == N_SPACE_SIZE:
   var_YANQIHU.append(3) # "DecodeWidth" 6
   var_YANQIHU.append(3) # "RUU_SIZE"192
   var_YANQIHU.append(0) # "Frequency" 1.3G