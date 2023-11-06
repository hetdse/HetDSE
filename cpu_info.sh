hostname
echo "package cpu number = "
grep 'physical id' /proc/cpuinfo | sort -u
echo "core number = "
grep 'cpu cores' /proc/cpuinfo | sort -u
echo "logical threads number = "
grep 'sibling' /proc/cpuinfo | sort -u
echo "model name = "
grep 'model name' /proc/cpuinfo | sort -u

grep 'cache size' /proc/cpuinfo | sort -u