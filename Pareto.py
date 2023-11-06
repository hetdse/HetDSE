import numpy as np

class Pareto:
    def __init__(self, pop_obj, pop_obj2):
        self.pop_obj = pop_obj
        self.pop_obj2 = pop_obj2
        self.pop_size = len(pop_obj)
        self.num_obj = 2#pop_obj.shape[1]
        self.f = []
        self.sp = [[] for _ in range(self.pop_size)]
        self.np = np.zeros(self.pop_size, dtype=int)
        self.rank = np.zeros(self.pop_size, dtype=int)
        self.cd = np.zeros([self.pop_size, 1])

    def __index(self, i, ):
        return np.delete(range(self.pop_size), i)

    def __is_dominate(self, i, j, ):
        dominate_flag = False
        if self.pop_obj[i] < self.pop_obj[j] and self.pop_obj2[i] < self.pop_obj2[j]:
            dominate_flag = True
        return dominate_flag

    def __f1_dominate(self, ):
        f1 = []
        for i in range(self.pop_size):
            for j in self.__index(i):
                if self.__is_dominate(i, j):
                    if j not in self.sp[i]:
                        self.sp[i].append(j)
                elif self.__is_dominate(j, i):
                    self.np[i] += 1
            if self.np[i] == 0:
                self.rank[i] = 1
                f1.append(i)
        return f1

    def fast_non_dominate_sort(self, ):
        rank = 1
        f1 = self.__f1_dominate()
        while f1:
            self.f.append(f1)
            q = []
            for i in f1:
                for j in self.sp[i]:
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = rank + 1
                        q.append(j)
            rank += 1
            f1 = q

    '''
    def sort_obj_by(self, f=None, j=0, ):
        if 0 == f:
            index = np.argsort(self.pop_obj[f, j])
        else:
            index = np.argsort(self.pop_obj[:, j])
        return index

    def crowd_distance(self, ):
        for f in self.f:
            len_f1 = len(f) - 1
            for j in range(self.num_obj):
                index = self.sort_obj_by(f, j)
                sorted_obj = self.pop_obj[f][index]
                obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
                self.cd[f[index[0]]] = np.inf
                self.cd[f[index[-1]]] = np.inf
                for i in f:
                    k = np.argwhere(np.array(f)[index] == i)[:, 0][0]
                    if 0 < index[k] < len_f1:
                        self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj
    '''