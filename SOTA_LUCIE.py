import numpy as np

from config import multiobj_mode
from simulation_metrics import power_metric, delay_metric


class SOTA_LUCIE():

    def __init__(self, core_space, benchmarks):
        self.core_space = core_space
        self.benchmarks = benchmarks
        self.core_space_id_list = [x for x in range(len(core_space))]
        self.benchmarks_id_list = [x for x in range(len(benchmarks))]
        self.core_costs = np.zeros(len(core_space))
        self.m = np.zeros((len(core_space), len(benchmarks)))
        self.cm = [[] for c in range(len(core_space))]
        self.pareto_metrics_best = self.set_normalizetion()
        self.BD = np.zeros((len(core_space), len(core_space), len(self.benchmarks_id_list)))
        for c1 in self.core_space_id_list:
            for c2 in self.core_space_id_list:
                for bench in self.benchmarks_id_list:
                    delta_t, delta_p = self.delta_delay_power(c1, c2, bench)
                    #delta_p = self.delta_power(c1, c2, bench)
                    #delta_t = self.delta_delay(c1, c2, bench)
                    bd_part1 = np.sqrt(np.square(delta_p) + np.square(delta_t))
                    self.BD[c1][c2][bench] = bd_part1 * (1 - (delta_p + delta_t))
                    self.BD[c2][c1][bench] = bd_part1 * (1 + (delta_p + delta_t))

    def set_normalizetion(self):
        pareto_metrics_best = [[], []]
        from get_real_pareto_frontier import get_pareto_optimality_from_file_ax_interface
        for each_program_id, each_program in enumerate(self.benchmarks):
            real_pareto_data = get_pareto_optimality_from_file_ax_interface(each_program, multiobj_mode=multiobj_mode)
            pareto_x = [x[:-1] for x in real_pareto_data[2]]
            #pareto_y = [y / max(real_pareto_data[0]) for y in real_pareto_data[0]]
            #pareto_y2 = [y / max(real_pareto_data[1]) for y in real_pareto_data[1]]
            #core_space += pareto_x
            #core_space_metrics0 += pareto_y
            #core_space_metrics1 += pareto_y2
            pareto_metrics_best[0].append(min(real_pareto_data[0]))
            pareto_metrics_best[1].append(min(real_pareto_data[1]))
            for c in pareto_x:
                for each_c in self.core_space_id_list:
                    if (c == self.core_space[each_c]).all():
                        self.cm[each_c].append(each_program_id)
                        break
        return pareto_metrics_best

    # displacement cost
    def D_cal(self, core, bench):
        cm = np.argmin([self.BD[core][k][bench] for k in self.core_space_id_list])
        value = self.BD[core][cm][bench]
        return value

    def C_cal(self, core):
        core_cost = 0
        for bench in self.benchmarks_id_list:
            core_cost += self.D_cal(core, bench) * (self.m[core][bench] + 1)
        return core_cost

    def find_min_cost_core(self):
        min_core_cost = self.core_costs[0]
        c_argmin = self.core_space_id_list[0] #len(self.core_space) + 1 # init be a invalid value
        for c in self.core_space_id_list[1:]:
            if self.core_costs[c] < min_core_cost:
                min_core_cost = self.core_costs[c]
                c_argmin = c
        return c_argmin

    '''
    def BD_cal(self, c1, c2, b):
        delta_p = self.delta_power(c1, c2, b)
        delta_t = self.delta_delay(c1, c2, b)
        bd = np.sqrt(np.square(delta_p) + np.square(delta_t)) * (1 - (delta_p + delta_t))
        return bd
    '''

    def delta_delay_power(self, c1, c2, b):
        core_var_copy1 = self.core_space[c1].tolist()
        core_var_copy1.append(core_var_copy1[0])
        core_var_copy2 = self.core_space[c2].tolist()
        core_var_copy2.append(core_var_copy2[0])
        t1 = delay_metric(core_var_copy1, case_name=self.benchmarks[b])
        t2 = delay_metric(core_var_copy2, case_name=self.benchmarks[b]) 
        p1 = power_metric(core_var_copy1, case_name=self.benchmarks[b])
        p2 = power_metric(core_var_copy2, case_name=self.benchmarks[b]) 
        return (t1 - t2) / self.pareto_metrics_best[0][b], (p1 - p2) / self.pareto_metrics_best[1][b]

    def get_LUCIE(self, target_core_num):
        if len(self.core_space) <= target_core_num:
            return self.core_space
        while target_core_num < len(self.core_space_id_list):
            for c in self.core_space_id_list:
                self.core_costs[c] = self.C_cal(c)
                #print(f"core_costs[{c}] => {self.core_costs[c]}") 
            c = self.find_min_cost_core()
            #print(f"delete core={c} at {len(self.core_space_id_list)} / {target_core_num}")
            #self.core_space = np.delete(self.core_space, c, axis=0)
            self.core_space_id_list.remove(c)
            for b in self.cm[c]:
                cm = np.argmin([self.BD[c][k][b] for k in self.core_space_id_list])
                if b not in self.cm[cm]:
                    self.cm[cm].append(b)
                self.m[cm][b] += self.m[c][b] + 1

        return self.core_space[self.core_space_id_list]
