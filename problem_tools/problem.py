import numpy as np
from copy import deepcopy as dc
from collections import defaultdict
from itertools import product
from numba import jit
class Problem(object):
    def __init__(self, func, row_tree, col_tree, symmetric, verbose=False):
        self._func = func
        if symmetric and row_tree is not col_tree:
            raise ValueError("row_tree and col_tree parameters must be the "
                "same (as Python objects) if flag symmetric is `True`")
        self.symmetric = symmetric
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.row_data = row_tree.data
        self.col_data = col_tree.data
        self.shape = (len(row_tree.data), len(col_tree.data))
        l = np.arange(1, dtype=np.uint64)
        tmp = self.func(l, l)
        self.func_shape = tmp.shape[1:-1]
        self.dtype = tmp.dtype
        self._build(verbose)
    def _build(self, verbose=False):
        row_check = [[0]]
        self.row_far = []
        self.row_close = []
        self.row_notransition = []
        self.col_far = self.row_far
        self.col_close = self.row_close
        self.col_notransition = self.row_notransition
        self.row_tree.aux =[self.row_data.compute_aux(self.row_tree.index[0])]
        print(self.row_tree.aux)
        cur_level = 0
        while (self.row_tree.level[cur_level] < self.row_tree.level[cur_level+1]):
            # print (f' level {cur_level}')
            for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
                self.row_far.append([])
                self.row_close.append([])
            for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
                for j in row_check[i]:
                    if self.row_tree.is_far(i, self.col_tree, j):
                        self.row_far[i].append(j)
                        if self.row_tree is not self.col_tree:
                            self.col_far[j].append(i)
                    else:
                        self.row_close[i].append(j)
                        if self.row_tree is not self.col_tree:
                            self.col_close[j].append(i)
                # print (i, self.row_close[i])
            for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
                if i == 0:
                    self.row_notransition.append(not self.row_far[i])
                else:
                    self.row_notransition.append(not(self.row_far[i] or
                        not self.row_notransition[self.row_tree.parent[i]]))
            for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
                if(cur_level == 1):
                    self.row_tree.divide(i)
                else:
                    if (self.row_close[i] and not self.row_tree.child[i] and
                            self.row_tree.index[i].size >
                            self.row_tree.block_size):
                        nonzero_close = False
                        for j in self.row_close[i]:
                            if (self.col_tree.index[j].size >
                                    self.col_tree.block_size):
                                nonzero_close = True
                                break
                        if nonzero_close:
                            self.row_tree.divide(i)
            for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
                whom_to_check = []
                for j in self.row_close[i]:
                    whom_to_check.extend(self.col_tree.child[j])
                for j in self.row_tree.child[i]:
                    row_check.append(whom_to_check)
            # print (f' 3:')
            # for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
            #     print (i, self.row_close[i])
            # for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
            #     tmp_close = []
            #     if self.row_tree.child[i]:
            #         for j in self.row_close[i]:
            #             if not self.col_tree.child[j]:
            #                 tmp_close.append(j)
            #         self.row_close[i] = tmp_close
            self.row_tree.level.append(len(self.row_tree))
            # print (f' End:')
            # for i in range(self.row_tree.level[cur_level],self.row_tree.level[cur_level+1]):
            #     print (i, self.row_close[i])
            cur_level += 1
        # update number of levels
        self.num_levels = len(self.row_tree.level)-1
        self.row_tree.num_levels = self.num_levels
        self.col_tree.num_levels = self.num_levels
    def func(self, row, col):
        return self._func(self.row_data, row, self.col_data, col)
    def multilevel_close(self):
        self.lvl_close = dc(self.row_close)

        tree = self.row_tree
        close = self.row_close
        level_count = len(tree.level)-2
        row_size = tree.level[-1]
        self.other_lvl_close = [[] for i in range(row_size)]
        if self.add_up_level_close:
            for i in range(level_count-1, 0, -1):
                job = [j for j in range(tree.level[i], tree.level[i+1])]
                for ind in job:
                    if tree.child[ind] == []:
                        for cl in close[ind]:
                            for ch_cl in tree.child[cl]:
                                if not tree.is_far(ch_cl, tree, ind):
                                    self.add_child_to_close(ind, ch_cl)
    def add_child_to_close(self, main_node, node):
        close = self.row_close
        tree = self.row_tree
        self.other_lvl_close[node].append(main_node)
        if not tree.child[node] == [] :
            for ch_cl in tree.child[node]:
                if not tree.is_far(ch_cl, tree, main_node):
                    self.add_child_to_close(main_node, ch_cl)
    def schur_precompute(self):
        row_tree = self.row_tree
        N = row_tree.level[-1]

        self.schur_list = [set() for i in range(N)] # list of sets
        tmp_schur_dict = defaultdict(list) # like a dict, but if element is not exist, empty list are genereted

        for ind in range(N):
            close = self.lvl_close[ind]
            other_close = self.other_lvl_close[ind]
            # if self.symmetric:
            #     col_close = self.lvl_close[ind]
            # else:
            #     col_close = self.col_lvl_close[ind]
            for c1, c2 in product(close, close):
                self.schur_list[c1].add(c2)
                tmp_schur_dict[c1, c2].append(ind)
            for c1,c2 in product(close, other_close):
                self.schur_list[c1].add(c2)
                tmp_schur_dict[c1, c2].append(ind)

        self.schur_dict = dict(tmp_schur_dict)
