import numpy as np
import sys
sys.path.insert(0,'..')
from copy import deepcopy as dc
from collections import defaultdict
from itertools import product
from time import time
from scipy.linalg import cho_factor, cho_solve, cholesky, lu
from scipy.linalg.blas import dgemm

from problem_tools.geometry_tools import Data, Tree
from problem_tools.problem import Problem
from functions import test_funcs
from numba import jit
from scipy.linalg.interpolative import interp_decomp
from scipy.sparse import csc_matrix as csc
from scipy.sparse import lil_matrix as lil
import scipy.sparse as sps

import matplotlib.pyplot as plt
import fmm3dbie as h3
import fmm3dpy as fmm3d
from memory_profiler import profile


class Factor(object):
    def __init__(self,pr, how_comp_T='block',proxy_p=(1,100), proxy_r=1., symmetric_fun = 0,proxy_type='fixed'):
        self.pr = pr
        self.csc_fun = pr.csc_fun
        self.count_proxy = []
        self.count_block_based = []
        self.proxy_p = proxy_p
        self.proxy_r = proxy_r
        self.tree = pr.row_tree
        self.n = self.tree.index[0].shape[0]
        tree = self.tree
        row_size = tree.level[-1]
        self.row_size = row_size
        self.L = [None for i in range(row_size)]
        self.inv_L = [None for i in range(row_size)]
        self.tail_od = [[] for i in range(row_size)]
        self.tail_L = [None for i in range(row_size)]
        self.off_diag = [None for i in range(row_size)]
        self.abasis = [None for i in range(row_size)]
        self.basis = [None for i in range(row_size)]
        self.local_abasis = [None for i in range(row_size)]
        self.local_basis = [None for i in range(row_size)]
        self.T = [None for i in range(row_size)]
        if pr.wtd_T:
            self.T_wtd = [None for i in range(row_size)]
        self.index_lvl = [None for i in range(row_size)]
        self.elim_list = set()
        self.prev_lvl_schur = {}
        self.symmetric_fun = symmetric_fun
        self.n_close_elem = 0
        if not pr.symmetric:
            self.col_tree = pr.col_tree
        if not symmetric_fun:
            self.U = [None for i in range(row_size)]
            self.inv_U = [None for i in range(row_size)]
            self.tail_od_U = [[] for i in range(row_size)]
            self.tail_U = [None for i in range(row_size)]
            self.off_diag_U = [None for i in range(row_size)]
    def init_index_lvl(self, ind):
        if not self.tree.child[ind]:
            self.index_lvl[ind] = self.tree.index[ind]
            if not self.pr.symmetric:
                self.index_lvl_U[ind] = self.col_tree.index[ind]
    def upd_index_lvl(self, ind, upd_type='row'):
            if upd_type == 'row':
                if self.tree.child[ind]:
                    basis_ch = np.array([], dtype='uint64')
                    for ch in self.tree.child[ind]:
                        basis_ch = np.hstack((basis_ch,self.basis[ch]))
                    self.index_lvl[ind] = basis_ch
                else:
                    self.index_lvl[ind] = dc(self.tree.index[ind])
            else:
                if self.col_tree.child[ind]:
                    basis_ch = np.array([], dtype='uint64')
                    for ch in self.col_tree.child[ind]:
                        basis_ch = np.hstack((basis_ch,self.basis_U[ch]))#row_basis[ch]))
                else:
                    basis_ch = self.col_tree.index[ind]
                self.index_lvl_U[ind] = basis_ch
    def ind_to_lvl(self, ind):
        tree = self.tree
        level_count = len(tree.level) - 2
        pr = self.pr
        for i in range(level_count-1, -1, -1):
            job = [j for j in
                    range(tree.level[i], tree.level[i+1])]
            if ind in job:
                return i
    def _dot_L(self, v, tr=0):
        pr = self.pr
        tree = self.tree
        level_count = len(tree.level) - 2
        close = pr.lvl_close
        #res_or = dc(v)
        res = dc(v)
        if tr:
            for i in range(level_count-1, -1, -1):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                for ind in job:
                    if pr.wtd_T:
                        T_T = np.linalg.inv(self.T_wtd[ind])
                    else:
                        T_T = np.linalg.inv(self.T[ind]).T
                    res[self.index_lvl[ind]] = T_T.dot(res[self.index_lvl[ind]])
                    if self.symmetric_fun:
                        U = self.L[ind].T
                    else:
                        U = self.U[ind]
                    if self.abasis[ind].shape[0] != 0:
                        res[self.abasis[ind]] = U.dot(res[self.abasis[ind]])
                    else:
                        continue
                    for ii in range(len(close[ind])):
                        if not self.tree.parent[close[ind][ii]] in close[ind]:
                            od_bl = self.off_diag[ind][ii]
                            if od_bl.shape[1] == self.basis[close[ind][ii]].shape[0] and od_bl.shape[1] != 0:
                                res[self.abasis[ind]] += od_bl.dot(res[self.basis[close[ind][ii]]])
                            elif od_bl.shape[1] == self.index_lvl[close[ind][ii]].shape[0] and od_bl.shape[1] != 0:
                                res[self.abasis[ind]] += od_bl.dot(res[self.index_lvl[close[ind][ii]]])
                    #res_or = dc(res)
            return res
        else:
            for i in range(level_count):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                job.reverse()
                for ind in job:
                    L = self.L[ind]
                    if self.abasis[ind].shape[0] != 0:
                        res[self.abasis[ind]] = L.dot(res[self.abasis[ind]])
                    else:
                        continue
                    for ii in range(len(close[ind])):
                        if not self.tree.parent[close[ind][ii]] in close[ind]:
                            if self.symmetric_fun:
                                od_bl = self.off_diag[ind][ii].T
                            else:
                                od_bl = self.off_diag_U[ind][ii]
                            if od_bl.shape[0] == self.basis[close[ind][ii]].shape[0] and od_bl.shape[0] != 0:
                                res[self.basis[close[ind][ii]]] += od_bl.dot(v[self.abasis[ind]])
                            elif od_bl.shape[0] == self.index_lvl[close[ind][ii]].shape[0] and od_bl.shape[0] != 0:
                                res[self.index_lvl[close[ind][ii]]] += od_bl.dot(v[self.abasis[ind]])
                    T = np.linalg.inv(self.T[ind])
                    res[self.index_lvl[ind]] = T.dot(res[self.index_lvl[ind]])
                    #res_or = dc(res)
            return res
    def _dot_L_tail(self, v, tr=0):
        l = self.tail_lvl
        pr = self.pr
        tree = self.tree
        level_count = len(tree.level) - 2
        res = np.zeros(v.shape[0], dtype = self.pr.dtype)
        if tr:
            # res = np.zeros(v.shape[0], dtype = self.pr.dtype)
            for i in range(level_count-1, -1, -1):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                if i == l:
                    rem_job = dc(job)
                    for ind in job:
                        if self.symmetric_fun:
                            U = self.tail_L[ind].T
                        else:
                            U = self.tail_U[ind]
                        res[self.basis[ind]] += U.dot(v[self.basis[ind]])
                        if self.abasis[ind].shape[0] != 0:
                            res[self.abasis[ind]] = v[self.abasis[ind]]
                        rem_job.remove(ind)
                        for ii in range(len(rem_job)):
                            od_bl = self.tail_od[ind][ii]
                            od_bl = np.linalg.inv(self.tail_L[ind]).dot(od_bl)
                            res[self.basis[ind]] += od_bl.dot(v[self.basis[rem_job[ii]]])
                else:
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            res[self.abasis[ind]] = dc(v[self.abasis[ind]])
            # if abs((res-v)[self.abasis[10]]).max() > 10e-5:
            #     print (f'tail_1 changed res in self.abasis[10]')
            return res
        else:
            # res = np.zeros(v.shape[0], dtype = self.pr.dtype)
            for i in range(level_count-1, -1, -1):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                if i == l:
                    rem_job = dc(job)
                    for ind in job:
                        res[self.basis[ind]] += self.tail_L[ind].dot(v[self.basis[ind]])
                        if self.abasis[ind].shape[0] != 0:
                            res[self.abasis[ind]] = v[self.abasis[ind]]
                        rem_job.remove(ind)
                        for ii in range(len(rem_job)):
                            if self.symmetric_fun:
                                od_bl = self.tail_od[ind][ii].T
                                od_bl = od_bl.dot(np.linalg.inv(self.tail_L[ind].T))
                            else:
                                od_bl = self.tail_od_U[ind][ii]
                                od_bl = od_bl.dot(np.linalg.inv(self.tail_U[ind]))
                            res[self.basis[rem_job[ii]]] += od_bl.dot(v[self.basis[ind]])
                else:
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            res[self.abasis[ind]] = dc(v[self.abasis[ind]])
            # if abs((res-v)[self.abasis[10]]).max() > 10e-5:
            #     print (f'tail_0 changed res in self.abasis[10]')
            return res
    def dot(self, v):
        res = self._dot_L(v, tr=1)
        res = self._dot_L_tail(res, tr=1)
        res = self._dot_L_tail(res, tr=0)
        res = self._dot_L(res, tr=0)
        return res
    def _solve_L(self, rhs, tr=0):
        pr = self.pr
        tree = self.tree
        level_count = len(tree.level) - 2
        close = pr.lvl_close
        if tr:
            ans = dc(rhs)
            for i in range(level_count):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                job.reverse()

                for ind in job:
                    if self.abasis[ind].shape[0] == 0:
                        continue
                    rhs_part = ans[self.abasis[ind]].copy() #new
                    for ii in range(len(close[ind])):
                        if not self.tree.parent[close[ind][ii]] in close[ind]:
                            od_bl = self.off_diag[ind][ii]
                            if od_bl.shape[1] == self.basis[close[ind][ii]].shape[0]:# and od_bl.shape[1] != 0:
                                rhs_part -= od_bl.dot(ans[self.basis[close[ind][ii]]]) # new
                                #rhs[self.abasis[ind]] -= od_bl.dot(ans[self.basis[close[ind][ii]]])
                            elif od_bl.shape[1] == self.index_lvl[close[ind][ii]].shape[0]:# and od_bl.shape[1] != 0:
                                rhs_part -= od_bl.dot(ans[self.index_lvl[close[ind][ii]]]) # new
                                #rhs[self.abasis[ind]] -= od_bl.dot(ans[self.index_lvl[close[ind][ii]]])
                    if self.symmetric_fun:
                        #if self.abasis[ind].shape[0] != 0:
                        #    ans[self.abasis[ind]] = (np.linalg.inv(self.L[ind]).T).dot(rhs[self.abasis[ind]])
                        ans[self.abasis[ind]] = (np.linalg.inv(self.L[ind]).T).dot(rhs_part) # new
                    else:
                        #if self.abasis[ind].shape[0] != 0:
                        #    ans[self.abasis[ind]] = (np.linalg.inv(self.U[ind])).dot(rhs[self.abasis[ind]])
                        ans[self.abasis[ind]] = (np.linalg.inv(self.U[ind])).dot(rhs_part) # new
                    if pr.wtd_T:
                        T_T = self.T_wtd[ind]
                    else:
                        T_T = self.T[ind].T

                    ans[self.index_lvl[ind]] = T_T.dot(ans[self.index_lvl[ind]])
                    #rhs = dc(ans)
            return ans
        else:
#            ans = np.zeros(self.n, dtype = self.pr.dtype)
            ans = dc(rhs)
            for i in range(level_count-1, -1, -1):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                for ind in job:
                    T = self.T[ind]
                    rhs[self.index_lvl[ind]] = T.dot(rhs[self.index_lvl[ind]])
                    if self.abasis[ind].shape[0] != 0:
                        ans[self.abasis[ind]] = np.linalg.inv(self.L[ind]).dot(rhs[self.abasis[ind]])
                    for ii in range(len(close[ind])):
                        if not self.tree.parent[close[ind][ii]] in close[ind]:
                            if self.symmetric_fun:
                                od_bl = self.off_diag[ind][ii].T
                            else:
                                od_bl = self.off_diag_U[ind][ii]
                            if od_bl.shape[0] == self.basis[close[ind][ii]].shape[0] and od_bl.shape[0] != 0:
                                rhs[self.basis[close[ind][ii]]] -= od_bl.dot(ans[self.abasis[ind]])
                            elif od_bl.shape[0] == self.index_lvl[close[ind][ii]].shape[0]  and od_bl.shape[0] != 0:
                                rhs[self.index_lvl[close[ind][ii]]] -= od_bl.dot(ans[self.abasis[ind]])
                for ind in job:
                    ans[self.basis[ind]] = rhs[self.basis[ind]]
            return ans
    def _solve_L_tail(self, rhs, tr=0):
        l = self.tail_lvl
        pr = self.pr
        tree = self.tree
        level_count = len(tree.level) - 2
        close = pr.lvl_close
        if tr:
            ans = dc(rhs) 
            for i in range(level_count):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                job.reverse()
#                print(tree.level[i])
#                print(tree.level[i+1])

                jlen = tree.level[i+1]-tree.level[i]
                juse = 0

                if i == l:
                    rem_job = dc(job)
#                    print(job)
                    for ind in job:
#                        print(ind)
                        if self.symmetric_fun:
                            U = self.tail_L[ind].T
                        else:
                            U = self.tail_U[ind]
                        ans[self.basis[ind]] = np.linalg.inv(U).dot(rhs[self.basis[ind]])
#                        print(ind,rem_job)
                        rem_job.remove(ind)
#                        print(len(self.tail_od[ind]))

                        for ii in range(juse):
                            jind = ind + ii+1
#                            print(ind,jind,ii)
                            od_bl = self.tail_od[ind][ii]
                            od_bl = np.linalg.inv(self.tail_U[ind]).dot(np.linalg.inv(self.tail_L[ind])).dot(od_bl)
                            ans[self.basis[ind]] -= od_bl.dot(ans[self.basis[jind]])
                        juse += 1    
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            ans[self.abasis[ind]] = rhs[self.abasis[ind]]
                else:
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            ans[self.abasis[ind]] = rhs[self.abasis[ind]]
            return ans
        else:
            ans = dc(rhs) 
            for i in range(level_count-1, -1, -1):
                job = [j for j in
                            range(tree.level[i], tree.level[i+1])
                            if not pr.row_notransition[j]]
                if i == l:
                    rem_job = dc(job)
                    for ind in job:
                        L = self.tail_L[ind]

                        ans[self.basis[ind]] = np.linalg.inv(L).dot(rhs[self.basis[ind]])
                        rem_job.remove(ind)
                        for ii in range(len(rem_job)):
                            if self.symmetric_fun:
                                od_bl = self.tail_od[ind][ii].T
                                od_bl = od_bl.dot(np.linalg.inv(L.T))
                            else:
                                od_bl = self.tail_od_U[ind][ii]
                                od_bl = od_bl.dot(np.linalg.inv(self.tail_U[ind]))
                            rhs[self.basis[rem_job[ii]]] -= od_bl.dot(ans[self.basis[ind]])
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            ans[self.abasis[ind]] = rhs[self.abasis[ind]]
                else:
                    for ind in job:
                        if self.abasis[ind].shape[0] != 0:
                            ans[self.abasis[ind]] = rhs[self.abasis[ind]]
            return ans
    def solve(self, rhs):
        ans = self._solve_L(rhs, tr=0)
        ans = self._solve_L_tail(ans, tr=0)
        ans = self._solve_L_tail(ans, tr=1)
        ans = self._solve_L(ans, tr=1)
        return ans
    def func(self, row, col):
        self.n_close_elem += row.shape[0]*col.shape[0]
        if self.csc_fun:
            return self.csc[np.ix_(row, col)].toarray()
        elif self.pr.q_fun:
            #pr = self.pr
            #norders = pr.order*np.ones(pr.npatches)
            #iptype = np.ones(pr.npatches)
            #npols = int((pr.order+1)*(pr.order+2)/2)
            #npts = pr.npatches*npols
            #bl = h3.helm_comb_dir_fds_matgen_woversamp(norders,pr.ixyzs,iptype,pr.srccoefs,pr.srcvals,pr.wts,pr.eps,pr.zpars,pr.ifds,pr.zfds,row+1,col+1)
            #if np.array_equal(row,col):
            return gen_block(self.pr, row, col)
        else:
            return self.pr._func(self.pr.row_data, row, self.pr.col_data, col)
def gen_block(pr,row,col):
    #norders = pr.order*np.ones(pr.npatches)
    #iptype = np.ones(pr.npatches)
    #npols = int((pr.order+1)*(pr.order+2)/2)
    #npts = pr.npatches*npols
#    norders, iptype, npols, npts = make_arrays(pr)
    bl = gen_block_fmm3dbie(pr, row, col)
    #h3.helm_comb_dir_fds_matgen_woversamp(norders,pr.ixyzs,iptype,pr.srccoefs,pr.srcvals,pr.wts,pr.eps,pr.zpars,pr.ifds,pr.zfds,row+1,col+1)
    bl = diag_block_upd(bl,pr.zpars[2], row, col, pr.coef)
    #if np.array_equal(row,col):
    #    bl -= diag_block_upd(pr.zpars[2], row.shape[0])#2*np.pi*pr.zpars[2]*np.identity(row.shape[0])
    return bl
def diag_block_upd(bl,kk, row, col, coef):
    if np.array_equal(row,col):
        bl += coef*kk*np.identity(row.shape[0])/2.0
    return bl
def gen_block_fmm3dbie(pr, row, col):
#    if(len(row)*len(col) > 10000):
#        print("In gen block:",len(row),len(col))
    return h3.helm_comb_dir_fds_block_matgen(pr.norders, pr.ixyzs,pr.iptype,pr.srccoefs,pr.srcvals,pr.wts,pr.eps,pr.zpars,pr.ifds,pr.zfds,row+1,col+1,ifwrite=0)
def make_arrays(pr):
    norders = pr.order*np.ones(pr.npatches)
    iptype = np.ones(pr.npatches)
    npols = int((pr.order+1)*(pr.order+2)/2)
    npts = pr.npatches*npols
    return norders, iptype, npols, npts





def buildmatrix_new(ind, factor, tau, l):
    pr = factor.pr
    tree = factor.tree
    ndim = tree.data.ndim
    pb_ind = []
    sch_len = 0
    for ii in factor.pr.schur_list[ind]:
        ban_ind = (ii in factor.pr.lvl_close[ind])
        ban_ind = ban_ind or (tree.parent[ii] in factor.pr.lvl_close[ind])
        if tree.child[ii]:
            for ii_ch in tree.child[ii]:
                ban_ind = ban_ind or (ii_ch in factor.pr.lvl_close[ind])
        if not ban_ind:
            pb_ind += [ii]
            if not factor.basis[ii] is None:
                sch_len += factor.basis[ii].shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_len += factor.index_lvl[ii].shape[0]
                else:
                    print(f'Error: buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
# proxy build points:
    lvl = len(tree.level) - 3
    p = factor.proxy_p[1]*(factor.proxy_p[0]**(lvl-l))
    #print(lvl, l, p, factor.proxy_p[0],factor.proxy_p[1])
    r = factor.proxy_r
    c = np.zeros(ndim)
    for i_ndim in range(ndim):
        c[i_ndim] = (factor.tree.aux[ind][:,i_ndim][0] + factor.tree.aux[ind][:,i_ndim][1])/2
    box_size = np.linalg.norm(factor.tree.aux[ind][1] - factor.tree.aux[ind][0])
    theta = np.linspace(0, 2*np.pi, p, endpoint=False)
    if ndim == 2:
        proxy =  r * box_size * np.vstack((c[0] + np.cos(theta), c[1] + np.sin(theta)))
        proxy[0] -= (c * r * box_size - c)[0]
        proxy[1] -= (c * r * box_size - c)[1]
    elif ndim == 3:
        proxy = fibonacci_sphere(p, r * box_size, c)
    else:
        raise NameError('In progress')
# zero matrix:
    if factor.symmetric_fun:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len + p), dtype=pr.dtype)
    else:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len * 2 + p * 2), dtype=pr.dtype)
# build sch blocks
    index0 = factor.index_lvl[ind]
    sch = np.zeros((factor.index_lvl[ind].shape[0], sch_len),dtype=pr.dtype)
    tmp = 0
    for ii in pb_ind:
        if ii in factor.elim_list:
            tmp_mat = factor.func(factor.index_lvl[ind], factor.index_lvl[ii])
            if not tmp_mat is None:
                if tmp_mat.shape[1] != factor.basis[ii].shape[0]:
                    tmp_mat = tmp_mat[:, factor.local_basis[ii]]
            sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='b')
            if not sch_tmp is None:
                sch[:,tmp:tmp+sch_tmp.shape[1]] = tmp_mat-sch_tmp
                tmp += sch_tmp.shape[1]
        else:
            if not factor.index_lvl[ii] is None:
                tmp_mat = factor.func(factor.index_lvl[ind], factor.index_lvl[ii])
                sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='i')
                if not sch_tmp is None:
                    sch[:,tmp:tmp+sch_tmp.shape[1]] = tmp_mat - sch_tmp
                    tmp += sch_tmp.shape[1]
            else:
                print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                raise KeyboardInterrupt
# add proxy to matrix
    proxy_data = Data(tree.data.ndim, p, proxy, close_r=tree.data.close_r)
    proxy_mat0 = factor.pr._func(tree.data, index0, proxy_data, np.arange(p))
    D = np.diag(factor.pr.wts[index0])
    proxy_mat = D.dot(proxy_mat0)
    tmp = 0
    matrix[:, tmp : tmp + p] = proxy_mat
    tmp += p

    matrix[:,tmp:tmp+sch_len] = sch
    tmp += sch_len
    if not pr.half_sym:
        sch = np.zeros((sch_len,factor.index_lvl[ind].shape[0]), dtype=pr.dtype)
        tmp1 = 0
        for ii in pb_ind:
            if ii in factor.elim_list:
                tmp_mat = factor.func(factor.index_lvl[ii], factor.index_lvl[ind])
                if not tmp_mat is None:
                    if tmp_mat.shape[0] != factor.basis[ii].shape[0]:
                        tmp_mat = tmp_mat[factor.local_basis[ii]]
                sch_tmp = schur(ii, ind , factor, ib_row='b', ib_col='i')
                if not sch_tmp is None:
                    sch[tmp1:tmp1+sch_tmp.shape[0]] = tmp_mat-sch_tmp
                    tmp1 += sch_tmp.shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    tmp_mat = factor.func(factor.index_lvl[ii], factor.index_lvl[ind])
                    sch_tmp = schur(ii, ind , factor, ib_row='i', ib_col='i')
                    if not sch_tmp is None:
                        sch[tmp1:tmp1+sch_tmp.shape[0]] = tmp_mat - sch_tmp
                        tmp1 += sch_tmp.shape[0]
                else:
                    print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
        if hasattr(tree.data, 'k'):
            proxy_data.k = tree.data.k
        # if hasattr(tree.data, 'norms'):
        #     proxy_data.norms = tree.data.norms

        test_mat20 = factor.pr._func(tree.data,index0,proxy_data, np.arange(p))
        test_mat2 = D.dot(test_mat20)
        matrix[:, tmp : tmp + p] = test_mat2
        tmp += p
        matrix[:,tmp:tmp+sch_len] = sch.T
        # print(ind)
        # if ind == 10:
        #     print (f'col sch: {np.linalg.norm(sch)}')
        tmp += sch_len
    return matrix








def buildmatrix_new3(ind, factor, tau, l):
    pr = factor.pr
    tree = factor.tree
    ndim = tree.data.ndim
    pb_ind = []
    sch_len = 0
    for ii in factor.pr.schur_list[ind]:
        ban_ind = (ii in factor.pr.lvl_close[ind])
        ban_ind = ban_ind or (tree.parent[ii] in factor.pr.lvl_close[ind])
        if tree.child[ii]:
            for ii_ch in tree.child[ii]:
                ban_ind = ban_ind or (ii_ch in factor.pr.lvl_close[ind])
        if not ban_ind:
            pb_ind += [ii]
            if not factor.basis[ii] is None:
                sch_len += factor.basis[ii].shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_len += factor.index_lvl[ii].shape[0]
                else:
                    print(f'Error: buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
# proxy build points:
    lvl = len(tree.level) - 3
    p = factor.proxy_p[1]*(factor.proxy_p[0]**(lvl-l))
    #print(lvl, l, p, factor.proxy_p[0],factor.proxy_p[1])
    r = factor.proxy_r
    c = np.zeros(ndim)
    for i_ndim in range(ndim):
        c[i_ndim] = (factor.tree.aux[ind][:,i_ndim][0] + factor.tree.aux[ind][:,i_ndim][1])/2
    box_size = np.linalg.norm(factor.tree.aux[ind][1] - factor.tree.aux[ind][0])
    theta = np.linspace(0, 2*np.pi, p, endpoint=False)
    if ndim == 2:
        proxy =  r * box_size * np.vstack((c[0] + np.cos(theta), c[1] + np.sin(theta)))
        proxy[0] -= (c * r * box_size - c)[0]
        proxy[1] -= (c * r * box_size - c)[1]
    elif ndim == 3:
        proxy = fibonacci_sphere(p, r * box_size, c)
    else:
        raise NameError('In progress')
# zero matrix:
    if factor.symmetric_fun:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len + p*3), dtype=pr.dtype)
    else:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len * 2 + p * 3), dtype=pr.dtype)
# build sch blocks
    index0 = factor.index_lvl[ind]
    sch = np.zeros((factor.index_lvl[ind].shape[0], sch_len),dtype=pr.dtype)
    tmp = 0
    for ii in pb_ind:
        if ii in factor.elim_list:
            sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='b')
            if not sch_tmp is None:
                sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                tmp += sch_tmp.shape[1]
        else:
            if not factor.index_lvl[ii] is None:
                sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='i')
                if not sch_tmp is None:
                    sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                    tmp += sch_tmp.shape[1]
            else:
                print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                raise KeyboardInterrupt
# add proxy to matrix
    proxy_data = Data(tree.data.ndim, p, proxy, close_r=tree.data.close_r)
    proxy_mats0 = test_funcs.exp_distance_h2t(tree.data, index0, proxy_data, np.arange(p))
    proxy_matd0 = test_funcs.double_layer(tree.data,index0,proxy_data,np.arange(p))
    D = np.diag(factor.pr.wts[index0])
    proxy_mat_s = D.dot(proxy_mats0)
    proxy_mat_d = D.dot(proxy_matd0)
    tmp = 0
    matrix[:, tmp : tmp + p] = proxy_mats0
    tmp += p
    matrix[:, tmp : tmp + p] = proxy_mat_s
    tmp += p
    matrix[:, tmp : tmp + p] = proxy_mat_d
    tmp += p


    matrix[:,tmp:tmp+sch_len] = sch
    tmp += sch_len
    if not pr.half_sym:
        sch = np.zeros((sch_len,factor.index_lvl[ind].shape[0]), dtype=pr.dtype)
        tmp1 = 0
        for ii in pb_ind:
            if ii in factor.elim_list:
                sch_tmp = schur(ii, ind , factor, ib_row='b', ib_col='i')
                if not sch_tmp is None:
                    sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                    tmp1 += sch_tmp.shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_tmp = schur(ii, ind , factor, ib_row='i', ib_col='i')
                    if not sch_tmp is None:
                        sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                        tmp1 += sch_tmp.shape[0]
                else:
                    print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
        # if hasattr(tree.data, 'norms'):
        #     proxy_data.norms = tree.data.norms

        matrix[:,tmp:tmp+sch_len] = sch.T
        # print(ind)
        # if ind == 10:
        #     print (f'col sch: {np.linalg.norm(sch)}')
        tmp += sch_len
    return matrix








def buildmatrix_new2(ind, factor, tau, l):
    pr = factor.pr
    tree = factor.tree
    ndim = tree.data.ndim
    pb_ind = []
    sch_len = 0
    for ii in factor.pr.schur_list[ind]:
        ban_ind = (ii in factor.pr.lvl_close[ind])
        ban_ind = ban_ind or (tree.parent[ii] in factor.pr.lvl_close[ind])
        if tree.child[ii]:
            for ii_ch in tree.child[ii]:
                ban_ind = ban_ind or (ii_ch in factor.pr.lvl_close[ind])
        if not ban_ind:
            pb_ind += [ii]
            if not factor.basis[ii] is None:
                sch_len += factor.basis[ii].shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_len += factor.index_lvl[ii].shape[0]
                else:
                    print(f'Error: buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
# proxy build points:
    lvl = len(tree.level) - 3
    p = factor.proxy_p[1]*(factor.proxy_p[0]**(lvl-l))
    #print(lvl, l, p, factor.proxy_p[0],factor.proxy_p[1])
    r = factor.proxy_r
    c = np.zeros(ndim)
    for i_ndim in range(ndim):
        c[i_ndim] = (factor.tree.aux[ind][:,i_ndim][0] + factor.tree.aux[ind][:,i_ndim][1])/2
    box_size = np.linalg.norm(factor.tree.aux[ind][1] - factor.tree.aux[ind][0])
    theta = np.linspace(0, 2*np.pi, p, endpoint=False)
    if ndim == 2:
        proxy =  r * box_size * np.vstack((c[0] + np.cos(theta), c[1] + np.sin(theta)))
        proxy[0] -= (c * r * box_size - c)[0]
        proxy[1] -= (c * r * box_size - c)[1]
    elif ndim == 3:
        proxy = fibonacci_sphere(p, r * box_size, c)
    else:
        raise NameError('In progress')
# zero matrix:
    if factor.symmetric_fun:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len + p), dtype=pr.dtype)
    else:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len * 2 + p * 2), dtype=pr.dtype)
# build sch blocks
    index0 = factor.index_lvl[ind]
    sch = np.zeros((factor.index_lvl[ind].shape[0], sch_len),dtype=pr.dtype)
    tmp = 0
    for ii in pb_ind:
        if ii in factor.elim_list:
            sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='b')
            if not sch_tmp is None:
                sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                tmp += sch_tmp.shape[1]
        else:
            if not factor.index_lvl[ii] is None:
                sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='i')
                if not sch_tmp is None:
                    sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                    tmp += sch_tmp.shape[1]
            else:
                print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                raise KeyboardInterrupt
# add proxy to matrix
    proxy_data = Data(tree.data.ndim, p, proxy, close_r=tree.data.close_r)
    proxy_mat0 = factor.pr._func(tree.data, index0, proxy_data, np.arange(p))
    D = np.diag(factor.pr.wts[index0])
    proxy_mat = D.dot(proxy_mat0)
    tmp = 0
    matrix[:, tmp : tmp + p] = proxy_mat
    tmp += p

    matrix[:,tmp:tmp+sch_len] = sch
    tmp += sch_len
    if not pr.half_sym:
        sch = np.zeros((sch_len,factor.index_lvl[ind].shape[0]), dtype=pr.dtype)
        tmp1 = 0
        for ii in pb_ind:
            if ii in factor.elim_list:
                sch_tmp = schur(ii, ind , factor, ib_row='b', ib_col='i')
                if not sch_tmp is None:
                    sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                    tmp1 += sch_tmp.shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_tmp = schur(ii, ind , factor, ib_row='i', ib_col='i')
                    if not sch_tmp is None:
                        sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                        tmp1 += sch_tmp.shape[0]
                else:
                    print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
        if hasattr(tree.data, 'k'):
            proxy_data.k = tree.data.k
        # if hasattr(tree.data, 'norms'):
        #     proxy_data.norms = tree.data.norms

        test_mat2 = factor.pr._func(tree.data,index0, proxy_data, np.arange(p))
        matrix[:, tmp : tmp + p] = test_mat2
        tmp += p
        matrix[:,tmp:tmp+sch_len] = sch.T
        # print(ind)
        # if ind == 10:
        #     print (f'col sch: {np.linalg.norm(sch)}')
        tmp += sch_len
    return matrix








def buildmatrix(ind, factor, tau, l):
    pr = factor.pr
    tree = factor.tree
    ndim = tree.data.ndim
    pb_ind = []
    sch_len = 0
    for ii in factor.pr.schur_list[ind]:
        ban_ind = (ii in factor.pr.lvl_close[ind])
        ban_ind = ban_ind or (tree.parent[ii] in factor.pr.lvl_close[ind])
        if tree.child[ii]:
            for ii_ch in tree.child[ii]:
                ban_ind = ban_ind or (ii_ch in factor.pr.lvl_close[ind])
        if not ban_ind:
            pb_ind += [ii]
            if not factor.basis[ii] is None:
                sch_len += factor.basis[ii].shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_len += factor.index_lvl[ii].shape[0]
                else:
                    print(f'Error: buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
# proxy build points:
    lvl = len(tree.level) - 3
    p = factor.proxy_p[1]*(factor.proxy_p[0]**(lvl-l))
    #print(lvl, l, p, factor.proxy_p[0],factor.proxy_p[1])
    r = factor.proxy_r
    c = np.zeros(ndim)
    for i_ndim in range(ndim):
        c[i_ndim] = (factor.tree.aux[ind][:,i_ndim][0] + factor.tree.aux[ind][:,i_ndim][1])/2
    box_size = np.linalg.norm(factor.tree.aux[ind][1] - factor.tree.aux[ind][0])
    theta = np.linspace(0, 2*np.pi, p, endpoint=False)
    if ndim == 2:
        proxy =  r * box_size * np.vstack((c[0] + np.cos(theta), c[1] + np.sin(theta)))
        proxy[0] -= (c * r * box_size - c)[0]
        proxy[1] -= (c * r * box_size - c)[1]
    elif ndim == 3:
        proxy = fibonacci_sphere(p, r * box_size, c)
    else:
        raise NameError('In progress')
# zero matrix:
    if factor.symmetric_fun:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len + p), dtype=pr.dtype)
    else:
        matrix = np.zeros((factor.index_lvl[ind].shape[0], sch_len * 2 + p * 2), dtype=pr.dtype)
# build sch blocks
    index0 = factor.index_lvl[ind]
    sch = np.zeros((factor.index_lvl[ind].shape[0], sch_len),dtype=pr.dtype)
    tmp = 0
    for ii in pb_ind:
        if ii in factor.elim_list:
            sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='b')
            if not sch_tmp is None:
                sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                tmp += sch_tmp.shape[1]
        else:
            if not factor.index_lvl[ii] is None:
                sch_tmp = schur(ind, ii , factor, ib_row='i', ib_col='i')
                if not sch_tmp is None:
                    sch[:,tmp:tmp+sch_tmp.shape[1]] = sch_tmp
                    tmp += sch_tmp.shape[1]
            else:
                print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                raise KeyboardInterrupt
# add proxy to matrix
    proxy_data = Data(tree.data.ndim, p, proxy, close_r=tree.data.close_r)
    proxy_mat = factor.pr._func(tree.data, index0, proxy_data, np.arange(p))
    tmp = 0
    matrix[:, tmp : tmp + p] = proxy_mat
    tmp += p

    matrix[:,tmp:tmp+sch_len] = sch
    tmp += sch_len
    if not pr.half_sym:
        sch = np.zeros((sch_len,factor.index_lvl[ind].shape[0]), dtype=pr.dtype)
        tmp1 = 0
        for ii in pb_ind:
            if ii in factor.elim_list:
                sch_tmp = schur(ii, ind , factor, ib_row='b', ib_col='i')
                if not sch_tmp is None:
                    sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                    tmp1 += sch_tmp.shape[0]
            else:
                if not factor.index_lvl[ii] is None:
                    sch_tmp = schur(ii, ind , factor, ib_row='i', ib_col='i')
                    if not sch_tmp is None:
                        sch[tmp1:tmp1+sch_tmp.shape[0]] = sch_tmp
                        tmp1 += sch_tmp.shape[0]
                else:
                    print(f'Error: _node_buildmatrix0 ind: {ind}, factor.index_lvl[{ii}] is None!')
                    raise KeyboardInterrupt
        if hasattr(tree.data, 'k'):
            proxy_data.k = tree.data.k
        # if hasattr(tree.data, 'norms'):
        #     proxy_data.norms = tree.data.norms

        test_mat2 = factor.pr._func(tree.data,index0, proxy_data, np.arange(p))
        matrix[:, tmp : tmp + p] = test_mat2
        tmp += p
        matrix[:,tmp:tmp+sch_len] = sch.T
        # print(ind)
        # if ind == 10:
        #     print (f'col sch: {np.linalg.norm(sch)}')
        tmp += sch_len
    return matrix





def block_schur(factor, ind, row_i, col_i, close, res, ib_row = 'i', ib_col = 'i',):
    if factor.off_diag[ind][close.index(col_i)].shape[0] == 0:
        return None
    if factor.symmetric_fun:
        if factor.off_diag[ind][close.index(row_i)].shape[0] == 0:
            return None
    else:
        if factor.off_diag_U[ind][close.index(row_i)].shape[0] == 0:
            return None
    if factor.symmetric_fun:
        col_mat = factor.off_diag[ind][close.index(row_i)].T
    else:
        col_mat = factor.off_diag_U[ind][close.index(row_i)]
    row_mat = factor.off_diag[ind][close.index(col_i)]
    upd_mat = col_mat.dot(row_mat)
    res = add_res_and_tmp(res, upd_mat, factor=factor, i_row=row_i, i_col=col_i)
    if  ib_row == 'b':
        if res.shape[0] != factor.basis[row_i].shape[0]:
            return res[factor.local_basis[row_i]]
    if ib_col == 'b':
        if res.shape[1] != factor.basis[col_i].shape[0]:
            return res[:, factor.local_basis[col_i]]
    return res
def level_schour(row_i, col_i, factor, ib_row = 'i', ib_col = 'i', lvl_type = 'row', ban_list=[]):
    close = factor.pr.lvl_close
    other_close = factor.pr.other_lvl_close
    pr = factor.pr
    res = None
#    if (row_i, col_i) in pr.schur_dict.keys():
#        job = pr.schur_dict[(row_i, col_i)]
#    elif (col_i, row_i) in pr.schur_dict.keys():
#        job = pr.schur_dict[(col_i, row_i)]
#    else:
#        print(f'Warning! level_schour: {row_i}, {col_i} are not in sch list, but shold be!')
    job = pr.schur_dict.get((row_i, col_i), None)
    if job is None:
        job = pr.schur_dict.get((col_i, row_i), None)
    if job is None:
        raise NameError(f'Error! level_schour: {row_i}, {col_i} are not in sch list, but shold be!')

    for ind in job:
        if (not ind in ban_list) and (ind in factor.elim_list):
            tmp = block_schur(factor, ind, row_i, col_i, close[ind]+other_close[ind], None, ib_row = ib_row, ib_col = ib_col)
            res = add_res_and_tmp(res, tmp, factor=factor, i_row=row_i, i_col=col_i)
    return res
def add_res_and_tmp(res, tmp, factor=None, i_row=0, i_col=0):
    if res is None:
        if tmp is None:
            return None
        else:
            return tmp
    else:
        if tmp is None:
            return res
        else:
            if res.shape[0] == tmp.shape[0]:
                if res.shape[1] == tmp.shape[1]:
                    return res + tmp
                elif res.shape[1] > tmp.shape[1]:
                    return res[:, factor.local_basis[i_col]] + tmp
                elif res.shape[1] < tmp.shape[1]:
                    return tmp[:, factor.local_basis[i_col]] + res
            elif res.shape[0] > tmp.shape[0]:
                if res.shape[1] == tmp.shape[1]:
                    return res[factor.local_basis[i_row]] + tmp
                elif res.shape[1] > tmp.shape[1]:
                    return res[np.ix_(factor.local_basis[i_row],factor.local_basis[i_col])] + tmp
            elif res.shape[0] < tmp.shape[0]:
                if res.shape[1] == tmp.shape[1]:
                    return tmp[factor.local_basis[i_row]] + res
                elif res.shape[1] < tmp.shape[1]:
                    return (tmp.ravel()[(factor.local_basis[i_col] + (factor.local_basis[i_row] * tmp.shape[1]).reshape((-1,1))).ravel()]).reshape(factor.local_basis[i_row].size, factor.local_basis[i_col].size) + res
                    # return tmp[np.ix_(factor.local_basis[i_row],factor.local_basis[i_col])] + res
    print ('Warning! add_res_and_tmp')
    return res
def schur_tail(row_i, col_i, factor, job, row_col = 'row'):
    res = schur(row_i, col_i, factor, ib_row = 'b', ib_col = 'b')
    for ind in job:
        if row_col == 'row':
            if ind == row_i:
                break
        else:
            if ind == col_i:
                break
        inv_L = np.linalg.inv(factor.tail_L[ind])
        if factor.symmetric_fun:
            inv_U = np.linalg.inv(factor.tail_L[ind].T)
        else:
            inv_U = np.linalg.inv(factor.tail_U[ind])
        row_mat = factor.tail_od[ind][col_i-ind-1]
        if factor.symmetric_fun:
            col_mat = factor.tail_od[ind][row_i-ind-1].T
        else:
            col_mat = factor.tail_od_U[ind][row_i-ind-1]
        if res is None:
            res = col_mat.dot(inv_U.dot(inv_L.dot(row_mat)))
        else:
            res += col_mat.dot(inv_U.dot(inv_L.dot(row_mat)))
    return res
#@profile
def factorize_lvl(factor, ind_l, tau = 1e-3, l = 0):
    pr = factor.pr
    close = pr.lvl_close
    other_close = factor.pr.other_lvl_close
    elim_list = factor.elim_list
    for ind in ind_l:
#   T:
        build_T(ind, factor, tau, l)
        if pr.wtd_T:
            build_T_wtd(ind, factor)
        T = factor.T[ind]
        if pr.wtd_T:
            T_T = factor.T_wtd[ind]
        else:
            T_T = T.T
        b = factor.basis[ind]
        abasis = factor.local_abasis[ind]
        basis = factor.local_basis[ind]
        if (set(factor.index_lvl[ind]) == set(b)):
            factor.L[ind] = np.zeros((0,0))
            inv_L = np.zeros((0,0))
            factor.off_diag[ind] = []
            for i_cl in close[ind]:
                factor.off_diag[ind].append(np.zeros((b.shape[0],0)))
            if not factor.symmetric_fun:
                factor.U[ind] = np.zeros((0,0))
                factor.off_diag_U[ind] = []
                for i_cl in close[ind]:
                    factor.off_diag_U[ind].append(np.zeros((b.shape[0],0)).T)
                inv_U = np.zeros((0,0))
            continue
# ==============================
# compute L, U
        tmp_mat = factor.func(factor.index_lvl[ind], factor.index_lvl[ind])
        sch_bl = schur(ind, ind, factor)
        if not sch_bl is None:
            try:
                tmp_mat -= sch_bl
            except:
                print (f'factorize_lvl:Warning! {ind, i_cl, sch_bl.shape, tmp_mat.shape}')
                if sch_bl.shape[0] < tmp_mat.shape[0]:
                    tmp_mat[factor.local_basis[ind]] -= sch_bl
                else:
                    print ('Warning! factorize_lvl Unexpected sizes!')
        matrix_diag = T.dot(tmp_mat).dot(T_T)
        if factor.symmetric_fun:
            factor.L[ind] = cholesky(matrix_diag[np.ix_(abasis,abasis)],lower=1)
            inv_L = np.linalg.inv(factor.L[ind])
            inv_U = inv_L.T
        else:
            factor.L[ind], factor.U[ind] = lu(matrix_diag[np.ix_(abasis,abasis)], permute_l=True)
            inv_L = np.linalg.inv(factor.L[ind])
            inv_U = np.linalg.inv(factor.U[ind])
#======================
#   factorize:
        cl_offdiag = []
        cl_offdiag_U = []
        for i_cl in close[ind]+other_close[ind]:
            if not i_cl in elim_list:
                try:
                    tmp_mat = factor.func(factor.index_lvl[ind], factor.index_lvl[i_cl])
                except:
                    raise NameError('err')
                sch_bl = schur(ind, i_cl, factor)
                if not sch_bl is None:
                    try:
                        tmp_mat -= sch_bl
                    except:
                        print (f'factorize_lvl:Warning! {ind, i_cl, sch_bl.shape, tmp_mat.shape}')
                        if sch_bl.shape[0] < tmp_mat.shape[0]:
                            tmp_mat[factor.local_basis[ind]] -= sch_bl
                        else:
                            print ('Warning! factorize_lvl Unexpected sizes!')
                if i_cl == ind:
                    matrix_diag = T.dot(tmp_mat).dot(T_T)
                    cl_offdiag.append(inv_L.dot(matrix_diag[np.ix_(abasis,basis)]))
                    if not factor.symmetric_fun:
                        cl_offdiag_U.append(matrix_diag[np.ix_(basis,abasis)].dot(inv_U))
                else:
                    matrix_offdiag = T.dot(tmp_mat)
                    cl_offdiag.append(inv_L.dot(matrix_offdiag[abasis]))
                    if not factor.symmetric_fun:
                        tmp_mat_U = factor.func(factor.index_lvl[i_cl], factor.index_lvl[ind])
                        sch_bl = schur(i_cl, ind, factor)
                        if not sch_bl is None:
                            try:
                                tmp_mat_U -= sch_bl
                            except:
                                print (f'factorize_lvl:Warning! {ind, i_cl, sch_bl.shape, tmp_mat_U.shape}')
                                if sch_bl.shape[0] < tmp_mat_U.shape[0]:
                                    tmp_mat_U[factor.local_basis[ind]] -= sch_bl
                                else:
                                    print ('Warning! factorize_lvl Unexpected sizes!')
                        matrix_offdiag_U = (tmp_mat_U).dot(T_T)
                        cl_offdiag_U.append(matrix_offdiag_U[:,abasis].dot(inv_U))
            else:
                tmp_mat = factor.func(factor.index_lvl[ind], factor.basis[i_cl])
                sch_bl = schur(ind, i_cl, factor)
                if not sch_bl is None:
                    if tmp_mat.shape == sch_bl.shape:
                        tmp_mat = tmp_mat - sch_bl
                    else:
                        try:
                            tmp_mat = tmp_mat - sch_bl[:,factor.local_basis[i_cl]]
                        except:
                            raise KeyboardInterrupt(f'factorize_lvl: Err!sch_bl wrong size')
                matrix_offdiag = T.dot(tmp_mat)
                cl_offdiag.append(inv_L.dot(matrix_offdiag[abasis]))
                if not factor.symmetric_fun:
                    tmp_mat_U = factor.func(factor.basis[i_cl], factor.index_lvl[ind])
                    sch_bl = schur(i_cl, ind, factor)
                    if not sch_bl is None:
                        if tmp_mat_U.shape == sch_bl.shape:
                            tmp_mat_U = tmp_mat_U - sch_bl
                        else:
                            try:
                                tmp_mat_U = tmp_mat_U - sch_bl[factor.local_basis[i_cl]]
                            except:
                                raise KeyboardInterrupt(f'factorize_lvl: Err!sch_bl wrong size')
                    matrix_offdiag_U =(tmp_mat_U).dot(T_T)
                    cl_offdiag_U.append(matrix_offdiag_U[:,abasis].dot(inv_U))
        factor.off_diag[ind] = cl_offdiag
        if not factor.symmetric_fun:
            factor.off_diag_U[ind] = cl_offdiag_U
        elim_list.add(ind)
    if l > factor.tail_lvl:
        prep_next_lvl_schur(factor, l-1)
    return factor
def build_T_old(ind,factor, tau):
    pr = factor.pr
    index_lvl = factor.index_lvl[ind]
    if pr.row_far[ind] == [] and not factor.tree.child[ind]:
        if factor.tree.child[ind]:
            print (f"Bad tree!!!!! {ind} has kids and has no far! ")
        b = factor.index_lvl[ind]
        T0 = np.identity(b.shape[0])
    else:
        matrix = buildmatrix(ind, factor)
        if matrix.size:
            matrix = matrix.reshape(matrix.shape[0], -1)
            pivots, T0 = maxvol_svd(matrix, tau, 0., 1.05, job='R')
            result_basis = index_lvl[pivots]
        else:
            result_basis = np.zeros(0, dtype=np.uint64)
            T0 = np.zeros((basis.size, 0), dtype=matrix.dtype)
    factor.abasis[ind] = np.setdiff1d(index_lvl, result_basis, assume_unique=True)
    factor.basis[ind] = result_basis
    sorter = np.argsort(index_lvl)
    factor.local_abasis[ind] =  sorter[np.searchsorted(index_lvl, factor.abasis[ind], sorter=sorter)]
    factor.local_basis[ind] = pivots
    T_1 = np.identity(T0.shape[0])
    T0[factor.local_abasis[ind]] *= -1
    T_1[:,pivots] = T0
    factor.T[ind] = T_1
@jit(nopython=True)
def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse
def build_T(ind,factor, tau, l):
    pr = factor.pr
    index_lvl = factor.index_lvl[ind]
    if(pr.ibuild_matrix == 1):
        matrix = buildmatrix(ind, factor, tau, l)
    elif(pr.ibuild_matrix == 2):
        matrix = buildmatrix_new(ind, factor, tau, l)
    elif(pr.ibuild_matrix == 3):
        matrix = buildmatrix_new2(ind, factor, tau, l)
    elif(pr.ibuild_matrix == 4):
        matrix = buildmatrix_new3(ind, factor, tau, l)

    matrix = matrix.T    
    #if matrix.T.shape[0] == 0 or matrix.T.shape[1] == 0:
#    print(ind, matrix.shape)
    if(matrix.shape[0] > 0 and matrix.shape[1] >0):
#        print(np.linalg.norm(matrix,2))
#        matrix = np.linalg.qr(matrix, mode='r')
#        print(ind, matrix.shape)
#        print(ind,np.linalg.norm(matrix,2))
        ss = np.sum(abs(matrix)**2,axis=0)
        smax = max(ss)
#        print(ind,np.sqrt(smax))
#
# This part of the code needs to be reverted back
#
    k, idx, proj = interp_decomp(matrix, tau)
#    print(ind,k,index_lvl.shape[0])
    if k == index_lvl.shape[0]:
        factor.abasis[ind] = np.zeros(0)
        factor.basis[ind] = dc(index_lvl)
        factor.local_abasis[ind] = np.zeros(0)
        factor.local_basis[ind] = np.arange(k)
        factor.T[ind] = np.identity(index_lvl.shape[0], dtype=pr.dtype)
    else:
        factor.abasis[ind] = index_lvl[idx[k:]]
        factor.basis[ind] = index_lvl[idx[:k]]
        factor.local_abasis[ind] =  idx[k:]
        factor.local_basis[ind] = idx[:k]
        T_1 = np.identity(index_lvl.shape[0], dtype=pr.dtype)
        if proj.T.shape[0] > 0:
            T_1[np.ix_(idx[k:],idx[:k])] = proj.T * -1
        factor.T[ind] = T_1


def build_T_wtd(ind, factor):
    wts = factor.pr.wts
    D = np.diag(wts[factor.index_lvl[ind]])
    inv_D = np.linalg.inv(D)
    factor.T_wtd[ind] = (inv_D).dot(factor.T[ind].T).dot(D)

def factorize_tail(factor):
    print("in factorize tail")
    l = factor.tail_lvl
    n0 = 0
    tree = factor.tree
    pr = factor.pr
    job = [j for j in
           range(tree.level[l], tree.level[l+1])
           if not pr.row_notransition[j]]
    rem_job = dc(job)
#    print(job)
#    print(rem_job)
    for ind in job:
#        print(ind)
        tmp_mat = factor.func(factor.basis[ind], factor.basis[ind])
        n0 = n0 + np.size(factor.basis[ind])
        sch_bl = schur_tail(ind, ind, factor, job)
        if not sch_bl is None:
            tmp_mat -= sch_bl

        if factor.symmetric_fun:
            factor.tail_L[ind] = cholesky(tmp_mat, lower=1)
        else:
            if tmp_mat.shape[0] == 0 or tmp_mat.shape[1] == 0:
                factor.tail_L[ind], factor.tail_U[ind] = np.zeros(tmp_mat.shape),np.zeros(tmp_mat.shape)
            else:
                factor.tail_L[ind], factor.tail_U[ind] = lu(tmp_mat, permute_l=True)
        rem_job.remove(ind)
        for ind_od in rem_job:
            od_tmp = factor.func(factor.basis[ind], factor.basis[ind_od])
            sch_bl = schur_tail(ind, ind_od, factor, job)
            if not sch_bl is None:
                od_tmp -= sch_bl
            factor.tail_od[ind].append(od_tmp)
            if not factor.symmetric_fun:
                od_tmp_U = factor.func(factor.basis[ind_od], factor.basis[ind])
                sch_bl = schur_tail(ind_od, ind, factor, job, row_col = 'col')
                if not sch_bl is None:
                    od_tmp_U -= sch_bl
                factor.tail_od_U[ind].append(od_tmp_U)
    return factor,n0


def build_cube_problem(func, n=15, ndim=2, block_size=28, symmetric=1, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', random_points=0, zk=None):
    count = n**ndim
    if random_points:
        position = np.random.rand(ndim,n**ndim)
    else:
        if ndim == 1:
            raise NameError('In progress')
        elif ndim == 2:
            x0, x1 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n)
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim)))
        elif ndim == 3:
            x0, x1, x2 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n, np.arange(1,n+1)/(n))
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim), x2.reshape(1,n**ndim)))

    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, tree, symmetric, verbose%2)
    return problem
def prep_next_lvl_schur(factor, lvl):
        tmp_scour_dict = {}
        tree = factor.tree
        pr = factor.pr
        level_count = len(tree.level)-2
        if lvl >= level_count-1:
            print('Error! can not compute prep_next_lvl_schur for this level!')
            raise KeyboardInterrupt
        job = [j for j in
                        range(tree.level[lvl], tree.level[lvl+1])
                        if not pr.row_notransition[j]]
        for row_i in job:
            for col_i in factor.pr.schur_list[row_i]:
                if tree.child[row_i] and tree.child[col_i]:
                    res = None
                    for ii in tree.child[row_i]:
                        for jj in tree.child[col_i]:
                            tmp = schur(ii, jj, factor, ib_row = 'b', ib_col='b')
                            if not tmp is None:
                                if res is None:
                                    res = 1
                                    tmp_scour_dict[row_i, col_i] = {}
                                tmp_scour_dict[row_i, col_i][ii,jj] = tmp
                elif not tree.child[row_i] and tree.child[col_i]:
                    res = None
                    for jj in tree.child[col_i]:
                        tmp = schur(row_i, jj, factor, ib_col='b')
                        if not tmp is None:
                            if res is None:
                                res = 1
                                tmp_scour_dict[row_i, col_i] = {}
                            tmp_scour_dict[row_i, col_i]['r', jj] = tmp
                elif tree.child[row_i] and not tree.child[col_i]:
                    res = None
                    for jj in tree.child[row_i]:
                        tmp = schur(jj, col_i, factor, ib_row='b')
                        if not tmp is None:
                            if res is None:
                                res = 1
                                tmp_scour_dict[row_i, col_i] = {}
                            tmp_scour_dict[row_i, col_i]['c', jj] = tmp
        factor.prev_lvl_schur = tmp_scour_dict
def prep_next_lvl_schur_old(factor, lvl):
    tmp_scour_dict = {}
    tree = factor.tree
    pr = factor.pr
    level_count = len(tree.level)-2
    if lvl >= level_count-1:
        print('Error! can not compute prep_next_lvl_schur for this level!')
        raise KeyboardInterrupt
    job = [j for j in
                    range(tree.level[lvl], tree.level[lvl+1])
                    if not pr.row_notransition[j]]
    two_ch = 0
    one_ch = 0
    other_ch = 0
    no_ch = 0
    two_ch_nr = 0
    one_ch_nr = 0
    other_ch_nr = 0
    no_ch_nr = 0
    schur_n_tot = 0
    avrg_nzb = 0
    for row_i in job:
        for col_i in factor.pr.schur_list[row_i]:
            schur_n_tot += 1
            if tree.child[row_i] and tree.child[col_i]:
                two_ch += 1
                res = None
                pointer_row = 0
                nzb = 0
                for ii in tree.child[row_i]:
                    pointer_col = 0
                    for jj in tree.child[col_i]:
                        tmp = schur(ii, jj, factor, ib_row = 'b', ib_col='b')
                        if not tmp is None:
                            nzb += 1
                            if res is None:
                                l_row = 0
                                l_col = 0
                                for i_tmp in tree.child[row_i]:
                                    l_row += factor.basis[i_tmp].shape[0]
                                for i_tmp in tree.child[col_i]:
                                    l_col += factor.basis[i_tmp].shape[0]
                                res = np.zeros((l_row,l_col), dtype=pr.dtype)
                            res[pointer_row:pointer_row+tmp.shape[0],pointer_col:pointer_col+tmp.shape[1]] = tmp
                        pointer_col += factor.basis[jj].shape[0]
                    pointer_row += factor.basis[ii].shape[0]
                #print(nzb, nzb/64.)
                avrg_nzb += nzb/64.
                if not res is None:
                    #print(nzb, nzb/64.)
                    two_ch_nr += 1
                    tmp_scour_dict[row_i, col_i] = res
            elif not tree.child[row_i] and tree.child[col_i]:
                one_ch += 1
                res = None
                pointer_col = 0
                for jj in tree.child[col_i]:
                    tmp = schur(row_i, jj, factor, ib_col='b')
                    if not tmp is None:
                        if res is None:
                            l_col = 0
                            for i_tmp in tree.child[col_i]:
                                l_col += factor.basis[i_tmp].shape[0]
                            res = np.zeros((factor.index_lvl[row_i].shape[0],l_col))
                        res[:,pointer_col:pointer_col+tmp.shape[1]] = tmp
                    pointer_col +=  factor.basis[jj].shape[0]
                if not res is None:
                    one_ch_nr += 1
                    tmp_scour_dict[row_i, col_i] = res
            elif tree.child[row_i] and not tree.child[col_i]:
                other_ch += 1
                res = None
                pointer_row = 0
                for jj in tree.child[row_i]:
                    tmp = schur(jj, col_i, factor, ib_row='b')
                    if not tmp is None:
                        if res is None:
                            l_row = 0
                            for i_tmp in tree.child[row_i]:
                                l_row += factor.basis[i_tmp].shape[0]
                            res = np.zeros((l_row,factor.index_lvl[col_i].shape[0]))
                        res[pointer_row:pointer_row+tmp.shape[0],:] = tmp
                    pointer_row +=  factor.basis[jj].shape[0]
                if not res is None:
                    other_ch_nr += 1
                    tmp_scour_dict[row_i, col_i] = res
            else:
                no_ch += 1
    print(f'lvl: {lvl}')
    if two_ch_nr != 0:
    	print(f'avrg_nzb: {avrg_nzb/two_ch_nr}')
    #print(f"dict: {sys.getsizeof(factor.prev_lvl_scour)}")
    print(f'tot: {schur_n_tot}, 2_ch: {two_ch}, 1_ch: {one_ch}, 1_ch:{other_ch}, 0_ch: {no_ch}')
    print(f'2_ch_nr: {two_ch_nr}, 1_ch_nr: {one_ch_nr}, 1_ch_nr:{other_ch_nr}, 0_ch_nr: {0}')
    factor.prev_lvl_scour = tmp_scour_dict
    a = dc(factor.prev_lvl_scour)
def build_matrix_from_dict(factor, row_i, col_i):
    pr = factor.pr
    tree = factor.tree
    if tree.child[row_i] and tree.child[col_i]:
        l_row = 0
        l_col = 0
        for i_tmp in tree.child[row_i]:
            l_row += factor.basis[i_tmp].shape[0]
        for i_tmp in tree.child[col_i]:
            l_col += factor.basis[i_tmp].shape[0]
        res = np.zeros((l_row, l_col), dtype=pr.dtype)
        pointer_row = 0
        for ii in tree.child[row_i]:
            pointer_col = 0
            for jj in tree.child[col_i]:
                if (ii,jj) in factor.prev_lvl_schur[row_i, col_i]:
                    tmp = factor.prev_lvl_schur[row_i, col_i][ii,jj]
                    res[pointer_row:pointer_row+tmp.shape[0],pointer_col:pointer_col+tmp.shape[1]] = tmp
                pointer_col += factor.basis[jj].shape[0]
            pointer_row += factor.basis[ii].shape[0]
        return res
    elif not tree.child[row_i] and tree.child[col_i]:
        l_col = 0
        for i_tmp in tree.child[col_i]:
            l_col += factor.basis[i_tmp].shape[0]
        res = np.zeros((factor.index_lvl[row_i].shape[0],l_col))
        pointer_col = 0
        for jj in tree.child[col_i]:
            if ('r', jj) in factor.prev_lvl_schur[row_i, col_i]:
                tmp = factor.prev_lvl_schur[row_i, col_i]['r',jj]
                res[:,pointer_col:pointer_col+tmp.shape[1]] = tmp
            pointer_col +=  factor.basis[jj].shape[0]
        return res
    elif tree.child[row_i] and not tree.child[col_i]:
        l_row = 0
        for i_tmp in tree.child[row_i]:
            l_row += factor.basis[i_tmp].shape[0]
        res = np.zeros((l_row,factor.index_lvl[col_i].shape[0]))
        pointer_row = 0
        for jj in tree.child[row_i]:
            if ('r', jj) in factor.prev_lvl_schur[row_i, col_i]:
                tmp =  factor.prev_lvl_schur[row_i, col_i]['c',jj]
                res[pointer_row:pointer_row+tmp.shape[0],:] = tmp
            pointer_row +=  factor.basis[jj].shape[0]
        return res
    else:
        raise NameError('Error!')
def schur(row_i, col_i, factor, ib_row = 'i', ib_col = 'i', ban_list=[]):
    if not col_i in factor.pr.schur_list[row_i] and not row_i in factor.pr.schur_list[col_i]:
        return None
    tree = factor.tree
    res = level_schour(row_i, col_i, factor, ib_row = ib_row, ib_col = ib_col, ban_list=ban_list)
    #if (row_i, col_i) in factor.prev_lvl_scour:
    #    res = add_res_and_tmp(res, factor.prev_lvl_scour[row_i, col_i], factor=factor, i_row=row_i, i_col=col_i)
    if (row_i, col_i) in factor.prev_lvl_schur:
            bl = build_matrix_from_dict(factor, row_i, col_i)
            res = add_res_and_tmp(res, bl, factor=factor, i_row=row_i, i_col=col_i)
            del(bl)
    if  ib_row == 'b' and not res is None:
        if res.shape[0] != factor.basis[row_i].shape[0]:
            res = res[factor.local_basis[row_i]]
    if ib_col == 'b' and not res is None:
        if res.shape[1] != factor.basis[col_i].shape[0]:
            res = res[:, factor.local_basis[col_i]]
    return res
def build_problem_old(block_size=26, n=15, ndim = 2, func = test_funcs.log_distance, point_based_tree=1, close_r = '1box', num_child_tree='hyper', random_points=1):
    iters = 2
    onfly = 1
    symmetric_tree = 1
    verbose = 0
    random_init = 2
    if (point_based_tree or num_child_tree == 2) and close_r == '1box':
        print("!!'1box' work only with space_based_tree num_child_tree = 'hyper'! \n !! close_r chanjed to 1.")
        close_r = 1.
    pr = build_cube_problem(func, n=n, ndim=ndim, block_size=block_size,
                              symmetric=symmetric_tree, verbose=verbose,point_based_tree=point_based_tree,
                              close_r=close_r,num_child_tree = num_child_tree,random_points = random_points)
    # pr.dtype = float
    if not pr.symmetric:
        raise NameError('Different row and column trees are not supported. Set symmetric=1')
    return pr
def build_sphere_problem(func, n=15, block_size=28, symmetric=1, point_based_tree = True, close_r='1box',num_child_tree = 'hyper', random_points=0, zk=None):
    count = n
    position = fibonacci_sphere(n, 1, [0,0,0])
    data = Data(3, count, position, close_r=close_r)
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, tree, symmetric, 0)
    return problem
def build_sphere_problem_double(func, n=15, ndim=3, block_size=28, symmetric=1, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', zk=None):
    if ndim == 2:
        raise NameError(f'ndim = 2 is in progress')
    r = 1.
    c = np.zeros(ndim)
    position = fibonacci_sphere(n, r, c)
    count = position.shape[1]
    print (f'Number of points is {count}')

    if (np.unique(position, axis=1)).shape[1] != count:
        raise NameError ('Duplocated points!')

    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, tree, symmetric, verbose%2)
    return problem
def build_problem_from_file(func, block_size=28, symmetric=1, verbose=1, point_based_tree=True, close_r='1box', num_child_tree='hyper',file=None, eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0,csc_fun=0, ifwrite=0):
    x = np.loadtxt(file)
    ndim = 3
    order = int(x[0]) # ndim
    npatches = int(x[1])
    npols = int((order+1)*(order+2)/2)
    n = npatches*npols

    zpars = np.array([zk,alpha,beta],dtype=complex)


    # setup geometry in the correct format
    norders = order*np.ones(npatches)
    iptype = np.ones(npatches)
    srcvals = x[2::].reshape(12,n).copy(order='F')
    x0 = srcvals[0]
    x1 = srcvals[1]
    x2 = srcvals[2]
    position = np.vstack((x0.reshape(1,n),x1.reshape(1,n), x2.reshape(1,n)))

    data = Data(ndim, n, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    elif func == test_funcs.double_layer:
        data.k = zk
        data.norms = srcvals[9:12]
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, tree, symmetric, verbose%2)
    problem.csc_fun = csc_fun
    problem.ndim = ndim
    problem.file = file
    problem.npatches = npatches
    problem.srcvals = srcvals
    problem.eps = eps
    problem.order = order
    problem.npatches = npatches

    ixyzs = np.arange(npatches+1)*npols+1
    srccoefs = h3.surf_vals_to_coefs(norders,ixyzs,iptype, srcvals[0:9,:])
    wts = h3.get_qwts(norders,ixyzs,iptype,srcvals)
    problem.wts = wts
    if csc_fun:
        nifds,nrfds,nzfds = h3.helm_comb_dir_fds_csc_mem(norders,ixyzs,iptype,srccoefs,srcvals,
                                                     eps,zpars)

        ifds,rfds,zfds = h3.helm_comb_dir_fds_csc_init(norders,ixyzs,iptype,srccoefs,srcvals,eps,
                                               zpars,nifds,nrfds,nzfds)
    else:
        nifds,nrfds,nzfds = h3.helm_comb_dir_fds_block_mem(norders,ixyzs,iptype,srccoefs,srcvals,eps,zpars,ifwrite)
        ifds,zfds = h3.helm_comb_dir_fds_block_init(norders,ixyzs,iptype,srccoefs,srcvals,eps,zpars,nifds,nzfds)
    problem.ixyzs = ixyzs
    problem.srccoefs = srccoefs
    problem.ifds = ifds
    if csc_fun:
        problem.rfds = rfds
    problem.zfds = zfds
    problem.zpars = zpars
    problem.beta = beta
    problem.norders = np.ones(npatches)*order
    problem.npols = npols
    problem.npts = npols*npatches
    problem.iptype = np.ones(npatches)
    problem.ifwrite = ifwrite
    return problem
def build_problem_wtorus(func, block_size=28, symmetric=1, verbose=1, point_based_tree=True, close_r='1.1', num_child_tree='hyper', eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0, csc_fun=0, ifwrite=0, nu=10, order=3):
    radii = np.array([1.0,2.0,0.25])
    scales = np.array([1.2,1.0,1.7])
    nosc = 5

    npatches = 2*nu*nu
    npols = int((order+1)*(order+2)/2)
    npts = int(npatches*npols)

    norders,ixyzs,iptype,srcvals,srccoefs,wts = h3.get_wtorus_geom(radii,scales,
     nosc,nu,nu,npatches,order,npts)

    # x = np.loadtxt(file)
    ndim = 3
    # order = int(x[0]) # ndim
    # npatches = int(x[1])
    # npols = int((order+1)*(order+2)/2)
    n = npatches*npols

    zpars = np.array([zk,alpha,beta],dtype=complex)


    # setup geometry in the correct format
    # norders = order*np.ones(npatches)
    iptype = np.ones(npatches)
    # srcvals = x[2::].reshape(12,n).copy(order='F')
    x0 = srcvals[0]
    x1 = srcvals[1]
    x2 = srcvals[2]
    position = np.vstack((x0.reshape(1,n),x1.reshape(1,n), x2.reshape(1,n)))

    data = Data(ndim, n, position, close_r=close_r)
    data.k = zk
    data.norms = srcvals[9:12]
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, tree, symmetric, verbose%2)
    problem.csc_fun = csc_fun
    problem.ndim = ndim
    problem.file = None
    problem.npatches = npatches
    problem.srcvals = srcvals
    problem.eps = eps
    problem.order = order
    problem.npatches = npatches

    ixyzs = np.arange(npatches+1)*npols+1
    srccoefs = h3.surf_vals_to_coefs(norders,ixyzs,iptype, srcvals[0:9,:])
    wts = h3.get_qwts(norders,ixyzs,iptype,srcvals)
    problem.wts = wts
    t0 = time()

    if csc_fun:
        nifds,nrfds,nzfds = h3.helm_comb_dir_fds_csc_mem(norders,ixyzs,iptype,srccoefs,srcvals,
                                                     eps,zpars)

        ifds,rfds,zfds = h3.helm_comb_dir_fds_csc_init(norders,ixyzs,iptype,srccoefs,srcvals,eps,
                                               zpars,nifds,nrfds,nzfds)
    else:
#        print("eps=",eps)
        nifds,nrfds,nzfds = h3.helm_comb_dir_fds_block_mem(norders,ixyzs,iptype,srccoefs,srcvals,eps,zpars,ifwrite)
#        print("nifds=",nifds)
#        print("nzfds=",nzfds)
        ifds,zfds = h3.helm_comb_dir_fds_block_init(norders,ixyzs,iptype,srccoefs,srcvals,eps,zpars,nifds,nzfds)
    t1 = time()
    print("time taken for quadrature correction="+str(t1-t0))
    problem.ixyzs = ixyzs
    problem.srccoefs = srccoefs
    problem.ifds = ifds
    if csc_fun:
        problem.rfds = rfds
    problem.zfds = zfds
    problem.zpars = zpars
    problem.beta = beta
    problem.norders = np.ones(npatches)*order
    problem.npols = npols
    problem.npts = npols*npatches
    problem.iptype = np.ones(npatches)
    problem.ifwrite = ifwrite
    return problem

def build_problem(geom_type='qube',block_size=26, n=15, ndim = 2, func = test_funcs.log_distance, point_based_tree=0, close_r = 1., num_child_tree='hyper', random_points=1, file = None, eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0, wtd_T=0, ibuild_matrix=1,add_up_level_close=0,half_sym=0,csc_fun=0,q_fun=0,ifwrite=0, nu=10, order=10):
    iters = 2
    onfly = 1
    symmetric_tree = 1
    verbose = 0
    random_init = 2
    if point_based_tree and close_r == '1box':
        print("!!'1box' does not work with point_based_tree = 1, close_r chanjed to 1.")
        close_r = 1.
    if geom_type == 'qube':
        pr = build_cube_problem(func, n=n, ndim=ndim, block_size=block_size,
                                  symmetric=symmetric_tree, verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,random_points = random_points,zk=zk)
    if geom_type == 'sphere_test':
        pr = build_sphere_problem(func, n=n, block_size=block_size,
                                  point_based_tree=point_based_tree, close_r=close_r,
                                  num_child_tree=num_child_tree)
    elif geom_type == 'sphere_double':
        pr = build_sphere_problem_double(func, n=n, ndim=ndim, block_size=block_size,
                                  symmetric=symmetric_tree, verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,zk=zk)
    elif geom_type == 'from_file':
        if file is None:
            raise NameError(f"Geometry type '{geom_type}' should have nonepty file!")
        pr = build_problem_from_file(func, block_size=block_size,symmetric=symmetric_tree, verbose=verbose,
                                     point_based_tree=point_based_tree, close_r=close_r,
                                     num_child_tree=num_child_tree, file=file,eps=eps, zk=zk, alpha=alpha,
                                     beta=beta,csc_fun=csc_fun,ifwrite=ifwrite)
    elif geom_type == 'wtorus':
        pr = build_problem_wtorus(func, block_size=block_size,symmetric=symmetric_tree, verbose=verbose,
                                  point_based_tree=point_based_tree, close_r=close_r,
                                  num_child_tree=num_child_tree, eps=eps, zk=zk, alpha=alpha,
                                  beta=beta,csc_fun=csc_fun,ifwrite=ifwrite, nu=nu, order=order)
    else:
        raise NameError (f"Geometry type '{geom_type}' is not supported. Try 'qube/sphere/from_file/wtorus'")
    if not pr.symmetric:
        raise NameError('Different row and column trees are not supported. Set symmetric=1')
    pr.add_up_level_close = add_up_level_close
    if add_up_level_close:
        print('Warning! up-level close is not well-tested!')
    pr.csc_fun = csc_fun
    pr.wtd_T = wtd_T
    pr.ibuild_matrix = ibuild_matrix
    pr.half_sym = half_sym
    pr.q_fun = q_fun
    tree = pr.row_tree
    level_count = len(tree.level) - 2
    for i in range(level_count-1, -1, -1):
        job = [j for j in
                        range(tree.level[i], tree.level[i+1])]
        exist_no_trans_t = False
        exist_no_trans_f = False
        for ind in job:
            if pr.row_notransition[ind]:
                exist_no_trans_t = True
            else:
                exist_no_trans_f = True
        if exist_no_trans_t and exist_no_trans_f:
            print ('lvl', i, '+-')
            pr.tail_lvl = i
            for ind in job:
                pr.row_notransition[ind] = False
        elif exist_no_trans_t:
            pr.tail_lvl = i+1
            break
#    print(tree.child[0:10])
    return pr
def fmm_lu_old(pr, proxy_p=1., proxy_r=1., symmetric_fun = 0, tau=1e-9):
    csc_fun = pr.csc_fun
    tree = pr.row_tree
    level_count = len(tree.level) - 2
    pr.multilevel_close()
    pr.schur_precompute()
    factor = Factor(pr, proxy_p=proxy_p, proxy_r=proxy_r,
                    symmetric_fun = symmetric_fun)
    ind_l = []
    factor.tail_lvl = pr.tail_lvl
    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
                    range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.init_index_lvl(ind)

    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
               range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.upd_index_lvl(ind)
        if csc_fun:
            factor.csc = pr.compute_csc(factor, i)
        ind_l += job
        factor = factorize_lvl(factor, job, tau=tau, l=i)
    if csc_fun:
        # factor.csc_fun = 0
        factor.csc = pr.compute_csc_tail(factor)
    factor = factorize_tail(factor)
    return factor
def fibonacci_sphere(numpts, k, c):
    ga = (3 - np.sqrt(5)) * np.pi # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(numpts)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1/numpts-1, 1-1/numpts, numpts)

    # a list of the radii at each height step of the unit circle
    radius = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    x = radius * k * np.cos(theta) + c[0]
    y = radius * k * np.sin(theta) + c[1]
    z = z * k + c[2]
    return np.array((x, y, z))
# FMM-LU solver
def build_csc(pr, row_ind, col_ptr):
#     npols = int((pr.ndim+1)*(pr.ndim+2)/2)
#     npts = pr.npatches*npols
    norders = pr.ndim*np.ones(pr.npatches)
    iptype = np.ones(pr.npatches)
    eps = pr.eps
    wts = h3.get_qwts(norders,pr.ixyzs,iptype,pr.srcvals)
    col_ptr = col_ptr + 1
    row_ind = row_ind + 1
    return h3.helm_comb_dir_fds_csc_matgen(norders,pr.ixyzs,iptype,pr.srccoefs,pr.srcvals,
                                    eps,pr.zpars,pr.ifds,pr.rfds,pr.zfds,col_ptr,row_ind)
def compute_csc(factor, i, coef):
    pr = factor.pr
    tree = pr.row_tree
#     tmp_lil = identity(factor.n, dtype=bool, format='lil')
    tmp_lil = lil((factor.n,factor.n))
    job = [j for j in
           range(tree.level[i], tree.level[i+1])]
    for ind in job:
        for cl in pr.lvl_close[ind]:
            tmp_lil[np.ix_(factor.index_lvl[ind], factor.index_lvl[cl])] = np.ones((factor.index_lvl[ind].shape[0], factor.index_lvl[cl].shape[0]))
    print (f'level: {i}, tmp_lil.nnz:{tmp_lil.nnz}, factor.n**2:{factor.n**2}, factor.n:{factor.n}, tmp_lil.nnz/factor.n:{tmp_lil.nnz/factor.n}')
#     factor.full_lil += tmp_lil
    tmp_csc = csc(tmp_lil)
#     print (tmp_csc.shape)
    row = tmp_csc.indices
    col = tmp_csc.indptr
#     print (tmp_csc.data.shape)
#     print ('---before csc_log_fun')
    data = build_csc(pr, row, col)
#     print ('---after csc_log_fun')
#     print(data.shape)
    npols = int((pr.ndim+1)*(pr.ndim+2)/2)
    npts = pr.npatches*npols
    addition = coef*2*np.pi*sps.identity(factor.n,format='csc')*pr.zpars[2]
#     print (2*np.pi*pr.zpars[2])
    csc_mat = csc((data, row, col),shape=(factor.n,factor.n))

    print (addition.shape, csc_mat.shape, factor.n)
    return csc_mat + addition
def compute_csc_tail(factor, coef):
    print(' compute_csc_tail pass 0')
    pr = factor.pr
    tree = pr.row_tree
    tmp_lil = lil((factor.n,factor.n))
    i = factor.tail_lvl

    job = [j for j in
           range(tree.level[i], tree.level[i+1])]
    for ind in job:
        for cl in job:
            tmp_lil[np.ix_(factor.basis[ind], factor.basis[cl])] = np.ones((factor.basis[ind].shape[0], factor.basis[cl].shape[0]))

#     print(' compute_csc_tail pass 1')
    print (f'tmp_lil.nnz:{tmp_lil.nnz}, factor.n**2:{factor.n**2}, factor.n:{factor.n}, tmp_lil.nnz/factor.n:{tmp_lil.nnz/factor.n}')
#     factor.full_lil += tmp_lil
    tmp_csc = csc(tmp_lil)

#     print(' compute_csc_tail pass 2')
    row = tmp_csc.indices
    col = tmp_csc.indptr
#     print(' compute_csc_tail pass 3')
    data = build_csc(pr, row, col)
    npols = int((pr.ndim+1)*(pr.ndim+2)/2)
    npts = pr.npatches*npols
    addition = coef*2*np.pi*sps.identity(factor.n,format='csc')*pr.zpars[2]
    csc_mat =  csc((data, row, col),shape=(factor.n,factor.n))
    return csc_mat + addition
@profile
def fmm_lu(pr, proxy_p=(1, 100), proxy_r=1., symmetric_fun = 0, tau=1e-9, out_f=0):
    coef = pr.coef
    if out_f:
        with open(out_f, 'a') as f:
             f.write('----- befor factor level: csc_fun: {pr.csc_fun}\n')
    else:
        print (f'----- befor factor level: csc_fun: {pr.csc_fun}')
    tree = pr.row_tree
    level_count = len(tree.level) - 2
    pr.multilevel_close()
    pr.schur_precompute()
    t0 = time()
    factor = Factor(pr, proxy_p=proxy_p, proxy_r=proxy_r,
                    symmetric_fun = symmetric_fun)
    ind_l = []
    factor.tail_lvl = pr.tail_lvl
    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
                    range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.init_index_lvl(ind)
    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
               range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.upd_index_lvl(ind)
        if pr.csc_fun:
            factor.csc = compute_csc(factor, i, coef)
            # print (f'------lvl {i}, CSC is done, start factor, csc[0,0]: {factor.csc[0,0]}')
        ind_l += job
#        print('level=',i)
        factor = factorize_lvl(factor, job, tau=tau, l=i)
#        if out_f:
#            with open(out_f, 'a') as f:
#                 f.write(f'lvl {i} done!\n')
#        else:
#            print(f'lvl {i} done!')
#    if out_f:
#        with open(out_f, 'a') as f:
#             f.write(f'----- after factor level, t={time()-t0}\n')
#    else:
#        print (f'----- after factor level, t={time()-t0}')
    t0 = time()
    if pr.csc_fun:
        factor.csc = compute_csc_tail(factor, coef)
    factor,n0 = factorize_tail(factor)
#    if out_f:
#        with open(out_f, 'a') as f:
#             f.write(f'----- after factor tail: t={time() - t0}sec\n')
#    else:
#        print (f'----- after factor tail, t={time()-t0}')
    return factor,n0
def fmm_lu_no_print(pr, proxy_p=(1,100), proxy_r=1., symmetric_fun = 0, tau=1e-9):
    #print (f'----- befor factor level: csc_fun: {pr.csc_fun}')
    coef = pr.coef
    tree = pr.row_tree
    level_count = len(tree.level) - 2
    pr.multilevel_close()
    pr.schur_precompute()
    t0 = time()
    factor = Factor(pr, proxy_p=proxy_p, proxy_r=proxy_r,
                    symmetric_fun = symmetric_fun)
    ind_l = []
    factor.tail_lvl = pr.tail_lvl
    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
                    range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.init_index_lvl(ind)
    for i in range(level_count-1, factor.tail_lvl-1, -1):
        job = [j for j in
               range(tree.level[i], tree.level[i+1])]
        for ind in job:
            factor.upd_index_lvl(ind)
        if pr.csc_fun:
            factor.csc = compute_csc(factor, i, coef)
            # print (f'------lvl {i}, CSC is done, start factor, csc[0,0]: {factor.csc[0,0]}')
        ind_l += job
        factor = factorize_lvl(factor, job, tau=tau, l=i)
    #print (f'----- after factor level, t={time()-t0}')
    t0 = time()
    if pr.csc_fun:
        factor.csc = compute_csc_tail(factor, coef)
    factor = factorize_tail(factor)
    #print (f'----- after factor tail, t={time()-t0}')
    return factor
def factor_csc_to_full(factor):
    col_ptr = np.arange(factor.n+1)*factor.n
    row_ind = np.tile(np.arange(factor.n),factor.n)
    data = build_csc(factor.pr, row_ind, col_ptr)
    factor.csc = csc((data, row_ind, col_ptr))
def fmm_lu_solve(pr,eps,rhst,proxy_p,proxy_r,verbose=0):
    t0 = time()
    if verbose:
      factor,n0 = fmm_lu(pr, tau = eps, proxy_p=proxy_p, proxy_r=proxy_r, symmetric_fun=0)
    else:
      factor = fmm_lu_no_print(pr, tau = eps, proxy_p=proxy_p, proxy_r=proxy_r, symmetric_fun=0)
    t1 = time()
    # print(f'Fact time: {t1 - t0}')
    ans = factor.solve(rhst)
    # print(f'Sol time: {time() - t1}')
    return ans, factor, t1 - t0, time() - t1,n0
