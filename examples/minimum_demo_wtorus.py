import sys
sys.path.insert(0,'..')
import numpy as np
from numpy.random import default_rng
import math
from time import time
from functions import test_funcs
from FMM_LU import FMM_LU as fmm_lu
from problem_tools import problem

import fmm3dbie as h3
import fmm3dpy as fmm3d

verbose = 1
np.random.seed(0)
bs = 64
func = test_funcs.double_layer
close_r = 1.1
num_child_tree = 'hyper'
point_based_tree = 0
eps = 0.51e-4
zk = 1.0 + 1j*0
alpha = 0.0
beta = 1.0
proxy_p = (2, 200)
proxy_r = 1.
csc_fun = 0
symmetric_fun = 0
half_sym = 1
t0 = time()
order = 3
nu = 20
q_fun = 1

pr = fmm_lu.build_problem(geom_type='wtorus', block_size=bs, func=func,
                          point_based_tree=point_based_tree, close_r=close_r,
                          num_child_tree=num_child_tree, eps=eps,
                          zk=zk, alpha=alpha, beta=beta, wtd_T=1, half_sym=half_sym, 
                          csc_fun=csc_fun, q_fun=q_fun, nu=nu, order=order)                     

pr.coef = coef = -1
print(f'n = {pr.shape[0]}\n')
print(f'problem-build time: {time() - t0}\n')

# FMM-LU solver

xyz_out = np.array([[31.17,-0.03,3.15],[6.13,-4.1,22.2]]).transpose()
xyz_in = np.array([[0.11,-2.13,0.05],[0.13,2.1,-0.01]]).transpose()
#coef = -1

# comparizon to FMM

c = np.array([1 + 1j*0,1+1.1j])
out = fmm3d.h3ddir(zk=zk, sources=xyz_out, targets=pr.srcvals[0:3,:], charges=c, pgt=1)
rhs = out.pottarg

sigma, factor, tf, ts = fmm_lu.fmm_lu_solve(pr, eps, rhs, proxy_p, proxy_r, verbose=verbose)

ntarg = np.shape(xyz_in)[1]

ipatch_id = coef*np.ones(2)
uvs_targ = np.zeros((2,ntarg))

norders = pr.order*np.ones(pr.npatches)
iptype = np.ones(pr.npatches)

pot_comp = h3.lpcomp_helm_comb_dir(norders, pr.ixyzs, iptype, pr.srccoefs, pr.srcvals,
   xyz_in, ipatch_id, uvs_targ, eps, pr.zpars, sigma)

out = fmm3d.h3ddir(zk=zk, sources=xyz_out, targets=xyz_in, charges=c,pgt=1)
pot_ex = out.pottarg
erra = np.linalg.norm(pot_ex-pot_comp)
print(f'Fact time: {tf}\n')
print(f'Sol time: {ts}\n')
print(f"error in solution = {erra}\n")

if verbose:
  tree = pr.row_tree
  level_count = len(tree.level) - 2
  print(f'Compression on levels. \nlevel_count: {level_count-2}')
  for l in range(level_count-1, factor.tail_lvl-1, -1):
      job = [j for j in
             range(tree.level[l], tree.level[l+1])]
      print(f'Level: {l}')
      proc = 0
      mean_b = 0
      mean_ind = 0
      nindl = 0
      mean_other_lvl_close = 0
      for i in job:
          mean_other_lvl_close +=len(pr.other_lvl_close[i])
          if factor.index_lvl[i].shape[0] != 0:
              proc += factor.basis[i].shape[0]/factor.index_lvl[i].shape[0]*100
              nindl += 1
              mean_b += factor.basis[i].shape[0]
              mean_ind += factor.index_lvl[i].shape[0]
      print(f'  Mean other lvl close: {mean_other_lvl_close/nindl}')
      print(f'  Mean compression: {proc/nindl:.2f}%, mean basis: {mean_b/nindl:.2f}, mean index: {mean_ind/nindl:.2f}')

