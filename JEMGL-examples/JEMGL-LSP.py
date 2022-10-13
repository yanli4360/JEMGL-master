# -*- coding: utf-8 -*-
"""
We investigate the JEMGL algorithm with Laplacian shrinkage penalty.
"""


import numpy as np
from JEMGL.Helper.Data_generation import  sample_covariance_matrix,group_modular_graph
from JEMGL.Helper.Utils import rho_n_grid,generate_Q,relative_error
from JEMGL.Algorithm.ADMM_solver import ADMM_JEMGL

p = 30
K = 3
N_train = 300
M = 3
penalty = 'LSP'

# graph construction
Sigma, Laplacian = group_modular_graph(p, K, M)
S_train, sample_train = sample_covariance_matrix(Sigma, N_train)
rho_n_list=rho_n_grid(num=20)
grid=rho_n_list.shape[0]
A_0=np.array([])
B_0=np.array([])
E_0 = np.array([])
F_0 = np.array([])
Lset=[]
RE=np.zeros(grid)
for i in np.arange(grid):
    rho_n=rho_n_list[i]
    rho_1=1
    Q=generate_Q(S_train,rho_n)
    sol_NL,info=ADMM_JEMGL(Q, penalty,rho_n,rho_1, A_0=A_0, B_0=B_0, E_0=E_0, F_0=F_0, tol=1e-5, rtol=1e-4, update_rho=True, rho=1., max_iter=1000, verbose=False,measure=False)
    Laplacian_sol=sol_NL['L']
    RE[i]=relative_error(Laplacian_sol,Laplacian)
    Lset.append(Laplacian_sol)
idx=np.max(np.unravel_index(np.nanargmin(RE),RE.shape))
hat_L=Lset[idx]




