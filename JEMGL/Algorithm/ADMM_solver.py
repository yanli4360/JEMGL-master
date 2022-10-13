# -*- coding: utf-8 -*-
"""
The ADMM-based Algorithm for Joint estimation of Multiple Graph Laplacians
"""

import numpy as np
import time


from JEMGL.Helper.Utils import orthogonal_complement_of_one,difference_operator,laplacian_shrinkage_operator
from JEMGL.Helper.Utils import get_K_identity,phimul,prox_p,prox_m,obj_f,get_K_semidef_matrix,Prox_L

def ADMM_JEMGL(Q,penalty,rho_n,rho_1,A_0=np.array([]),B_0=np.array([]),E_0=np.array([]),F_0=np.array([]),tol=1e-5, rtol=1e-4, update_rho=True,rho=1., max_iter=1000, verbose=False,measure=False):
    assert Q.shape[1] == Q.shape[2]
    assert penalty in ['GGL', 'TVGL','LSP']
    (K, p, p) = Q.shape
    #initiailization
    P=orthogonal_complement_of_one(p)
    if penalty in ('GGL'):
        J=np.eye(K)
    elif penalty in ('TVGL'):
        J=difference_operator(K)
    else:
        J=laplacian_shrinkage_operator(K)
    J_const=np.linalg.pinv(np.eye(K)+J.T@J)
    if len(A_0)==0:
        A_0=get_K_semidef_matrix(K,p)
    if len(B_0)==0:
        B_0=get_K_semidef_matrix(K,p)
    if len(E_0) == 0:
        E_0 = np.zeros((K,p,p))
    if len(F_0) == 0:
        F_0 = np.zeros((K, p, p))

    runtime=np.zeros(max_iter)
    residual=np.zeros(max_iter)
    objective=np.zeros(max_iter)
    status=''


    Xi_m=get_K_identity(K,p-1)
    Laplacian_m=np.zeros((K,p,p))
    Laplacian_o=np.zeros((K,p,p))
    for k in np.arange(K):
        Laplacian_m[k,:,:]=P@Xi_m[k,:,:]@P.T
    A_m=A_0.copy()
    B_m=B_0.copy()
    E_m=E_0.copy()
    F_m=F_0.copy()
    R_m=np.zeros((K,p-1,p-1))

    if verbose:
        print("------------ADMM Algorithm for JEMGL----------------")
        hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
        out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
        print(hdr_fmt % ("iter_m", "r_m", "s_m", "eps_pri", "eps_dual"))

    for iter_m in np.arange(max_iter):
        if measure:
            start=time.time()
        B_m_1=B_m.copy()

        W_m=Q+E_m-rho*B_m
        for k in np.arange(K):
            R_m[k,:,:]=(1/rho)*P.T@W_m[k,:,:]@P
            eigD_k,eigV_k=np.linalg.eigh(R_m[k,:,:])
            Xi_m[k,:,:]=phimul(beta=1/rho,D=eigD_k,V=eigV_k)
            Laplacian_m[k,:,:]=P@Xi_m[k,:,:]@P.T
            Laplacian_o[k,:,:]=Prox_L(Laplacian_m[k,:,:])

        for i in np.arange(p):
            for j in np.arange(p):
                b_m = B_m[:, i,j]
                f_m=F_m[:,i,j]
                A_m[:, i,j] = prox_p(J, b_m, f_m, beta=rho_n * rho_1 / rho)
        for i in np.arange(p):
            for j in np.arange(p):
                if i==j:
                    e_m=E_m[:,i,j]
                    l_m=Laplacian_m[:,i,j]
                    B_m[:,i,j]=np.maximum(1/rho*e_m+l_m,0)
                else:
                    e_m=E_m[:,i,j]
                    f_m=F_m[:,i,j]
                    l_m=Laplacian_m[:,i,j]
                    a_m=A_m[:,i,j]
                    B_m[:,i,j]=prox_m(J,J_const,e_m,f_m,l_m,a_m,rho)

        for k in range(K):
            E_m[k,:,:]+=rho*(Laplacian_m[k,:,:]-B_m[k,:,:])

        for i in range(p):
            for j in range(p):
                F_m[:,i,j]+=rho*J@A_m[:,i,j]-rho*J@B_m[:,i,j]

        if measure:
            end=time.time()
            runtime[iter_m]=end-start
            objective[iter_m]=obj_f(Xi_m,Q,P,J,A_m,rho_n,rho_1)

        r_m,s_m,e_pri,e_dual=Stopping_criterion(Laplacian_m,B_m,B_m_1,A_m,E_m,F_m,Q,rho,tol,rtol)

        if update_rho:
            if r_m>= 10*s_m:
                rho_new=2*rho
            elif s_m>=10*r_m:
                rho_new=0.5*rho
            else:
                rho_new=1.*rho

            E_m=(rho/rho_new)*E_m
            F_m=(rho/rho_new)*F_m
            rho=rho_new

        residual[iter_m]=max(r_m,s_m)

        if verbose:
            print(out_fmt % (iter_m, r_m, s_m, e_pri, e_dual))

        if(r_m<=e_pri) and (s_m<=e_dual):
            status='optimal'
            break
    if status!= 'optimal':
        if(r_m<=e_pri):
            status='primal optimal'
        elif (s_m<=e_dual):
            status='dual optimal'
        else:
            status='max iterations reached'
    print(f"ADMM terminated after {iter_m+1} iterations with status: {status}.")
    sol_NL={'L':Laplacian_o}


    if measure:
        info= {'status': status , 'runtime': runtime[:iter_m+1], 'residual': residual[:iter_m+1], 'objective': objective[:iter_m+1]}
    else:
        info={'status': status}

    return sol_NL,info



def Stopping_criterion(Laplacian, B,B_m_1,A, E,F, Q, rho, eps_abs, eps_rel):
    (K, p, p) = Q.shape

    dim = K * ((p ** 2 + p) / 2)
    e_pri = dim * eps_abs + eps_rel * np.maximum(np.linalg.norm(Laplacian), np.linalg.norm(B))
    e_dual = dim * eps_abs + eps_rel * rho * np.maximum(np.linalg.norm(E),np.linalg.norm(F))

    r1 = np.linalg.norm(Laplacian-B)
    r2 = np.linalg.norm(A-B)
    r=r1+r2
    s = rho * np.linalg.norm(B - B_m_1)
    return r, s, e_pri, e_dual




