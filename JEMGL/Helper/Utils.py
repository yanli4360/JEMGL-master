# -*- coding: utf-8 -*-
import numpy as np


def get_K_identity(K, p):
    res = np.zeros((K, p, p))
    for k in np.arange(K):
        res[k, :, :] = np.eye(p)

    return res

def get_K_semidefmatrix(K, P):
    p=P.shape[0]
    res = np.zeros((K, p, p))
    for k in np.arange(K):
        res[k, :, :] = np.matmul(P,P.T)
        D=np.linalg.eigvalsh(res[k,:,:])
        if D.min()<1e-8:
            res[k,:,:]+=0.001*np.eye(p)
    return res


def get_K_semidef_matrix(K,p):
    res=np.zeros((K,p,p))
    for k in np.arange(K):
        res[k,:,:]=2*np.eye(p)-np.ones((p,p))
    return res

def laplacian_shrinkage_operator(K):
    J_tilde=np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if abs(i-j)>1:
                J_tilde[i][j]=-0.1
            if abs(i-j)==1:
                J_tilde[i][j]=-1
    for i in range(K):
        J_tilde[i][i]=abs(sum(J_tilde[i,:]))
    D = np.linalg.eigvalsh(J_tilde)
    if D.min() < 1e-8:
        J_tilde += (0.001 + abs(D.min())) * np.eye(K)
    J=np.linalg.cholesky(J_tilde)
    return J

def difference_operator(K):
    J=np.zeros((K,K))
    for i in np.arange(1,K):
        J[i][i-1]=-1
        J[i][i]=1
    return J

def orthogonal_complement_of_one(p):
    p_ones = np.ones((p, 1))
    P = np.linalg.qr(np.linalg.qr(p_ones)[0], mode='complete')[0][:, 1:p]
    return P

def rho_n_grid(num):
    r_min=-2
    r_max=-2+2*num/15
    rho_n=np.logspace(r_min,r_max,num,10)
    return rho_n

def rho_1_grid(num):
    start=-0.5
    stop=-2.5
    rho_1=np.logspace(start,stop,num,10)
    return rho_1

def generate_Q(S_train,rho_n):
    (K,p,p)=S_train.shape
    Q=np.zeros((K, p, p))
    H=K*rho_n*(np.eye(p)-np.ones((p,1))*np.ones((1,p)))
    for k in np.arange(K):
        Q[k,:,:]=S_train[k,:,:]+H
    return Q

def phiminus(d,beta):
    return 0.5 * (np.sqrt(d ** 2 + 4 * beta) - d)

def phimul(beta,D,V):
    matD=(V*phiminus(D,beta))@V.T
    return matD

def Prox_L(mat_L):
    p=mat_L.shape[0]
    tmp_L=mat_L.copy()
    np.fill_diagonal(tmp_L,0)
    max_val=np.max(np.abs(tmp_L))
    min_val=np.min(np.abs(tmp_L))
    avg_val=0.5*(max_val+min_val)
    tmp_A=np.abs(tmp_L)/avg_val
    symme_A=0.5*(tmp_A+tmp_A.T)
    symme_A[symme_A>2]=2
    symme_A[symme_A<0.1]=0
    np.fill_diagonal(symme_A, 0)
    symme_A=-1*np.abs(symme_A)
    mat_D=np.diag(np.abs(np.sum(symme_A,axis=1)))+0.001*np.eye(p)
    max_L_o=mat_D+symme_A
    return max_L_o

def prox_p(J,b_m,f_m,beta):
    tmp_vec1=np.matmul(J,b_m)-f_m
    tmp_vec2=b_m-np.matmul(np.linalg.pinv(J),f_m)
    if np.linalg.norm(tmp_vec1)==0:
        a_m_1=np.matmul(np.linalg.pinv(J),f_m)
    else:
        s = 1 - beta / np.linalg.norm(tmp_vec1)
        a_m_1=np.maximum(s,0)*tmp_vec2
    return a_m_1

def prox_m(J,J_const,e_m,f_m,l_m,a_m,rho):
    tmp_vec1=1/rho*e_m
    tmp_vec2=1/rho*J.T@f_m
    tmp_vec3=J.T@J@a_m
    tmp_vec=tmp_vec1+tmp_vec2+l_m+tmp_vec3
    b_m_1=np.minimum(J_const@tmp_vec,0)
    return b_m_1

def tr_dot(X,Y):
    K=X.shape[0]
    res=0
    for k in range(K):
        res+=np.sum(X[k,:,:]*Y[k,:,:])
    return res


def obj_f(Xi,Q,P,J,A,rho_n,rho_1):
    res1=- np.log(np.linalg.det(Xi))
    Q_tilde=P.T@Q@P
    res2=res1.sum()+tr_dot(Xi,Q_tilde)
    (K,p,p)=A.shape
    res3=0
    for i in range(p):
        for j in range(start=i+1,stop=p):
            tmp_vec=J@A[:,i,j]
            res3+=rho_n*rho_1*np.linalg.norm(tmp_vec)
    return res2+2*res3


def relative_error(L_sol , L_true):
    assert L_true.shape==L_sol.shape
    (K, p, p) = L_true.shape
    tmp_LA=L_true.copy()
    tmp_LB=L_sol.copy()
    for k in range(K):
        np.fill_diagonal(tmp_LA[k,:,:], 0)
        np.fill_diagonal(tmp_LB[k,:,:], 0)
    return np.linalg.norm(tmp_LB-tmp_LA)/(K*np.linalg.norm(tmp_LA))


