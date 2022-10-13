# -*- coding: utf-8 -*-
"""
Generates data for numerical experiments
"""

import numpy as np
import networkx as nx
from scipy.sparse import csgraph
import random

from JEMGL.Helper.Basic_linearalgbra import trp


def generate_laplacian_matrix(p=15, M=1, style='erdos', gamma=2.8, prob=0.2, scale=False, seed=None):
    """
    Generates a laplacian matrix with associated covariance matrix from a random network.
    Parameters
    """

    L = int(p / M)
    assert M * L == p

    A = np.zeros((p, p))
    Sigma = np.zeros((p, p))

    if seed is not None:
        nxseed = seed
    else:
        nxseed = None

    for m in np.arange(M):

        if nxseed is not None:
            nxseed = int(nxseed + m)

        if style == 'powerlaw':
            G_m = nx.generators.random_graphs.random_powerlaw_tree(n=L, gamma=gamma, tries=max(5 * p, 1000),
                                                                   seed=nxseed)
        elif style == 'erdos':
            G_m = nx.generators.random_graphs.erdos_renyi_graph(n=L, p=prob, seed=nxseed, directed=False)
        else:
            raise ValueError(f"{style} is not a valid choice for the network generation.")
        A_m = nx.to_numpy_array(G_m)

        # generate random numbers for the nonzero entries
        if seed is not None:
            np.random.seed(seed)

        B1 = np.random.uniform(low=.75, high=2, size=(L, L))
        B2 = np.random.choice(a=[-1, 1], p=[.5, .5], size=(L, L))

        A_m = A_m * (B1 * B2)

        A[m * L:(m + 1) * L, m * L:(m + 1) * L] = A_m

    row_sum_od = 1.5 * abs(A).sum(axis=1) + 1e-10
    # broadcasting in order to divide ROW-wise
    A = A / row_sum_od[:, np.newaxis]

    A = .5 * (A + A.T)

    # A has 0 on diagonal, fill with 1s
    A = A + np.eye(p)
    assert all(np.diag(A) == 1), "Expected 1s on diagonal"

    # make sure A is pos def
    D = np.linalg.eigvalsh(A)
    if D.min() < 1e-8:
        A += (0.1 + abs(D.min())) * np.eye(p)

    Ainv = np.linalg.pinv(A, hermitian=True)

    # scale by inverse of diagonal and 0.6*1/sqrt(d_ii*d_jj) on off-diag
    if scale:
        d = np.diag(Ainv)
        scale_mat = np.tile(np.sqrt(d), (Ainv.shape[0], 1))
        scale_mat = (1 / 0.6) * (scale_mat.T * scale_mat)
        np.fill_diagonal(scale_mat, d)

        Sigma = Ainv / scale_mat
        Laplacian = None

    else:
        Sigma = Ainv.copy()
        Laplacian = A.copy()

    assert abs(Sigma.T - Sigma).max() <= 1e-8
    D = np.linalg.eigvalsh(Sigma)
    assert D.min() > 0, "generated matrix Sigma is not positive definite"

    return Sigma, Laplacian


def time_varying_graph(p=15, K=3, M=1,seed=None):
    """
    generates ER time-varying networks.
    p: dimension
    K: number of instances/time-stamps
    M: number of sublocks in each instance
    """
    Laplacian = np.zeros((K, p, p))
    Sigma = np.zeros((K, p, p))

    Adj_0 = generate_adjacency_matrix(p=p, M=M, style='erdos',prob=0.3, seed=seed)
    graph = nx.from_numpy_array(Adj_0, create_using=nx.Graph)
    for k in np.arange(K):
        edge_len=len(list(graph.edges))
        cnt=int(0.1*edge_len)
        for i in np.arange(cnt):
            edges = list(graph.edges)
            chosen_edge = random.choice(edges)
            graph.remove_edge(chosen_edge[0], chosen_edge[1])
        Adj_k=nx.to_numpy_array(graph)
        Laplacian_k=csgraph.laplacian(Adj_k)
        # make sure Laplacian_k is pos def
        D = np.linalg.eigvalsh(Laplacian_k)
        if D.min() < 1e-8:
            Laplacian_k += (0.01 + abs(D.min())) * np.eye(p)

        Sigma_k = np.linalg.pinv(Laplacian_k, hermitian=True)
        assert abs(Sigma_k.T - Sigma_k).max() <= 1e-8
        D = np.linalg.eigvalsh(Sigma_k)
        assert D.min() > 0, "generated matrix Sigma is not positive definite"
        Laplacian[k, :, :] = Laplacian_k
        Sigma[k, :, :] = Sigma_k

    return Sigma, Laplacian


def group_modular_graph(p=30, K=3, M=3, scale=False, seed=None):
    """
    generates a group of similar modular networks. In each single network one block disappears (randomly)
    p: dimension
    K: number of instances/time-stamps
    M: number of sublocks in each instance, M should greater than 3
    """
    Sigma = np.zeros((K, p, p))

    L = int(p / M)
    assert M * L == p

    Sigma_0, laplacian_0 = generate_laplacian_matrix(p=p, M=M, style='powerlaw', scale=scale, seed=seed)
    # contains the number of the block disappearing for each k=1,..,K
    if seed is not None:
        np.random.seed(seed)

    block = np.random.randint(M, size=K)

    for k in np.arange(K):
        Sigma_k = Sigma_0.copy()
        if k > 1:
            Sigma_k[block[k] * L: (block[k] + 1) * L, block[k] * L: (block[k] + 1) * L] = np.eye(L)

        Sigma[k, :, :] = Sigma_k

    Laplacian= np.linalg.pinv(Sigma, hermitian=True)
    Sigma, Laplacian = ensure_sparsity(Sigma, Laplacian)

    return Sigma, Laplacian


def ensure_sparsity(Sigma, Laplacian):
    Laplacian[abs(Laplacian) <= 1e-2] = 0

    D = np.linalg.eigvalsh(Laplacian)
    assert D.min() > 0
    Sigma = np.linalg.pinv(Laplacian, hermitian=True)
    return Sigma, Laplacian


def sample_covariance_matrix(Sigma, N, seed=None):
    """
    samples data for a given covariance matrix Sigma (with K layers)
    return: sample covariance matrix S
    """
    if seed is not None:
        np.random.seed(seed)

    if len(Sigma.shape) == 2:
        assert abs(Sigma - Sigma.T).max() <= 1e-10
        (p, p) = Sigma.shape

        sample = np.random.multivariate_normal(np.zeros(p), Sigma, N).T
        S = np.cov(sample, bias=True)

    else:
        assert abs(Sigma - trp(Sigma)).max() <= 1e-10
        (K, p, p) = Sigma.shape

        sample = np.zeros((K, p, N))
        for k in np.arange(K):
            sample[k, :, :] = np.random.multivariate_normal(np.zeros(p), Sigma[k, :, :], N).T

        S = np.zeros((K, p, p))
        for k in np.arange(K):
            # normalize with N --> bias = True
            S[k, :, :] = np.cov(sample[k, :, :], bias=True)

    return S, sample


def generate_adjacency_matrix(p=15, M=1, style='erdos', gamma=2.8, prob=0.2, seed=None):
    """
    Generates the adjacency matrix from a random network.
    Parameters
    """

    L = int(p / M)
    assert M * L == p

    A = np.zeros((p, p))

    if seed is not None:
        nxseed = seed
    else:
        nxseed = None

    for m in np.arange(M):

        if nxseed is not None:
            nxseed = int(nxseed + m)

        if style == 'powerlaw':
            G_m = nx.generators.random_graphs.random_powerlaw_tree(n=L, gamma=gamma, tries=max(5 * p, 1000),seed=nxseed)
        elif style == 'erdos':
            G_m = nx.generators.random_graphs.erdos_renyi_graph(n=L, p=prob, seed=nxseed, directed=False)
        else:
            raise ValueError(f"{style} is not a valid choice for the network generation.")
        A_m = nx.to_numpy_array(G_m)

        # generate random numbers for the nonzero entries
        if seed is not None:
            np.random.seed(seed)

        B1 = np.random.uniform(low=.75, high=2, size=(L, L))
        A_m = A_m *B1
        # only use upper triangle and symmetrize
        A_m = np.triu(A_m)
        A_m = A_m + A_m.T

        A[m * L:(m + 1) * L, m * L:(m + 1) * L] = A_m

    return A

def construct_group_graph(p=15, K=3, M=1,  seed=None):
    Laplacian = np.zeros((K, p, p))
    Sigma = np.zeros((K, p, p))

    L = int(p / M)
    assert M * L == p

    Adjacency_c = generate_adjacency_matrix(p=p, M=M, style='erdos', seed=seed)
    if seed is not None:
        np.random.seed(seed)
    for k in np.arange(K):
        B1 = np.random.uniform(low=.5, high=1, size=(L, L))
        B2 = np.random.choice(a=[-1, 1], p=[.5, .5], size=(L, L))
        B3=np.random.choice(a=[0, 1], p=[.95, .05], size=(L, L))
        U_k=B1*B2*B3
        W_k=Adjacency_c+U_k
        W_k[W_k < 0] = 0
        W_k[W_k>2]=2
        np.fill_diagonal(W_k, 0)
        Laplacian_k=csgraph.laplacian(W_k)

        # make sure Laplacian_k is pos def
        D = np.linalg.eigvalsh(Laplacian_k)
        if D.min() < 1e-8:
            Laplacian_k += (0.01 + abs(D.min())) * np.eye(p)

        Sigma_k = np.linalg.pinv(Laplacian_k,hermitian=True)
        assert abs(Sigma_k.T - Sigma_k).max() <= 1e-8
        D = np.linalg.eigvalsh(Sigma_k)
        assert D.min() > 0, "generated matrix Sigma is not positive definite"
        Laplacian[k, :, :] = Laplacian_k
        Sigma[k, :, :] = Sigma_k

    return Sigma, Laplacian
