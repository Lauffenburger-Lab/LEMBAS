import bionetwork
from line_profiler import LineProfiler
import types
import torch
import scipy.sparse
import numpy.random
import numpy
from scipy.sparse.linalg import eigs
from numpy.linalg import eig
import time

#Generate random test data
batchSize = 6
networkSize = 800

networkList, nodeNames = bionetwork.getRandomNet(networkSize, 0.01)
MOA = numpy.full(networkList.shape, False, dtype=bool)
parameters = bionetwork.trainingParameters()

input = torch.randn(batchSize, networkSize, dtype=torch.double, requires_grad=True)
backError = input

#Construct network objects
#net1 = bionetwork.bionetworkAutoGrad(networkList, networkSize)
net2 = bionetwork.bionet(networkList, networkSize, MOA, parameters, torch.double)

netFuction = bionetwork.bionetworkFunction

print('Forward pass network function...')
lp = LineProfiler()
ctxTmp = types.SimpleNamespace()
lp_wrapper = lp(netFuction.forward)
lp_wrapper(ctxTmp, input, net2.weights, net2.bias, net2.A, net2.networkList, parameters)
lp.print_stats()


print('Backward pass network function...')
lp = LineProfiler()
lp_wrapper = lp(netFuction.backward)
lp_wrapper(ctxTmp, backError)
lp.print_stats()

# =============================================================================
# print('Forwardpass pass pytorch auto function...')
# lp = LineProfiler()
# lp_wrapper = lp(net1.forward)
# lp_wrapper(input)
# lp.print_stats()
# =============================================================================




#bias = torch.randn(size, 1, dtype=torch.double, requires_grad=True)
#A = scipy.sparse.csc_matrix((weights, networkList), shape=(size, size), dtype='d')
#AT = scipy.sparse.csc_matrix((weights, (networkList[1], networkList[0])), shape=(size, size), dtype='d')
#func = bionetwork(networkList, size, 'float64').double()
#bionetworkFunction.apply(input, weights, bias, A, AT, networkList)
#a = func(input)

# print('Spectral regularization...')
# lp = LineProfiler()
# lp_wrapper = lp(bionetwork.matrixRegularization)
# lp_wrapper(ctxTmp.AT, ctxTmp.networkList, ctxTmp.xRaw, 0.05)
# lp.print_stats()

# print('Spectral regularization...')
# lp = LineProfiler()
# lp_wrapper = lp(bionetwork.eigRegularization)
# lp_wrapper(ctxTmp.AT, ctxTmp.networkList, 0.05)
# lp.print_stats()

# =============================================================================
#
# lp = LineProfiler()
# lp_wrapper = lp(bionetwork.lreigs)
# lp_wrapper(net2.A)
# lp.print_stats()
# =============================================================================
# =============================================================================
# lp_wrapper = lp(eigenvaluRegularization)
# lp_wrapper(A, networkList, stepSize)
# lp.print_stats()
# =============================================================================


# =============================================================================
# start = time.time()
# for i in range(reps):
#     e1, v = eigs(A, k=1)
# print('Time:',  time.time()-start)
#
# start = time.time()
# for i in range(reps):
#     #tmp = A.T.toarray()
#     eT = e1[0] + 10**-6
#     tmp = A.toarray().T
#     e2, w = eigs(tmp, k=1, sigma=eT)
# print('Time:',  time.time()-start)
#
# =============================================================================
# =============================================================================
# start = time.time()
# for i in range(reps):
#     tmp = A.T.toarray()
#     e, v = eig(tmp)
# print('Time:',  time.time()-start)
#
# =============================================================================

# =============================================================================
#
# print('With autograd:')
# start = time.time()
# test = gradcheck(net1.double(), input, eps=1e-4, atol=1e-6)
# end = time.time()
# autoTime = end - start
# print(autoTime)
# print(test)
#
#
# print('Without autograd:')
# start = time.time()
# test = gradcheck(net2, input, eps=1e-4, atol=1e-6)
# end = time.time()
# manualTime = end-start
# print(manualTime)
# print(test)
# =============================================================================

# =============================================================================
# start = time.time()
# for i in range(reps):
#     #tmp = A.T.toarray()
#     eT = e1[0] + 10**-6
#     tmp = A.toarray().T
#     w = v.copy()
#     w = w/(w.T.dot(v))
#     w = w.squeeze()
#     e2, w = eigs(tmp, k=1, sigma=eT, v0=w)
# print('Time:',  time.time()-start)
# =============================================================================

# =============================================================================
# start = time.time()
# for i in range(reps):
#     tmp = A.T.toarray()
#     eT = e1[0] + 10**-6
#     e2, w = eigs(tmp, k=1, sigma=eT)
# print('Time:',  time.time()-start)
# =============================================================================

# =============================================================================
# start = time.time()
# for i in range(reps):
#     tmp = A.T
#     eT = e1[0] + 10**-6
#     e2, w = eigs(tmp, k=1, sigma=eT)
# print('Time:',  time.time()-start)
#
# =============================================================================
