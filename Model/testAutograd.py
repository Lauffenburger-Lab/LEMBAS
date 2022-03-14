import torch
import bionetwork
import matplotlib.pyplot as plt
import numpy
import plotting
import time

networkSize = 50
batchsize = 5
activationFunction = 'MML'

networkList, nodeNames = bionetwork.getRandomNet(networkSize, 0.1)
MOA = numpy.full(networkList.shape, False, dtype=bool)

input = torch.randn(batchsize, len(nodeNames), dtype=torch.double, requires_grad=True)


parameters = bionetwork.trainingParameters(iterations=150, clipping=1)
net1 = bionetwork.bionetworkAutoGrad(networkList, len(nodeNames))
net2 = bionetwork.bionet(networkList, len(nodeNames), MOA, parameters, activationFunction, torch.double)
net2.weights.data = net1.A.values.data
net2.bias.data = net1.bias.data

#test = torch.autograd.gradcheck(net1, input, eps=1e-4, atol=1e-6)
#test = torch.autograd.gradcheck(net2, input, eps=1e-6, atol=1e-6)


networkSize = 100
batchsize = 5
networkList, nodeNames = bionetwork.getRandomNet(networkSize, 0.5)
MOA = numpy.full(networkList.shape, False, dtype=bool)

input1 = torch.randn(batchsize, len(nodeNames), dtype=torch.double, requires_grad=True)
input2 = input1.clone().detach().requires_grad_(True)

net1 = bionetwork.bionetworkAutoGrad(networkList, len(nodeNames), parameters['iterations'])
net2 = bionetwork.bionet(networkList, len(nodeNames), MOA, parameters, activationFunction, torch.double)

net2.weights.data = net1.A.values.data.detach().clone()
net2.bias.data = net1.bias.data.detach().clone()

criterion = torch.nn.MSELoss()
prediction1 = net1(input1)

predictionForLoss = torch.randn(input1.shape).double()
predictionForLoss.requires_grad = False

start = time.time()
loss1 = criterion(prediction1, predictionForLoss)
a = loss1.backward()
gradWeights = net1.A.grad.coalesce()
print(time.time() - start)

start = time.time()
prediction2 = net2(input2)
loss2 = criterion(prediction2, predictionForLoss)
loss2.backward()
print(time.time() - start)

#net1.A.to_dense().detach().numpy()
#net2.A.toarray()


#%%
plt.rcParams["figure.figsize"] = (6,6)
ax1=plt.subplot(2, 2, 1)
plt.plot(torch.flatten(prediction2.data, 0), torch.flatten(prediction1.data, 0), 'o', color='black')

plt.title('Prediction')
plt.xlabel('steady state')
plt.ylabel('autograd')
plotting.lineOfIdentity()
#print(torch.flatten(prediction1.data, 0)- torch.flatten(prediction2.data, 0))


ax1=plt.subplot(2, 2, 2)
plt.plot(torch.flatten(input2.grad, 0), torch.flatten(input1.grad, 0), 'o', color='black')
plt.title('Input gradient')
plt.xlabel('steady state')
plt.ylabel('autograd')
plotting.lineOfIdentity()
#print(torch.flatten(prediction1.data, 0)- torch.flatten(prediction2.data, 0))

ax1=plt.subplot(2, 2, 3)
plt.plot(net2.weights.grad, gradWeights.values(), 'o', color='black')
plt.title('Weight gradient')
plt.xlabel('steady state')
plt.ylabel('autograd')
plotting.lineOfIdentity()
#print(net1.bias.grad-net2.bias.grad)

ax2=plt.subplot(2, 2, 4)
plt.plot(net2.bias.grad, net1.bias.grad, 'o', color='black')
plt.title('Bias gradient')
plt.xlabel('steady state')
plt.ylabel('autograd')
plotting.lineOfIdentity()
plt.tight_layout()

# =============================================================================
# plt.figure
#
# ax1=plt.subplot(1, 2, 1)
# plt.plot(net1.A.values.data, net2.weights.data, 'o', color='black')
# plt.ylabel('manual')
# plt.title('Weights')
# plt.plot([-1, 1], [-1, 1], transform=ax1.transAxes)
#
# ax1=plt.subplot(1, 2, 2)
# plt.plot(net1.bias.data, net2.bias.data, 'o', color='black')
# plt.ylabel('manual')
# plt.title('bias')
# plt.plot([-1, 1], [-1, 1], transform=ax1.transAxes)
#
#
# =============================================================================
