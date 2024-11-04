import torch
import bionetwork
import matplotlib.pyplot as plt
import numpy
import plotting
import time
import pandas
import saveSimulations

batchsize = 5
activationFunction = 'MML'


parameters = bionetwork.trainingParameters(iterations=150, clipping=1)
networkSize = 100
batchsize = 5
#networkList, nodeNames = bionetwork.getRandomNet(networkSize, 0.5)
#MOA = numpy.full(networkList.shape, False, dtype=bool)

input1 = torch.randn(batchsize, networkSize, dtype=torch.double, requires_grad=True)
input2 = input1.clone().detach().requires_grad_(True)

#net1 = bionetwork.bionetworkAutoGrad(networkList, len(nodeNames), parameters['iterations'])
#net2 = bionetwork.bionet(networkList, len(nodeNames), MOA, parameters, activationFunction, torch.double)

net1 = torch.load('testAutogradModel/Model-Autograd.pt')
net2 = torch.load('testAutogradModel/Model-Bionet.pt')

#net2.weights.data = net1.A.coalesce().values().data.clone()
#net2.bias.data = net1.bias.data.detach().clone()
#saveSimulations.save('simulations', 'parameterTest', {'Model-Autograd':net1,  'Model-Bionet':net2})
#plt.scatter(net2.weights.data.detach().numpy().flatten(), net1.A.coalesce().values().data.detach().numpy().flatten())
#plt.scatter(net2.bias.data.detach().numpy().flatten(), net1.bias.data.detach().numpy().flatten())


criterion = torch.nn.MSELoss()
prediction1 = net1(input1)

predictionForLoss = torch.randn(input1.shape).double()
predictionForLoss.requires_grad = False

start = time.time()
loss1 = criterion(prediction1, predictionForLoss)
a = loss1.backward()
gradWeights = net1.A.grad.coalesce().detach().values().numpy().flatten()
print(time.time() - start)

start = time.time()
prediction2 = net2(input2)
loss2 = criterion(prediction2, predictionForLoss)
loss2.backward()
print(time.time() - start)


#net1.A.to_dense().detach().numpy()
#net2.A.toarray()


#%%
folder = 'figures/SI Figure 2/'

titles = ['Prediction', 'Input gradient', 'Weight gradient', 'Bias gradient']


df = pandas.DataFrame((prediction2.data.detach().numpy().flatten(), prediction1.data.detach().numpy().flatten()), index=['steady state', 'autograd']).T
df.to_csv(folder + titles[0] + '.tsv', sep='\t', index=False)

df = pandas.DataFrame((input2.grad.detach().numpy().flatten(), input1.grad.detach().numpy().flatten()), index=['steady state', 'autograd']).T
df.to_csv(folder + titles[1] + '.tsv', sep='\t', index=False)

df = pandas.DataFrame((net2.weights.grad.numpy().flatten(), gradWeights), index=['steady state', 'autograd']).T
df.to_csv(folder + titles[2] + '.tsv', sep='\t', index=False)

df = pandas.DataFrame((net2.bias.grad.numpy().flatten(), net1.bias.grad.numpy().flatten()), index=['steady state', 'autograd']).T
df.to_csv(folder + titles[3] + '.tsv', sep='\t', index=False)


plt.rcParams["figure.figsize"] = (6,6)
plt.figure()
for i in range(len(titles)):
    plt.subplot(2, 2, 1+i)
    df = pandas.read_csv(folder + titles[i] + '.tsv', sep='\t')
    plt.plot(df['steady state'], df['autograd'], 'o', color='black')

    plt.xlabel('steady state')
    plt.ylabel('autograd')
    plt.gca().axis('equal')

    plotting.lineOfIdentity()
    plt.title(titles[i])

plt.tight_layout()
plt.savefig(folder + 'fig.svg')



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
