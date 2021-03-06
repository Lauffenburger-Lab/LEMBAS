import bionetwork
import numpy
import matplotlib.pyplot as plt
import torch
import activationFunctions
from scipy.optimize import minimize

plt.rcParams["figure.figsize"] = (4,4)

leak = 0.01

x = numpy.linspace(-5, 5, 2001)
delta = (max(x)-min(x))/len(x)

functionType = 'leakyRelu' #MML leakyRelu  sigmoid
if functionType == 'MML':
    activation = activationFunctions.MMLactivation
    deltaActivation = activationFunctions.MMLDeltaActivation
    oneStepDeltaActivationFactor = activationFunctions.MMLoneStepDeltaActivationFactor
    invActivation = activationFunctions.MMLInvActivation
elif functionType == 'leakyRelu':
    activation = activationFunctions.leakyReLUActivation
    deltaActivation = activationFunctions.leakyReLUDeltaActivation
    oneStepDeltaActivationFactor = activationFunctions.leakyReLUoneStepDeltaActivationFactor
    invActivation = activationFunctions.leakyReLUInvActivation
elif functionType == 'sigmoid':
    activation = activationFunctions.sigmoidActivation
    deltaActivation = activationFunctions.sigmoidDeltaActivation
    oneStepDeltaActivationFactor = activationFunctions.sigmoidOneStepDeltaActivationFactor
    invActivation = activationFunctions.sigmoidInvActivation



y = activation(x.copy(), leak)
yd = deltaActivation(x.copy(), leak)

yNum = (y[1:len(y)]-y[0:(len(y)-1)])/delta

#plt.plot(x, yd)
#plt.plot(x[1:len(y)], yNum, 'o')
# plt.legend(['Analytical', 'Numerical'], frameon=False)
# plt.xlabel('X')
# plt.ylabel('dY/dX')
# plt.title('Derivative check')


plt.rcParams["figure.figsize"] = (8,6)
plt.figure()
ax1=plt.subplot(2, 3, 1)
plt.plot(x, y)
plt.title('activation function')

ax1=plt.subplot(2, 3, 2)
plt.plot(x, yd)
plt.plot(x[1:len(y)], yNum, 'o')
plt.legend(['Analytical', 'Numerical'], frameon=False)
plt.xlabel('X')
plt.ylabel('dY/dX')
plt.title('Derivative check')

# ax1=plt.subplot(2, 2, 3)
# plt.plot(yNum, yd[1:(len(y))], 'o')
# plt.title('Deviations numerical vs analytical')

ax1=plt.subplot(2, 3, 3)
plt.plot(x[0:len(y)-1], yNum-yd[1:len(y)], 'o')
plt.title('Deviations numerical vs analytical')
plt.tight_layout()


ax1=plt.subplot(2, 3, 4)
xInv = invActivation(y, leak)
plt.plot(xInv, x, 'o')
plt.xlabel('inv(y)')
plt.ylabel('x')
plt.title('check inverse')

ax1=plt.subplot(2, 3, 5)
ydOnestep = oneStepDeltaActivationFactor(torch.tensor(y), leak)
plt.plot(y, ydOnestep)
plt.xlabel('y')
plt.ylabel('onestep(y)=deltaActivation(inv(y))')
plt.title('one step')


ax1=plt.subplot(2, 3, 6)
plt.plot(yd, ydOnestep, 'o')
plt.xlabel('onestep(y)')
plt.ylabel('delta(x)')
plt.title('check onestep')





#%%
# plt.figure()
# plt.rcParams["figure.figsize"] = (4,4)

# leak = 0.01

# MM = lambda km, x: 1/((km/x) + 1)
# x = numpy.linspace(0.00001, 5, 201)
# y = bionetwork.activation(x.copy(), leak)
# res = minimize(lambda km: sum((MM(km, x)-y)**2), 1)
# optimalMM = MM(res.x[0], x)
# print('MM', res.fun)

# tanh = lambda k, x: numpy.tanh(k*x)
# res = minimize(lambda k: sum((tanh(k, x)-y)**2), 1)
# optimalTanh = tanh(res.x[0], x)
# print('tanh', res.fun)


# sigmoid = lambda k, x: 1/(1 + numpy.exp(-x*k[0] + k[1]))
# res = minimize(lambda k: sum((sigmoid(k, x)-y)**2), numpy.array([1, 0]))
# optimalSigmoid = sigmoid(res.x, x)
# print('Sigmoid', res.fun)


# plt.figure()
# plt.plot(x, y, color='black')
# plt.plot(x, optimalMM)
# plt.plot(x, optimalTanh)
# plt.plot(x, optimalSigmoid)
# plt.ylim([0, 1])
# plt.legend(['MML', 'MM', 'tanh', 'sigmoid'], frameon=False)

#%%
plt.figure()
plt.rcParams["figure.figsize"] = (4,4)

leak = 0.01

MM = lambda km, x: numpy.where(x>0, 1/((km/x) + 1), 0)
x = numpy.linspace(-0.5, 10, 201)
y = MM(2, x.copy())

MML = lambda x: activationFunctions.MMLactivation(x.copy(), leak)
res = minimize(lambda k: sum((MML(k[0]*x+k[1])-y)**2), numpy.array([1, 0]))
k = res.x
optimalMML = MML(k[0]*x+k[1])
print('MML', res.fun)

# tanh = lambda k, x: numpy.tanh(k[0]*x + k[1])
# res = minimize(lambda k: sum((tanh(k, x)-y)**2), numpy.array([1, 0]))
# optimalTanh = tanh(res.x, x)
# print('tanh', res.fun)

relu = lambda x: activationFunctions.leakyReLUActivation(x.copy(), leak)
res = minimize(lambda k: sum((relu(k[0]*x+k[1])-y)**2), numpy.array([1, 0]))
k = res.x
optimalReLU = relu(k[0]*x+k[1])
print('Relu', res.fun)

sigmoid = lambda x: activationFunctions.sigmoidActivation(x.copy(), leak)
res = minimize(lambda k: sum((sigmoid(x*k[0] + k[1])-y)**2), numpy.array([1, 0]))
k = res.x
optimalSigmoid = sigmoid(k[0]*x+k[1])
print('Sigmoid', res.fun)


plt.figure()
plt.plot(x, y, color='black')
plt.plot(x, optimalMML)
plt.plot(x, optimalReLU)
#plt.plot(x, optimalTanh)
plt.plot(x, optimalSigmoid)
plt.ylim([0, 1])
plt.legend(['MM', 'MML', 'ReLU', 'sigmoid'], frameon=False)
plt.title('Fit to a MM')
plt.xlabel('x')
plt.ylabel('y')
