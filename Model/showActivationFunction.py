import activationFunctions
import numpy
import matplotlib.pyplot as plt


leak = 0.01

x = numpy.linspace(-2, 3, 1001)
y = activationFunctions.MMLactivation(x.copy(), leak)

# x2 = numpy.linspace(0, 1, 10)
# y2 = activationFunctions.MMLactivation(x2.copy(), leak)

plt.rcParams["figure.figsize"] = (3,3)
plt.plot(x, y)
#plt.scatter(x2, y2)

plt.plot([-2, 3], [0, 0], 'black')
plt.ylim([-0.05, 1])

#plt.gca().set_aspect('equal')