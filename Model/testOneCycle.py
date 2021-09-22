# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:30:23 2020

@author: Avlant-MIT
"""
import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork

maxIter = 10000
peak = 1000
maxLr = 2e-3
minLr = 1e-8

test = numpy.array(range(maxIter))
testY = numpy.zeros(maxIter)
for i in range(len(testY)):
    testY[i] = bionetwork.oneCycle(test[i], maxIter, maxHeight = maxLr, minHeight = minLr, peak = peak)
plt.plot(test, testY)


optimizer = torch.optim.Adam([torch.tensor(1)], lr=0.0004, amsgrad=True)


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, maxLr, epochs=maxIter, pct_start=(peak/maxIter), final_div_factor=1e6, steps_per_epoch=1, cycle_momentum=False)
testO = numpy.zeros(len(testY))
testO[:] = min(testY)
for i in range(maxIter):
    testO[i] = optimizer.param_groups[0]['lr']
    scheduler.step()
plt.plot(test, testO)

plt.plot([peak,peak], [0, 0.01], 'black')