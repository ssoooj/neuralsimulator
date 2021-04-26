import matplotlib.pyplot as plot
import numpy as np

np.random.seed(2)

neuralData = np.random.random([8, 50])
colorCodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]])
lineSize = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

plot.eventplot(neuralData, color = colorCodes, linelengths = lineSize)
plot.title('Spike raster plot')
plot.xlabel('Neuron')
plot.ylabel('Spike')
plot.show()