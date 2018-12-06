scores = [3.0, 1.0, 0.2]

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

#print('exp:', np.exp(scores) )
#print('sum', np.sum(np.exp(scores), axis=0))
#print('arange', np.arange(-2.0, 6.0, 0.1))
#print('div', np.exp(scores) / np.sum(np.exp(scores)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print('softMax', softmax(scores))

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.savefig('softmax.svg')  

print('softMax', softmax([4.0, 1.0, 0.3]))
print('softMax', softmax([2.0, 1.0, 7.0]))
print('softMax', softmax([2.0, 1.0, 7.0, 8.0, 10.0]))

