#démonstration de wavedec

import numpy as np
import matplotlib.pyplot as plt
import pywt

sine = np.sin(2 * np.pi * np.linspace(0, 1, 1024))
coeffs = pywt.wavedec(sine, 'db1', level=1)
print(len(coeffs))
for k in range(len(coeffs)):
    print(coeffs[k].shape)

reconstruit = pywt.waverec(coeffs, 'db1')

plt.plot(reconstruit)
plt.show()