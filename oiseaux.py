import numpy as np
import os

path_apprentissage = "sons_oiseaux"
print(os.listdir())
os.chdir(path_apprentissage)
data = []
for i,folder in enumerate(os.listdir()):
    os.chdir(path_apprentissage)
    os.chdir(folder)
    for filename in os.listdir():
        file = np.fromfile(filename)
        features = np.fft.fft(file)
        print(file)
        data.append()
