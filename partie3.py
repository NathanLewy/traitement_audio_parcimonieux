import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


path_validation = os.path.abspath("./sons_oiseaux/SONS-VC")  # Convertir en chemin absolu
path_apprentissage = os.path.abspath("./sons_oiseaux/SONS")


# importer les données
os.chdir(path_apprentissage)
data = []
colors=['blue','grey','red']

for i,folder in enumerate(os.listdir()):
    folder_path = os.path.join(path_apprentissage,folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        sound, sr = sf.read(file_path)
        fft = np.abs(np.fft.rfft(sound, n=2**17))
        max = np.max(fft)
        f0 = np.argmax(fft)
        data.append({'features':(f0*sr/len(fft)/2, max), 'label':i})
    os.chdir(path_apprentissage)
plt.figure()

#tensorisation des données
X = np.array([d['features'] for d in data])
y = np.array([d['label'] for d in data])
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
C=1
svm = SVC(C=C, kernel='linear')
svm.fit(X, y)

z_test = svm.predict(X_test)
print(classification_report(y_test,z_test))


# Affichage de la frontière de décision avec des contours
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.75, cmap='coolwarm')
for i,x in enumerate(X_train):
    plt.scatter(x[0],x[1],color=colors[y_train[i]])
plt.title("SVM avec marge douce (kernel linéaire)")
plt.legend()
plt.show()


#prédire les données de validation
os.chdir(path_validation)
data = []



for i,folder in enumerate(os.listdir()):
    folder_path = os.path.join(path_apprentissage,folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        sound, sr = sf.read(file_path)
        fft = np.abs(np.fft.rfft(sound, n=2**17))
        max = np.max(fft)
        f0 = np.argmax(fft)
        data.append({'features':(f0*sr/len(fft)/2, max), 'label':i})

        
    os.chdir(path_apprentissage)

X = np.array([d['features'] for d in data])
y = np.array([d['label'] for d in data])
X = scaler.fit_transform(X)
Z = svm.predict(X)

for i,d in enumerate(data):
    plt.scatter(d['features'][0],d['features'][1],color=colors[Z[i]])

plt.show()

print(classification_report(y, Z))

