# -*- coding: utf-8 -*-
"""

# Pour les filtres en python, voir (lien donné dans le cours) :
#
# https://www.f-legrand.fr/scidoc/docmml/numerique/filtre/filtrenum/filtrenum.html
#
# Note : comme indiqué pour les filtres de matlab, n'utilisez que les filtres FIR !!!
"""

import numpy
import scipy.signal
import matplotlib.pyplot as plt
import math

### on fait une sinusoïde

fs=8000.   # fréquence d'échantillonnage, en Hz
f0=3000.   # fréquence de la sinusoïde (doit être <fs/2), en Hz
print("frequence de la sinusoide : ", f0)

tt=numpy.arange(0.,1.,1./fs)

xx=numpy.zeros(len(tt))
for kk in range(1,len(tt)):
	xx[kk]=math.cos(2*math.pi*f0*tt[kk])

### design du filtre passe-bas

P=100
b1 = scipy.signal.firwin(numtaps=2*P+1,cutoff=[0.25],window='hann',fs=1)


plt.figure(1,figsize=(10, 8))
plt.plot(b1)
plt.title('filtre passe-bas')
plt.draw()


w,h=scipy.signal.freqz(b1)

plt.figure(2,figsize=(10, 10))
plt.subplot(211)
plt.plot(w/(2*numpy.pi),20*numpy.log10(numpy.absolute(h)))
plt.xlabel('filtre passe-bas -- reponse en frequence - amplitude')
plt.subplot(212)      
plt.plot(w/(2*numpy.pi),numpy.unwrap(numpy.angle(h)))
plt.xlabel('filtre passe-bas -- reponse en frequence - phase')
plt.draw()


### design du filtre passe-haut

b2 = scipy.signal.firwin(numtaps=2*P+1,cutoff=[0.25],window='hann',fs=1,pass_zero=False)

plt.figure(3,figsize=(10, 8))
plt.title('filtre passe-haut')
plt.plot(b2)
plt.draw()


w2,h2=scipy.signal.freqz(b2)

plt.figure(4,figsize=(10, 10))
plt.subplot(211)
plt.plot(w2/(2*numpy.pi),20*numpy.log10(numpy.absolute(h2)))
plt.xlabel('filtre passe-haut -- reponse en frequence - amplitude')
plt.subplot(212)      
plt.plot(w2/(2*numpy.pi),numpy.unwrap(numpy.angle(h2)))
plt.xlabel('filtre passe-haut -- reponse en frequence - phase')
plt.draw()


### et maintenant on filtre notre sinusoïde

yy=scipy.signal.filtfilt(b1, 1, xx)

plt.figure(5,figsize=(10, 10))
plt.subplot(211)
plt.plot(tt,xx)
plt.subplot(212)
plt.plot(tt,yy)
plt.xlabel('sunusoide filtree passe-bas')
plt.draw()

yy2=scipy.signal.filtfilt(b2, 1, xx)

plt.figure(6,figsize=(10, 10))
plt.subplot(211)
plt.plot(tt,xx)
plt.subplot(212)
plt.plot(tt,yy2)
plt.xlabel('sinusoide filtree passe-haut')
plt.draw()




hilbert_passbas=numpy.unwrap(numpy.angle(scipy.signal.hilbert(yy)),period=2*numpy.pi)
hilbert_passhaut=numpy.unwrap(numpy.angle(scipy.signal.hilbert(yy2)),period=2*numpy.pi)
f_passbas = fs/(2*numpy.pi)*numpy.diff(hilbert_passbas)
f_passhaut = fs/(2*numpy.pi)*numpy.diff(hilbert_passhaut)


plt.figure()
plt.plot(scipy.signal.hilbert(yy2).real,color="red")
plt.plot(scipy.signal.hilbert(yy2).imag,color="blue")
plt.show()

plt.figure()
plt.plot(hilbert_passbas,color="red")
plt.plot(hilbert_passhaut,color="blue")
plt.show()

plt.figure(7,figsize=(10, 10))
plt.subplot(211)
plt.plot(f_passbas)
plt.title('filtre passe-bas')
plt.subplot(212)
plt.plot(f_passhaut)
plt.title('filtre passe-haut')
plt.show()