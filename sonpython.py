import wave
import math
import numpy as np



### son original

nomfichier = 'son.wav'
monson = wave.open(nomfichier,'wb')

ncanal = 1       # mono
noctet = 2       # taille d'un echantillon : 1 octet = 8 bits
fe     = 16000   # frequence d'echantillonnage (en Hz)
duree  = 2       # duree du son (en s)
f0     = 440     # frequence de la sinusoide (en Hz)

nechantillon = int(duree*fe)

# contenu de l'en-tete
parametres = (ncanal,noctet,fe,nechantillon,'NONE','not compressed')
# creation de l'en-tete (44 octets)
monson.setparams(parametres)

### on cree le signal
val=[]
for i in range(0,nechantillon):
    if noctet==1:
       val.append(int(128.0 + 127.*math.sin(2.0*math.pi*f0*i/fe)))

    if noctet==2:
       val.append(int(32767.*math.sin(2.0*math.pi*f0*i/fe)))

### on le convertit en "son"
signal=[]
for i in range(0,nechantillon):
    if noctet==1:
       signal = wave.struct.pack('<B',val[i])   # <: little endian
                                                # B: unsigned char (1 octet) 
    if noctet==2:
       signal = wave.struct.pack('<h',val[i])   # <: little endian
                                                # h: short int (2 octets)

    monson.writeframes(signal) # ecriture de l'echantillon sonore courant

monson.close()

