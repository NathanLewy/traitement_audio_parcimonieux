# Comment faire de jolis sons en python
#
# Stéphane Rossignol -- 2024

import wave, math, struct, numpy

fe = 8000 # fréquence d'échantillonnage en Hertz
du = 1*fe # nombre d'échantillons dans le signal
f0 = 880  # fréquence fondamentale du signal périodique en Hertz

obj = wave.open('sound.wav','w')
obj.setnchannels(1)  # mono (on n'a pas besoin de la stéréo, ici)
obj.setsampwidth(2)  # nombre d'octets par échantillon pour la quantification
                     # => ne pas changer ça
obj.setframerate(fe) 
for i in range(du):
   value = 0.1*math.cos(2*math.pi*f0*i/fe) + \
           0.2*math.cos(2*math.pi*2*f0*i/fe) + \
           0.3*math.cos(2*math.pi*3*f0*i/fe) + \
           0.4*math.cos(2*math.pi*4*f0*i/fe)
   value=value/1.1         # pour être sûr que les échantillons sont entre -1 et 1
   if value>1 or value<-1:
      print('échec : certains échantillons ne sont pas dans [-1 1] (diviser "value" ci-dessus par un flottant > 1.')
      break
   value=int(value*32768); # si 'obj.setsampwidth(2)' change, changer 
                           # le 32768 aussi, car il faut :
                           #   2^(obj.setsampwidth(2)*8)/2=32768 
                           # et il faut aussi changer le '<h' ci-dessous 
                           # (voir doc de 'wave')
   data = struct.pack('<h', value)
   obj.writeframesraw( data )
obj.close()

