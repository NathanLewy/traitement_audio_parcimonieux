import pandas as pd
import matplotlib.pyplot as plt

# Chemins vers les fichiers CSV
oiseau1_path = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre/SONS_features/oiseau1/frequences_amplitudes.csv'
oiseau2_path = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre/SONS_features/oiseau2/frequences_amplitudes.csv'
oiseau3_path = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre/SONS_features/oiseau3/frequences_amplitudes.csv'

# Lire les fichiers CSV avec la première ligne comme en-tête
oiseau1_data = pd.read_csv(oiseau1_path, header=0, delimiter=',')
oiseau2_data = pd.read_csv(oiseau2_path, header=0, delimiter=',')
oiseau3_data = pd.read_csv(oiseau3_path, header=0, delimiter=',')

# Tracer les graphiques en scatter plot
plt.figure(figsize=(12, 6))

plt.scatter(oiseau1_data['Frequence Fondamentale (Hz)'], oiseau1_data['Amplitude Maximale'], label='Oiseau 1', alpha=0.6)
plt.scatter(oiseau2_data['Frequence Fondamentale (Hz)'], oiseau2_data['Amplitude Maximale'], label='Oiseau 2', alpha=0.6)
plt.scatter(oiseau3_data['Frequence Fondamentale (Hz)'], oiseau3_data['Amplitude Maximale'], label='Oiseau 3', alpha=0.6)

plt.xlabel('Fréquence Fondamentale (Hz)')
plt.ylabel('Amplitude Maximale')
plt.title('Amplitude en fonction de la Fréquence')
plt.legend()
plt.grid(True)
plt.show()
