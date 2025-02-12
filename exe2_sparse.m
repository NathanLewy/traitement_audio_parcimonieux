%%% Comment faire des filtres 'fir' en matlab : regardez lignes 67 à 82
%%%
%%% S. Rossignol -- 13/02/24

clear all;
close all;

pkg load signal; %%% nécessaire pour octave ; si vous utilisez 'matlab', commentez cette ligne


%%% on fait les signaux

fe=8000;  %%% fréquence d'échantillonnage
f0=440;   %%% fréquence de la sinusoïde

ttt=[0:1/fe:2]; %%% instants d'échantillonnage

xxx=0.1*cos(2*pi*f0*ttt);     %%% x(t)

nnn=0.1*randn(1,length(xxx)); %%% n(t)

signal=(xxx+nnn)';            %%% s(t)


%%% on analyse le signal somme -- 1

tfen=0.03;
lfen=round(fe*tfen);
tfft=8192*4; %%% taille de la FFT
[Sspec, fspec, tspec] = specgram(signal, tfft, fe, hanning(lfen), lfen-round(fe*0.005));

dofig=1;
if dofig==1
  figure(1);
  clf;
  plot(fspec,abs(Sspec(:,1)),'Linewidth',3);
  title('module FFT ; frequences positives','Fontsize',20);
  grid on;
  xlabel('frequence (Hz)','Fontsize',20);
  hold off;
  print -depsc2 exe2_p11_1.eps  %%% permet de faire les figures qui sont dans les transparents
end;


%%% détection -- 1

for ii=1:size(Sspec,2)
  [vv,pp]=max(abs(Sspec(:,ii)));
  res(ii) = fspec(pp);
end;

fprintf(1,'%f pc between 438 and 442 --- mean=%f\n',sum(res>438 & res<442)/length(res)*100.,mean(res));

figure(2);
clf;
hist(res);
title('position du max des spectres','Fontsize',20);
xlabel('frequence (Hz)','Fontsize',20);
hold off;
print -depsc2 exe2_p11_2.eps  %%% permet de faire les figures qui sont dans les transparents

doit=1;
if doit==1

  %%% filtrage

  larg=30;correct=10;
  aa=1;
  bb=fir2(300,[0 2*(f0-larg+correct)/fe 2*(f0-larg+correct)/fe 2*(f0+larg+correct)/fe 2*(f0+larg+correct)/fe 1],[0 0 1 1 0 0]);

  [hh,ww]=freqz(bb,1,1000);
  figure(5); %%% réponse en fréquence du filtre
  clf;
  plot(ww/pi*4000,abs(hh));
  hold off;
  pause

  sigfil=filtfilt(bb,aa,signal);

  figure(3);
  plot(sigfil);
  hold off;


  %%% on analyse le signal -- 2

  tfen=0.03;
  lfen=round(fe*tfen);
  tfft=8192*4;
  [Sspec, fspec, tspec] = specgram(sigfil, tfft, fe, hanning(lfen), lfen-round(fe*0.005));

  dofig=0;
  if dofig==1
    figure(4);
    clf;
    imagesc(tspec+tfen/2, fspec, (log10(abs(Sspec))));
    title('module FFT ; frequences positives','Fontsize',20);
    xlim([0 max(ttt)]);
    grid on;
    xlabel('temps (s)','Fontsize',20);
    ylabel('frequence (Hz)','Fontsize',20);
    hold off;
  end;


  %%% détection -- 2

  for ii=1:size(Sspec,2)
    [vv,pp]=max(abs(Sspec(:,ii)));
    res(ii) = fspec(pp);
  end;

  fprintf(1,'%f pc between 438 and 442\n',sum(res>438 & res<442)/length(res)*100.)

  figure(5);
  clf;
  hist(res);
  title('position du max des spectres','Fontsize',20);
  xlabel('frequence (Hz)','Fontsize',20);
  hold off;

end;

