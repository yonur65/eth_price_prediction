function NMSE_calc = NMSE( wb, net, input, target)
% wb weights ve bias vektoru 
% wbnin transposu alýnýyor.
 net = setwb(net, wb');
% Hata matrixi elde ediliyor
 error = target - net(input);
% Hatanýn karesinin ortalamasý sonucun ortalamasýna bölünüyor
 NMSE_calc = mean(error.^2)/mean(var(target',1));
 
 
% It is independent of the scale of the target components and related to the Rsquare statistic via
% Rsquare = 1 - NMSEcalc ( see Wikipedia)
%figure;
plot(error);