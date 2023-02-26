% Temizleme iþlemi yapýlýp veriler okunup hazýrlanýyor
clc;
clear;
close all;
veri_okuma;
data = Close1;
ndata=size(data,1);
train_rate = 0.80;
num_train = round(ndata*train_rate);
[scaled_data,PS] = mapminmax(data');
scaled_data = scaled_data';
traindata=scaled_data(1:num_train,:);
x_train = [];
y_train = [];

% Data 7 günlük olarak hazýrlanýyor
n1 = 7;
n2 = n1-1;
for c= n1:num_train
   x_train(c-n2,1:n1) =  traindata(c-n2:c);
   y_train(c-n2,1) = traindata(c);
end
% Test Datasý hazýrlanýyor
testdata=scaled_data(num_train-n1:end,:);
x_test = [];
for c= n1:length(testdata)
   x_test(c-n2,1:n1) =  testdata(c-n2:c);
end
y_test = data(num_train-1:end);
% Diðer fonksiyonlarla uyuþmasý için eþitlemeler yapýlýyor
traininput=x_train';
traintarget=y_train';
testinput=x_test';
testtarget=y_test';
inputs = traininput; %[0 0;0 1;1 0;1 1]';
targets = traintarget; %[0;0;0;1]';

% nöron sayýmýzý setliyoruz
n = 7;
% Sinir aðý oluþturuluyor
net = feedforwardnet(n);
% Datalar veriliyor
net = configure(net, inputs, targets);
% Aðýrlýk ve biaslarý getiriliyor
getwb(net)
% Sinir aðýnýn hatalarý hesaplanýyor
error = targets - net(inputs);
calc = mean(error.^2)/mean(var(targets',1))
% Hesaplama fonksiyonu hazýrlanýyor
% MSE hesaplanýyor
h = @(x) NMSE(x, net, inputs, targets);
% Setting the Genetic Algorithms tolerance for
% minimum change in fitness function before
% terminating algorithm to 1e-8 and displaying
% each iteration's results.
% GAnýn tolerans ayarlamasý ve sonuçlarý göstermesi için ayarlama yapýlýyor
ga_opts = gaoptimset('PopInitRange', [-1;1], 'TolFun', 1e-10,'display','iter');
ga_opts = gaoptimset(ga_opts, 'StallGenLimit', 100, 'FitnessLimit', 1e-5, 'Generations', 10);

m=length(inputs(:,1));
o=length(targets(:,1));
% 7 nöron için , aðýrlýk ve bias için m*n+n+n+o formulu kullanýlýyor
% a. n for the input weights=(features*n)=m*n
% b. n for the input biases=(n bias)=n
% c. n for the output weights=(n weights)=n
% d. o for the output bias=(1 bias)=o
[x, err_ga] = ga(h, m*n+n+n+o,ga_opts);

disp(x);
disp(err_ga);
%Newnetwork=x;
% Yenibir network oluþturuluyor
net = setwb(net, x');
% Aðýrlýk ve biaslarý getiriliyor
getwb(net)
% Sinir Aðýnýn hatalarý hesaplanýyor
error = targets - net(inputs);
calc = mean(error.^2)/mean(var(targets',1))
% Eðitim datasý çýkartýlýyor
sinavtr=net(inputs);%sim(enetwork,traininput);
hatatr=error;%(sinavtr-traintarget);
% Test datasý orginal haline getiriliyor
sinavts=net(testinput);%sim(enetwork,testinput);
sinavts = mapminmax('reverse',sinavts',PS);
sinavts = sinavts';
% Test datasýný hatasý hesaplanýyor
hatats=(sinavts-testtarget);

%% Grafikler
figure;
subplot(2,2,[1 2])
plot(traintarget,'b')
hold on
plot(sinavtr,'r')
legend('gercek','ysadan gelen sonuc')
title('Train Data')
subplot(2,2,3)
plot(hatatr)
legend('hatalar')
subplot(2,2,4)
histfit(hatatr)
legend('histogram')
figure;
subplot(2,2,[1 2])
plot(testtarget,'b')
hold on
plot(sinavts,'r')
legend('gercek','ysadan gelen sonuc')
title('Test Data')
subplot(2,2,3)
plot(hatats)
legend('hatalar')
subplot(2,2,4)
histfit(hatats)
legend('histogram')
