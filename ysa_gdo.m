% Temizleme işlemi yapılıp veriler okunup hazırlanıyor
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
% Veriler 7 günlük haline getiriliyor
n1 = 7;
n2 = n1-1;
for c= n1:num_train
   x_train(c-n2,1:n1) =  traindata(c-n2:c);
   y_train(c-n2,1) = traindata(c);
end
% Test datası hazırlanıyor
testdata=scaled_data(num_train-n1:end,:);
x_test = [];
for c= n1:length(testdata)
   x_test(c-n2,1:n1) =  testdata(c-n2:c);
end
y_test = data(num_train-1:end);
% Transposeları alınıyor
traininput=x_train';
% traininput = mapminmax(traininput);
traintarget=y_train';
% [traintarget ps] = mapminmax(traintarget) 
testinput=x_test';
% testinput = mapminmax(testinput);
testtarget=y_test';
%mapminmax('reverse',data_scaled(1:5),PS)

% 7 katmanlı bir noron oluşturuluyor
layers=[7];
%transferfun={'tansig','purelin'};
Network=feedforwardnet(layers,'traingd');
% Configurasyonları yapılıyor

% Network.trainParam.lr = 0.01;
Network.trainParam.epochs = 200;
% Network.trainParam.goal = 1e-20; % belirli bir yakınsamada dur 
% net.trainParam.show = 50;
% net.trainParam.lr = 0.05;
% net.trainParam.epochs = 200;
% net.trainParam.goal = 1e-5;

% Optimizasyon Çalıştırılıyor
[enetwork,tr] = train(Network,traininput,traintarget)
% Veriler hazırlanıp hataları hesaplanıyor
sinavtr=sim(enetwork,traininput);
% sinavtr = mapminmax('reverse',sinavtr',PS);
% sinavtr = sinavtr';
hatatr=(sinavtr-traintarget);
sinavts=sim(enetwork,testinput);
% sinavts = mapminmax('reverse', sinavts , ps) %denormalizasyon
% Test datası eski haline getiriliyor
sinavts = mapminmax('reverse',sinavts',PS);
sinavts = sinavts';
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
