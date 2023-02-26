
function network3=creatnet(x,network)
% Deðiþtirilen Bias ve Aðýrlýklar ile yeniden model oluþturulup 
% Hatalarý test ediliyor.
% Að oluþturuluyor
IW=network.IW{1,1}; IWnum=numel(IW);
LW=network.LW{2,1}; LWnum=numel(LW);

b1=network.b{1,1};  b1num=numel(b1);
b2=network.b{2,1};  b2num=numel(b2);

toplamparams=IWnum+LWnum+b1num+b2num;


IWs=x(1:IWnum); 
IWnew=reshape(IWs,size(IW,1),size(IW,2)); 
x=x(IWnum+1:end);

LWs=x(1:LWnum);  LWnew=reshape(LWs,size(LW,1),size(LW,2)); x=x(LWnum+1:end);

b1s=x(1:b1num); b1new=reshape(b1s,size(b1,1),size(b1,2)); x=x(b1num+1:end);
b2s=x(1:b2num); b2new=reshape(b2s,size(b2,1),size(b2,2)); 


network2=network;
network2.IW{1,1}=IWnew;

network2.LW{2,1}=LWnew;


network2.b{1,1}=b1new;
network2.b{2,1}=b2new;
network3=network2;


end