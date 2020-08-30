function y=psk_2(snr_in);
sn=snr_in;
%载波频率20000；采样频率 100000，每个正弦波有5个点。码元100个，每个码元采数据1000个
s=round(rand(1,100));    %产生数值为0或1的随机矩阵
t=0:1/99999:1;
fc=20000;
m1=[];c1=[];
for n=1:length(s)
    if s(n)==0;
        m=-ones(1,1000);
    else 
        m=ones(1,1000);
    end  
    m1=[m1 m];
end
c=sin(2*pi*fc*t);     
a=c.*m1;
y=awgn(a,sn);  
end
