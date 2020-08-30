function y=fsk_4(snr_in);
sn=0;
%载波频率20000，采样频率100000，码元100个，每个码元内点数1000
s=rand(1,100);    %产生数值为0或1的随机矩阵
f1=12500; 
f2=15000; 
f3=17500; 
f4=20000;
t=0:1/999:1;
m1=[];
c1=[];
for i=1:length(s)      %根据4FSK 取4种不同情况
    if(s(i)<0.25)
        m=ones(1,1000);
        c=sin(2*pi*f1*t);
    elseif(s(i)<0.5)      
        m=ones(1,1000);
        c=sin(2*pi*f2*t);
    elseif(s(i)<0.75)     
        m=ones(1,1000);
        c=sin(2*pi*f3*t);
    else     
        m=ones(1,1000);
        c=sin(2*pi*f4*t);
    end
    m1=[m1 m];
    c1=[c1 c];
end
a=c1.*m1; 
y=awgn(a,sn);  
end
