function y=fsk_2(snr_in);
sn=snr_in;
%码元100个，每个码元取数1000个，每个码元有10个正弦波，每个正弦波取100个数
s=round(rand(1,100));    %产生数值为0或1的随机矩阵
t=0:1/999:1;             
fc0=10000;fc1=20000;     %载波频率20000左右，采样频率1000
m1=[];c1=[];
for n=1:length(s)
    if s(n)==0;
        m=ones(1,1000);
        c=sin(2*pi*fc0*t);      
    else
        m=ones(1,1000);
        c=sin(2*pi*fc1*t);     
    end
    m1=[m1 m];
    c1=[c1 c];
end 
a=c1.*m1;
y=awgn(a,sn);  
end

