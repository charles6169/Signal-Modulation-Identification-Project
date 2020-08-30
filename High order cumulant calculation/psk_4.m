function y=psk_4(snr_in);
sn=snr_in;
%载波频率20000；采样频率 100000，每个正弦波有5个点。码元100个，每个码元采数据1000个
s=rand(1,100);    %产生数值为0或1的随机矩阵
t=0:1/999:1;
fc=20000;
m1=[];c1=[];
s1=sin(5*t);
s2=sin(5*t+0.5*pi);
s3=sin(5*t+pi);
s4=sin(5*t+1.5*pi);
for i=1:length(s)      %根据4PSK 取4种不同情况
    if(s(i)<0.25)
        m=ones(1,1000);
        c=s1;
    elseif(s(i)<0.5)      
        m=ones(1,1000);
        c=s2;
    elseif(s(i)<0.75)     
        m=ones(1,1000);
        c=s3;
    else     
        m=ones(1,1000);
        c=s4;
    end
    m1=[m1 m];
    c1=[c1 c];
end
a=c1.*m1; 
y=awgn(a,sn);  
end

