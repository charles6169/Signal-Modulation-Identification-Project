function y=fsk_4(snr_in);
sn=0;
%�ز�Ƶ��20000������Ƶ��100000����Ԫ100����ÿ����Ԫ�ڵ���1000
s=rand(1,100);    %������ֵΪ0��1���������
f1=12500; 
f2=15000; 
f3=17500; 
f4=20000;
t=0:1/999:1;
m1=[];
c1=[];
for i=1:length(s)      %����4FSK ȡ4�ֲ�ͬ���
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
