function y=fsk_2(snr_in);
sn=snr_in;
%��Ԫ100����ÿ����Ԫȡ��1000����ÿ����Ԫ��10�����Ҳ���ÿ�����Ҳ�ȡ100����
s=round(rand(1,100));    %������ֵΪ0��1���������
t=0:1/999:1;             
fc0=10000;fc1=20000;     %�ز�Ƶ��20000���ң�����Ƶ��1000
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

