function y=psk_2(snr_in);
sn=snr_in;
%�ز�Ƶ��20000������Ƶ�� 100000��ÿ�����Ҳ���5���㡣��Ԫ100����ÿ����Ԫ������1000��
s=round(rand(1,100));    %������ֵΪ0��1���������
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
