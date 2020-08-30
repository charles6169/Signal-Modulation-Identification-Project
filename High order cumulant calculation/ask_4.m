function y=ask_4(snr_in);
sn=snr_in;
%载波频率20000，采样频率100000，码元个数100，每个码元中点数为1000；
 s=rand(1,100);    %产生数值为0或1的随机矩阵
 fc=20000;
    t=0:1/99999:1;
    m1=[];
    for i=1:length(s)                       %M=4，设置4种取值情况
        if(s(i)<0.25)
            m=zeros(1,1000);
        elseif(s(i)<0.5)
              m=ones(1,1000);
        elseif(s(i)<0.75)
              m=2*ones(1,1000);
        else
             m=3*ones(1,1000);
        end
        m1=[m1 m];
    end
    c=sin(2*pi*fc*t);   
    ASK4=c.*m1;
y=awgn(ASK4,sn);    
end
