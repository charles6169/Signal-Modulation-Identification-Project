function y=ask_2(snr_in);
%载波频率20000，采样频率100000，码元个数100，每个码元中点数为1000；
sn=snr_in;
    s=round(rand(1,100));    %产生数值为0或1的随机矩阵
    t=0:1/99999:1;
    fc=20000;
    m1=[];
    for n=1:length(s)
        if s(n)==0;
            m=zeros(1,1000);
        else 
            m=ones(1,1000);
        end 
        m1=[m1 m];
    end
    c=sin(2*pi*fc*t);
    a=c.*m1;
y=awgn(a,sn);                %awgn（x，SNR）是在信号x中加入信噪比为SNR的高斯白噪声

end

