clear
clc
fc=1.0e7;               %载波频率
fs=6e7;                 %采样频率
fd=2.5e6;               %符号速率 

Ac=1;
f1=5.5e6;
f2=8.5e6;
f3=11.5e6;
f4=14.5e6;
deltaF = 1.5e6;
min = -3;              %信噪比范围
max = 16;
delta = 1;
nSymb = 1000; % 每次发送符号数
nLoop = 50; % 循环次数
SigNum=10;

% CellC20=Cell(1,10);
% CellC21=Cell(1,10);
% CellC40=Cell(1,10);
% CellC41=Cell(1,10);
% CellC42=Cell(1,10);
% CellC60=Cell(1,10);


saveC20 = zeros(length(min:delta:max),8);                    %一共八种形式,不明白为什么是八种形式
saveC21 = zeros(length(min:delta:max),8);
saveC40 = zeros(length(min:delta:max),8);
saveC41 = zeros(length(min:delta:max),8);
saveC42 = zeros(length(min:delta:max),8);
saveC60 = zeros(length(min:delta:max),8);

iSNR = 0;
for i=min:delta:max
    iSNR = iSNR+1;
    disp('信噪比')
    disp(i)
    for k=1:nLoop
        % 随机产生原始信息
        d2 = randi(1,nSymb,2);
        d4 = randi(1,nSymb,4);
        
        % 调制
        s1 = ask2(d2,fd,fc,fs,Ac);
        s2 = ask4(d4,fd,fc,fs,Ac);
        s3 = fsk2(d2,fd,f2,f3,fs,Ac);
        s4 = fsk4(d4,fd,f1,f2,f3,f4,fs,Ac);
        s5 = psk2(d2,fd,fc,fs,Ac);
        s6 = psk4(d4,fd,fc,fs,Ac);
       
        
        
        % 加噪
        ss1 = awgn(s1,i,'measured');
        ss2 = awgn(s2,i,'measured');
        ss3 = awgn(s3,i,'measured');
        ss4 = awgn(s4,i,'measured');
        ss5 = awgn(s5,i,'measured');
        ss6 = awgn(s6,i,'measured');
    
% %         傅里叶变换
        ss1=fft(ss1);
        ss2=fft(ss2);
        ss3=fft(ss3);
        ss4=fft(ss4);
        ss5=fft(ss5);
        ss6=fft(ss6);
% 
%         %小波变换
%         scal=25;
%         ss1=cwt(ss1,scal,'haar'); 
%         ss2=cwt(ss2,scal,'haar'); 
%         ss3=cwt(ss3,scal,'haar'); 
%         ss4=cwt(ss4,scal,'haar'); 
%         ss5=cwt(ss5,scal,'haar'); 
%         ss6=cwt(ss6,scal,'haar'); 
%         

        %===========接收信号乘以载波频率，变为基带信号===========
        sigLocal = sig_gen(nSymb,fd,fc,fs);
        hh1 = ss1.*sigLocal; % 2ASK
        hh2 = ss2.*sigLocal; % 4ASK
        hh3 = ss3.*sigLocal; % 2FSK
        hhh3 = hh3.*sig_gen(nSymb,fd,deltaF,fs);
        hh4 = ss4.*sigLocal; % 4FSK
        hhh4 = hh4.*sig_gen(nSymb,fd,deltaF,fs);
        hh5 = ss5.*sigLocal; % 2PSK
        hh6 = ss6.*sigLocal; % 4PSK
  
           
        % 二阶距 M20 = E[X(k)X(k)] M21 = E[X(k)X'(k)]
        m20 = [mean(hh1.*hh1) mean(hh2.*hh2) mean(hh3.*hh3) mean(hh4.*hh4) mean(hh5.*hh5) mean(hh6.*hh6) ...
            mean(hhh3.*hhh3) mean(hhh4.*hhh4)];
        m21 = [mean(hh1.*conj(hh1)) mean(hh2.*conj(hh2)) mean(hh3.*conj(hh3)) mean(hh4.*conj(hh4)) mean(hh5.*conj(hh5)) mean(hh6.*conj(hh6))...
             mean(hhh3.*conj(hhh3)) mean(hhh4.*conj(hhh4))];

        % 四阶距 M40 = E[X(k)X(k)X(k)X(k)]  M41 = E[X(k)X(k)X(k)X'(k)]  M42 = E[X'(k)X(k)X(k)X'(k)]
        m40 = [mean(hh1.*hh1.*hh1.*hh1) mean(hh2.*hh2.*hh2.*hh2) mean(hh3.*hh3.*hh3.*hh3)...
            mean(hh4.*hh4.*hh4.*hh4) mean(hh5.*hh5.*hh5.*hh5) mean(hh6.*hh6.*hh6.*hh6) ...
             mean(hhh3.*hhh3.*hhh3.*hhh3) mean(hhh4.*hhh4.*hhh4.*hhh4)];
        m41 = [mean(hh1.*hh1.*hh1.*conj(hh1)) mean(hh2.*hh2.*hh2.*conj(hh2)) mean(hh3.*hh3.*hh3.*conj(hh3))...
            mean(hh4.*hh4.*hh4.*conj(hh4)) mean(hh5.*hh5.*hh5.*conj(hh5)) mean(hh6.*hh6.*hh6.*conj(hh6)) ...
            mean(hhh3.*hhh3.*hhh3.*conj(hhh3)) mean(hhh4.*hhh4.*hhh4.*conj(hhh4))];
        m42 = [mean(hh1.*hh1.*conj(hh1).*conj(hh1)) mean(hh2.*hh2.*conj(hh2).*conj(hh2)) mean(hh3.*hh3.*conj(hh3).*conj(hh3))...
            mean(hh4.*hh4.*conj(hh4).*conj(hh4)) mean(hh5.*hh5.*conj(hh5).*conj(hh5)) mean(hh6.*hh6.*conj(hh6).*conj(hh6)) ...
             mean(hhh3.*hhh3.*conj(hhh3).*conj(hhh3)) mean(hhh4.*hhh4.*conj(hhh4).*conj(hhh4))];

        % 六阶距 M60 = E[X(k)X(k)X(k)X(k)X(k)X(k)]
        m60 = [mean(hh1.*hh1.*hh1.*hh1.*hh1.*hh1) mean(hh2.*hh2.*hh2.*hh2.*hh2.*hh2) mean(hh3.*hh3.*hh3.*hh3.*hh3.*hh3)...
            mean(hh4.*hh4.*hh4.*hh4.*hh4.*hh4) mean(hh5.*hh5.*hh5.*hh5.*hh5.*hh5) mean(hh6.*hh6.*hh6.*hh6.*hh6.*hh6) ...
            mean(hhh3.*hh3.*hhh3.*hhh3.*hhh3.*hhh3) mean(hhh4.*hhh4.*hhh4.*hhh4.*hhh4.*hhh4)];

        %=========计算高阶累计量===========
        c20 = m20;
        c21 = m21;
        
        c40 = m40-3*m20.^2;
        c41 = m41-3*m21.*m20;
        c42 = m42-abs(m20).^2-2*m21.^2;

        c60 = m60-15.*m40.*m20+30.*m20.^3;

        saveC20(iSNR,:) = saveC20(iSNR,:)+c20;
        saveC21(iSNR,:) = saveC21(iSNR,:)+c21;
        saveC40(iSNR,:) = saveC40(iSNR,:)+c40;
        saveC41(iSNR,:) = saveC41(iSNR,:)+c41;
        saveC42(iSNR,:) = saveC42(iSNR,:)+c42;
        saveC60(iSNR,:) = saveC60(iSNR,:)+c60;
    end

    saveC20(iSNR,:) = saveC20(iSNR,:)/nLoop;
    saveC21(iSNR,:) = saveC21(iSNR,:)/nLoop;
    saveC40(iSNR,:) = saveC40(iSNR,:)/nLoop;
    saveC41(iSNR,:) = saveC41(iSNR,:)/nLoop;
    saveC42(iSNR,:) = saveC42(iSNR,:)/nLoop;
    saveC60(iSNR,:) = saveC60(iSNR,:)/nLoop;

end




%% 原高阶特征C20,C21,C40,C41,C42,C60 仿真
figure
plot(min:delta:max,saveC20(:,1),'-r+',...
    min:delta:max,saveC20(:,2),'-g*',...        
    min:delta:max,saveC20(:,3),'-bx',...
    min:delta:max,saveC20(:,4),'-cs',...
    min:delta:max,saveC20(:,5),'-md',...
    min:delta:max,saveC20(:,6),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('C20')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC20(:,1)),'-r+',...
    min:delta:max,abs(saveC20(:,2)),'-g*',...        
    min:delta:max,abs(saveC20(:,3)),'-bx',...
    min:delta:max,abs(saveC20(:,4)),'-cs',...
    min:delta:max,abs(saveC20(:,5)),'-md',...
    min:delta:max,abs(saveC20(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C20|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC21(:,1)),'-r+',...
    min:delta:max,abs(saveC21(:,2)),'-g*',...        
    min:delta:max,abs(saveC21(:,3)),'-bx',...
    min:delta:max,abs(saveC21(:,4)),'-cs',...
    min:delta:max,abs(saveC21(:,5)),'-md',...
    min:delta:max,abs(saveC21(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C21|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC40(:,1)),'-r+',...
    min:delta:max,abs(saveC40(:,2)),'-g*',...        
    min:delta:max,abs(saveC40(:,3)),'-bx',...
    min:delta:max,abs(saveC40(:,4)),'-cs',...
    min:delta:max,abs(saveC40(:,5)),'-md',...
    min:delta:max,abs(saveC40(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C40|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC41(:,1)),'-r+',...
    min:delta:max,abs(saveC41(:,2)),'-g*',...        
    min:delta:max,abs(saveC41(:,3)),'-bx',...
    min:delta:max,abs(saveC41(:,4)),'-cs',...
    min:delta:max,abs(saveC41(:,5)),'-md',...
    min:delta:max,abs(saveC41(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C41|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC42(:,1)),'-r+',...
    min:delta:max,abs(saveC42(:,2)),'-g*',...        
    min:delta:max,abs(saveC42(:,3)),'-bx',...
    min:delta:max,abs(saveC42(:,4)),'-cs',...
    min:delta:max,abs(saveC42(:,5)),'-md',...
    min:delta:max,abs(saveC42(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C42|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,abs(saveC60(:,1)),'-r+',...
    min:delta:max,abs(saveC60(:,2)),'-g*',...        
    min:delta:max,abs(saveC60(:,3)),'-bx',...
    min:delta:max,abs(saveC60(:,4)),'-cs',...
    min:delta:max,abs(saveC60(:,5)),'-md',...
    min:delta:max,abs(saveC60(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('|C60|')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

%% 唐师兄的fx1和fx2

figure
% |C40|/|C42| 得到两组 {2ask 4ask 4psk} {2fsk 4fsk}
plot(min:delta:max,abs(saveC40(:,1))./abs(saveC42(:,1)),'-r+',...
    min:delta:max,abs(saveC40(:,2))./abs(saveC42(:,2)),'-g*',...
    min:delta:max,abs(saveC40(:,3))./abs(saveC42(:,3)),'-bx',...
    min:delta:max,abs(saveC40(:,4))./abs(saveC42(:,4)),'-cs',...    
    min:delta:max,abs(saveC40(:,6))./abs(saveC42(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('fx1')
legend('2ASK','4ASK','2FSK','4FSK','4PSK');

figure
% |C41|/|C42| 得到两组 {2ask 4ask} {4psk}
plot(min:delta:max,abs(saveC41(:,1))./abs(saveC42(:,1)),'-r+',...
    min:delta:max,abs(saveC41(:,2))./abs(saveC42(:,2)),'-g*',...     
    min:delta:max,abs(saveC41(:,6))./abs(saveC42(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('fx2')
legend('2ASK','4ASK','4PSK');  

%% 通过比值构造的特征 仿真

figure
plot(min:delta:max,abs(saveC60(:,1))./(abs(saveC21(:,1)).*abs(saveC21(:,1)).*abs(saveC21(:,1))),'-r+',...
    min:delta:max,abs(saveC60(:,2))./(abs(saveC21(:,2)).*abs(saveC21(:,2)).*abs(saveC21(:,2))),'-g*',...
    min:delta:max,abs(saveC60(:,3))./(abs(saveC21(:,3)).*abs(saveC21(:,3)).*abs(saveC21(:,3))),'-bx',...
    min:delta:max,abs(saveC60(:,4))./(abs(saveC21(:,4)).*abs(saveC21(:,4)).*abs(saveC21(:,4))),'-cs',...
    min:delta:max,abs(saveC60(:,5))./(abs(saveC21(:,5)).*abs(saveC21(:,5)).*abs(saveC21(:,5))),'-md',...
    min:delta:max,abs(saveC60(:,6))./(abs(saveC21(:,6)).*abs(saveC21(:,6)).*abs(saveC21(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('T2')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')


figure
plot(min:delta:max,abs(saveC40(:,1))./abs(saveC42(:,1)),'-r+',...
    min:delta:max,abs(saveC40(:,2))./abs(saveC42(:,2)),'-g*',...
    min:delta:max,abs(saveC40(:,3))./abs(saveC42(:,3)),'-bx',...
    min:delta:max,abs(saveC40(:,4))./abs(saveC42(:,4)),'-cs',...
    min:delta:max,abs(saveC40(:,5))./abs(saveC42(:,5)),'-md',...
    min:delta:max,abs(saveC40(:,6))./abs(saveC42(:,6)),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('T3')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')

figure
plot(min:delta:max,(abs(saveC60(:,1)).*abs(saveC60(:,1)))./(abs(saveC21(:,1)).*abs(saveC21(:,1)).*abs(saveC21(:,1))),'-r+',...
    min:delta:max,(abs(saveC60(:,2)).*abs(saveC60(:,2)))./(abs(saveC21(:,2)).*abs(saveC21(:,2)).*abs(saveC21(:,2))),'-g*',...
    min:delta:max,(abs(saveC60(:,3)).*abs(saveC60(:,3)))./(abs(saveC21(:,3)).*abs(saveC21(:,3)).*abs(saveC21(:,3))),'-bx',...
    min:delta:max,(abs(saveC60(:,4)).*abs(saveC60(:,4)))./(abs(saveC21(:,4)).*abs(saveC21(:,4)).*abs(saveC21(:,4))),'-cs',...
    min:delta:max,(abs(saveC60(:,5)).*abs(saveC60(:,5)))./(abs(saveC21(:,5)).*abs(saveC21(:,5)).*abs(saveC21(:,5))),'-md',...
    min:delta:max,(abs(saveC60(:,6)).*abs(saveC60(:,6)))./(abs(saveC21(:,6)).*abs(saveC21(:,6)).*abs(saveC21(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('T4')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')


figure
plot(min:delta:max,abs(saveC41(:,1))./(abs(saveC42(:,1)).*abs(saveC42(:,1))) ,'-r+',...
    min:delta:max,abs(saveC41(:,2))./(abs(saveC42(:,2)).*abs(saveC42(:,2))),'-g*',...        
    min:delta:max,abs(saveC41(:,3))./(abs(saveC42(:,3)).*abs(saveC42(:,3))),'-bx',...
    min:delta:max,abs(saveC41(:,4))./(abs(saveC42(:,4)).*abs(saveC42(:,4))),'-cs',...
    min:delta:max,abs(saveC41(:,5))./(abs(saveC42(:,5)).*abs(saveC42(:,5))),'-md',...
    min:delta:max,abs(saveC41(:,6))./(abs(saveC42(:,6)).*abs(saveC42(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('T5')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')


%% 通过变换组合构造的特征 仿真

figure
plot(min:delta:max,(abs(saveC42(:,1))+abs(saveC40(:,1)))./(abs(saveC21(:,1)).*abs(saveC21(:,1))),'-r+',...
    min:delta:max,(abs(saveC42(:,2))+abs(saveC40(:,2)))./(abs(saveC21(:,2)).*abs(saveC21(:,2))),'-g*',...
    min:delta:max,(abs(saveC42(:,3))+abs(saveC40(:,3)))./(abs(saveC21(:,3)).*abs(saveC21(:,3))),'-bx',...
    min:delta:max,(abs(saveC42(:,4))+abs(saveC40(:,4)))./(abs(saveC21(:,4)).*abs(saveC21(:,4))),'-cs',...
    min:delta:max,(abs(saveC42(:,5))+abs(saveC40(:,5)))./(abs(saveC21(:,5)).*abs(saveC21(:,5))),'-md',...
    min:delta:max,(abs(saveC42(:,6))+abs(saveC40(:,6)))./(abs(saveC21(:,6)).*abs(saveC21(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('F1')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')




figure
plot(min:delta:max,(abs(saveC60(:,1)).*abs(saveC60(:,1)))./(abs(saveC40(:,1)).*abs(saveC40(:,1)).*abs(saveC40(:,1))),'-r+',...
    min:delta:max,(abs(saveC60(:,2)).*abs(saveC60(:,2)))./(abs(saveC40(:,2)).*abs(saveC40(:,2)).*abs(saveC40(:,2))),'-g*',...
    min:delta:max,(abs(saveC60(:,3)).*abs(saveC60(:,3)))./(abs(saveC40(:,3)).*abs(saveC40(:,3)).*abs(saveC40(:,3))),'-bx',...
    min:delta:max,(abs(saveC60(:,4)).*abs(saveC60(:,4)))./(abs(saveC40(:,4)).*abs(saveC40(:,4)).*abs(saveC40(:,4))),'-cs',...
    min:delta:max,(abs(saveC60(:,5)).*abs(saveC60(:,5)))./(abs(saveC40(:,5)).*abs(saveC40(:,5)).*abs(saveC40(:,5))),'-md',...
    min:delta:max,(abs(saveC60(:,6)).*abs(saveC60(:,6)))./(abs(saveC40(:,6)).*abs(saveC40(:,6)).*abs(saveC40(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('F3')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')


figure
plot(min:delta:max,(abs(saveC42(:,1))-abs(saveC40(:,1)))./(abs(saveC21(:,1)).*abs(saveC21(:,1))),'-r+',...
    min:delta:max,(abs(saveC42(:,2))-abs(saveC40(:,2)))./(abs(saveC21(:,2)).*abs(saveC21(:,2))),'-g*',...
    min:delta:max,(abs(saveC42(:,3))-abs(saveC40(:,3)))./(abs(saveC21(:,3)).*abs(saveC21(:,3))),'-bx',...
    min:delta:max,(abs(saveC42(:,4))-abs(saveC40(:,4)))./(abs(saveC21(:,4)).*abs(saveC21(:,4))),'-cs',...
    min:delta:max,(abs(saveC42(:,5))-abs(saveC40(:,5)))./(abs(saveC21(:,5)).*abs(saveC21(:,5))),'-md',...
    min:delta:max,(abs(saveC42(:,6))-abs(saveC40(:,6)))./(abs(saveC21(:,6)).*abs(saveC21(:,6))),'-y^'),grid on
xlabel('SNR(dB)')
ylabel('F4')
legend('2ASK','4ASK','2FSK','4FSK','2PSK','4PSK')






save 1
