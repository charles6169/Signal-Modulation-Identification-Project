clc;
clear all;
%所有调制方信噪比>20dB的仿真可视化
%config
L=8192;  %显示长度
snr=0;  %信噪比

yAM=AM(snr);
yAM=yAM(1:L);

yFM=FM(snr);
yFM=yFM(1:1500);

yask_2=ask_2(snr);
yask_2=yask_2(1:L);

yask_4=ask_4(snr);
yask_4=yask_4(1:L);

yfsk_2=fsk_2(snr);
yfsk_2=yfsk_2(1:L);
yfsk_4=fsk_4(snr);
yfsk_4=yfsk_4(1:L);


ypsk_2=psk_2(snr);
ypsk_2=ypsk_2(1:L);
ypsk_4=psk_4(snr);
ypsk_4=ypsk_4(1:L);

yOFDM=OFDM(snr);
yOFDM=yOFDM(1:L);

%% 小波去噪前
figure(1);
subplot(3,3,1),plot(yAM),title("AM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,4),plot(yFM),title("FM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,2),plot(yask_2),title("2ASK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,3),plot(yask_4),title("4ASK"),xlabel("时间/s"),ylabel("幅度");

subplot(3,3,5),plot(yfsk_2),title("2FSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,6),plot(yfsk_4),title("4FSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,7),plot(yOFDM),title("OFDM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,8),plot(ypsk_2),title("2PSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,9),plot(ypsk_4),title("4PSK"),xlabel("时间/s"),ylabel("幅度");
suptitle('九种调制信号仿真（小波去噪前）,SNR:0') 

%% 小波变换

yAM_W=Wavelet(yAM);
y_FM_W=Wavelet(yFM);
yask_2_W=Wavelet(yask_2);
yask_4_W=Wavelet(yask_4);
yfsk_2_W=Wavelet(yfsk_2);
yfsk_4_W=Wavelet(yfsk_4);
yOFDM_W=Wavelet(yOFDM);
ypsk_2_W=Wavelet(ypsk_2);
ypsk_4_W=Wavelet(ypsk_4);
%% 一层小波去噪后
figure(2);
subplot(3,3,1),plot(yAM_W),title("AM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,4),plot(y_FM_W),title("FM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,2),plot(yask_2_W),title("2ASK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,3),plot(yask_2_W),title("4ASK"),xlabel("时间/s"),ylabel("幅度");

subplot(3,3,5),plot(yfsk_2_W),title("2FSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,6),plot(yfsk_4_W),title("4FSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,7),plot(yOFDM_W),title("OFDM"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,8),plot(ypsk_2_W),title("2PSK"),xlabel("时间/s"),ylabel("幅度");
subplot(3,3,9),plot(ypsk_4_W),title("4PSK"),xlabel("时间/s"),ylabel("幅度");
suptitle('九种调制信号仿真（小波去噪后）,SNR:0') 



%% 
% hh1=yAM_W;
% hh2=y_FM_W;
% hh3=yask_2_W;
% hh4=yask_2_W;
% hh5=yfsk_2_W;
% hh6=yfsk_4_W;
% % 二阶距 M20 = E[X(k)X(k)] M21 = E[X(k)X'(k)]
% m20 = [mean(hh1.*hh1) mean(hh2.*hh2) mean(hh3.*hh3) mean(hh4.*hh4) mean(hh5.*hh5) mean(hh6.*hh6) ];
% 
% 
% m21 = [mean(hh1.*conj(hh1)) mean(hh2.*conj(hh2)) mean(hh3.*conj(hh3)) mean(hh4.*conj(hh4)) mean(hh5.*conj(hh5)) mean(hh6.*conj(hh6))];

         
         
         
% xCum4=cum4est();




% yAM=ThreeWavelet(yAM);


% % figure(2);
% % X=fft(yask_2);
% % plot(X);
% % title('amplitude spectrum'),xlabel('frequency'),ylabel('amplitude')
% 
% 
% % yQAM16=QAM16(1,10000,20);
% yQAM16=QAM16(20);
% 
% % 
% % figure(2);
% % 
% yQAM16=yQAM16(1:L);
% plot(yQAM16)





