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
nLoop = 10; % 循环次数

%训练数据：1200个
%测试数据：300个
SigNum=1200;%生成信号的个数

CellC20=cell(1,SigNum);
CellC21=cell(1,SigNum);
CellC40=cell(1,SigNum);
CellC41=cell(1,SigNum);
CellC42=cell(1,SigNum);
CellC60=cell(1,SigNum);

t1=clock;
for m=1:SigNum
    disp('信号：')
    disp(m)
    saveC20 = zeros(length(min:delta:max),8);                    %一共八种形式,不明白为什么是八种形式
    saveC21 = zeros(length(min:delta:max),8);
    saveC40 = zeros(length(min:delta:max),8);
    saveC41 = zeros(length(min:delta:max),8);
    saveC42 = zeros(length(min:delta:max),8);
    saveC60 = zeros(length(min:delta:max),8);

    iSNR = 0;
    for i=min:delta:max
        iSNR = iSNR+1;
%         disp('信噪比')
%         disp(i)
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
    CellC20{1,m}=saveC20;
    CellC21{1,m}=saveC21;
    CellC40{1,m}=saveC40;
    CellC41{1,m}=saveC41;
    CellC42{1,m}=saveC42;
    CellC60{1,m}=saveC60;
    
    
end
t2=clock;
etime(t2,t1);



CellT3=cell(1,SigNum);
CellT5=cell(1,SigNum);


CellASK2=cell(1,SigNum);
CellASK4=cell(1,SigNum);
CellFSK2=cell(1,SigNum);
CellFSK4=cell(1,SigNum);
CellPSK2=cell(1,SigNum);
CellPSK4=cell(1,SigNum);

CellSNR=cell((max-min+1),6);



for i=1:SigNum
    for j=1:(max-min+1)
                
        
%         abs(saveC40(:,1))./abs(saveC42(:,1))
        CellT3{1,i}(j,1)=abs(CellC40{1,i}(j,1))./abs(CellC42{1,i}(j,1));
        CellT3{1,i}(j,2)=abs(CellC40{1,i}(j,2))./abs(CellC42{1,i}(j,2));
        CellT3{1,i}(j,3)=abs(CellC40{1,i}(j,3))./abs(CellC42{1,i}(j,3));
        CellT3{1,i}(j,4)=abs(CellC40{1,i}(j,4))./abs(CellC42{1,i}(j,4));
        CellT3{1,i}(j,5)=abs(CellC40{1,i}(j,5))./abs(CellC42{1,i}(j,5));
        CellT3{1,i}(j,6)=abs(CellC40{1,i}(j,6))./abs(CellC42{1,i}(j,6));
        
        
%         abs(saveC41(:,1))./(abs(saveC42(:,1)).*abs(saveC42(:,1)))
        CellT5{1,i}(j,1)=abs(CellC41{1,i}(j,1))./(abs(CellC42{1,i}(j,1)).*abs(CellC42{1,i}(j,1)));
        CellT5{1,i}(j,2)=abs(CellC41{1,i}(j,2))./(abs(CellC42{1,i}(j,2)).*abs(CellC42{1,i}(j,2)));
        CellT5{1,i}(j,3)=abs(CellC41{1,i}(j,3))./(abs(CellC42{1,i}(j,3)).*abs(CellC42{1,i}(j,3)));
        CellT5{1,i}(j,4)=abs(CellC41{1,i}(j,4))./(abs(CellC42{1,i}(j,4)).*abs(CellC42{1,i}(j,4)));
        CellT5{1,i}(j,5)=abs(CellC41{1,i}(j,5))./(abs(CellC42{1,i}(j,5)).*abs(CellC42{1,i}(j,5)));
        CellT5{1,i}(j,6)=abs(CellC41{1,i}(j,6))./(abs(CellC42{1,i}(j,6)).*abs(CellC42{1,i}(j,6)));


      
        CellASK2{1,i}(j,1)=CellT3{1,i}(j,1);
        CellASK2{1,i}(j,2)=CellT5{1,i}(j,1);
        CellASK2{1,i}(j,3)=CellC20{1,i}(j,1);
        CellASK2{1,i}(j,4)=CellC41{1,i}(j,1);
        CellASK2{1,i}(j,5)=CellC60{1,i}(j,1);
        CellASK2{1,i}(j,6)=1;
        
        CellASK4{1,i}(j,1)=CellT3{1,i}(j,2);
        CellASK4{1,i}(j,2)=CellT5{1,i}(j,2);
        CellASK4{1,i}(j,3)=CellC20{1,i}(j,2);
        CellASK4{1,i}(j,4)=CellC41{1,i}(j,2);
        CellASK4{1,i}(j,5)=CellC60{1,i}(j,2);
        CellASK4{1,i}(j,6)=2;
        
        CellFSK2{1,i}(j,1)=CellT3{1,i}(j,3);
        CellFSK2{1,i}(j,2)=CellT5{1,i}(j,3);
        CellFSK2{1,i}(j,3)=CellC20{1,i}(j,3);
        CellFSK2{1,i}(j,4)=CellC41{1,i}(j,3);
        CellFSK2{1,i}(j,5)=CellC60{1,i}(j,3);
        CellFSK2{1,i}(j,6)=3;
        
        CellFSK4{1,i}(j,1)=CellT3{1,i}(j,4);
        CellFSK4{1,i}(j,2)=CellT5{1,i}(j,4);
        CellFSK4{1,i}(j,3)=CellC20{1,i}(j,4);
        CellFSK4{1,i}(j,4)=CellC41{1,i}(j,4);
        CellFSK4{1,i}(j,5)=CellC60{1,i}(j,4);
        CellFSK4{1,i}(j,6)=4;
        
        
        
        CellPSK2{1,i}(j,1)=CellT3{1,i}(j,5);
        CellPSK2{1,i}(j,2)=CellT5{1,i}(j,5);
        CellPSK2{1,i}(j,3)=CellC20{1,i}(j,5);
        CellPSK2{1,i}(j,4)=CellC41{1,i}(j,5);
        CellPSK2{1,i}(j,5)=CellC60{1,i}(j,5);
        CellPSK2{1,i}(j,6)=5;
        
        CellPSK4{1,i}(j,1)=CellT3{1,i}(j,6);
        CellPSK4{1,i}(j,2)=CellT5{1,i}(j,6);
        CellPSK4{1,i}(j,3)=CellC20{1,i}(j,6);
        CellPSK4{1,i}(j,4)=CellC41{1,i}(j,6);
        CellPSK4{1,i}(j,5)=CellC60{1,i}(j,6);
        CellPSK4{1,i}(j,6)=6;    
        
       
        
        

       
%         CellASK2=[CellT2{1,i}(j,1),CellT3{1,i}(j,1),CellT4{1,i}(j,1),CellT5{1,i}(j,1),CellF1{1,i}(j,1)]
        
    end
end


for i=1:SigNum
    for j=1:(max-min+1)
        CellSNR{j,1}(i,:)=CellASK2{1,i}(j,:);
        CellSNR{j,2}(i,:)=CellASK4{1,i}(j,:);
        CellSNR{j,3}(i,:)=CellFSK2{1,i}(j,:);
        CellSNR{j,4}(i,:)=CellFSK4{1,i}(j,:);
        CellSNR{j,5}(i,:)=CellPSK2{1,i}(j,:);
        CellSNR{j,6}(i,:)=CellPSK4{1,i}(j,:);
    end
end



% CellCSV为不同信噪比下计算的数据，格式为6*SigNum*8 
CellCSV=cell(20,1);
for j=1:(max-min+1)
    temp=1;
    for i=1:SigNum
        CellCSV{j,1}(temp,:)=CellSNR{j,1}(i,:);
        temp=temp+1;
        CellCSV{j,1}(temp,:)=CellSNR{j,2}(i,:);
        temp=temp+1;
        CellCSV{j,1}(temp,:)=CellSNR{j,3}(i,:);
        temp=temp+1;
        CellCSV{j,1}(temp,:)=CellSNR{j,4}(i,:);
        temp=temp+1;
        CellCSV{j,1}(temp,:)=CellSNR{j,5}(i,:);
        temp=temp+1;
        CellCSV{j,1}(temp,:)=CellSNR{j,6}(i,:);
        temp=temp+1;
    end
end

Matlabtrainpath='D:\1_MatlabDev\WorkSpace\ModRecogWorkSpace\Wavelet-OrderCumulants-NetworkWorkSpace\data\train\';
Matlabtestpath='D:\1_MatlabDev\WorkSpace\ModRecogWorkSpace\Wavelet-OrderCumulants-NetworkWorkSpace\data\test\';
% 注意修改文件夹
pythonpath='D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\data\HighOrderCumData\experiment1200_300';

% 输出模式
mode=1;
if mode==1
    for i=1:(max-min+1)
        temp1=['train',num2str(i)];
        temp=fullfile(pythonpath,temp1);
        xlswrite(temp,CellCSV{i,1});
    end
else
    for i=1:(max-min+1)
        temp1=['test',num2str(i)];
        temp=fullfile(pythonpath,temp1);
        xlswrite(temp,CellCSV{i,1});
    end
end
    


save 1
