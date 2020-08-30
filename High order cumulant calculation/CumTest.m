% clc
% clear all
% close all
% n = 10000;                                   % 随机数列的元数
% u = 6;                                      % 随机序列的均值
% a = 3;                                      % 随机序列的均方差
%  
% x = random('norm',u,a,1,n);                 % 产生一个1*200大小的均值为3，方差0.5的高斯分布随机序列
% t = (1-a)*u:0.1:(1+a)*u;
% fn = normpdf(t,u,a);                        % 计算均值3，方差0.5的高斯分布密度函数
%  
% % figure(1)
% % subplot(3,1,1)
% % plot(x)                                     % 绘制生成的随机序列x
% % subplot(3,1,2)
% % hist(x,20)                                  % 统计生成的随机序列x
% % xlim([min(t),max(t)])
% % subplot(3,1,3)
% % plot(t,fn)                                  % 绘制相应的概率密度曲线
% % xlim([min(t),max(t)])
%  
% % 高阶累积量对高斯随机过程是盲的，故而四阶累积量期望应该是0
% xCum4 = myCUM4EST (x, 50, 1000, 50, 'biased', 10, 10);
% xCum4Size = size(xCum4);
% % y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)  求四阶统计量
% % y              : 输入数据向量(列)
% % maxlag         : 最大切片数（将频域w分切成不同的子带，完整的频带 = 子带数*2+1）
% % 分段求高阶累积量，然后对结果求期望？
% % samp _ seg     : 每段样本数
% % overlap        : 分段时候，设计重叠部分占每个分段长度的百分比
% % flag           : 'biased'，计算有偏见的估计
% %                : “unbiased”，计算无偏估计。
% % k1，k2         : C3 ( m，k1 )或C4 ( m，k1，k2 )中的固定滞后；见下文
% % y _ cum        : 估计的四阶累积量切片
% % C4 ( m，k1，k2 ) -最大滞后< = m < =最大滞后
% % 注意           : 必须指定所有参数
%  
% % 二阶累积量对应高斯分布的方差a^2，出现在w=0处，也就是出现在图中横轴的第 maxlag + 1 个点处
% % xCum2 = cum2est (x, 50, 30, 50, 'unbiased');
% % xCum2Size = size(xCum2);
% % y_cum = cum2est (y, maxlag, nsamp, overlap, flag)
% %CUM2EST Covariance function.
% %	Should be involed via "CUMEST" for proper parameter checks.
% %	y_cum = cum2est (y, maxlag, samp_seg, overlap,  flag)
%  
% %	       y: input data vector (column)
% %	  maxlag: maximum lag to be computed
% %	samp_seg: samples per segment (<=0 means no segmentation)
% %	 overlap: percentage overlap of segments
% %	    flag: 'biased', biased estimates are computed
% %	          'unbiased', unbiased estimates are computed.
% %	   y_cum: estimated covariance,
% %	          C2(m)  -maxlag <= m <= maxlag
% %	all parameters must be specified!
%  
% figure(2)
% subplot(4,1,2)
% plot(xCum2)
% subplot(4,1,4)
% plot(xCum4)
