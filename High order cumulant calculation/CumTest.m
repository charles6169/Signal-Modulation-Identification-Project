% clc
% clear all
% close all
% n = 10000;                                   % ������е�Ԫ��
% u = 6;                                      % ������еľ�ֵ
% a = 3;                                      % ������еľ�����
%  
% x = random('norm',u,a,1,n);                 % ����һ��1*200��С�ľ�ֵΪ3������0.5�ĸ�˹�ֲ��������
% t = (1-a)*u:0.1:(1+a)*u;
% fn = normpdf(t,u,a);                        % �����ֵ3������0.5�ĸ�˹�ֲ��ܶȺ���
%  
% % figure(1)
% % subplot(3,1,1)
% % plot(x)                                     % �������ɵ��������x
% % subplot(3,1,2)
% % hist(x,20)                                  % ͳ�����ɵ��������x
% % xlim([min(t),max(t)])
% % subplot(3,1,3)
% % plot(t,fn)                                  % ������Ӧ�ĸ����ܶ�����
% % xlim([min(t),max(t)])
%  
% % �߽��ۻ����Ը�˹���������ä�ģ��ʶ��Ľ��ۻ�������Ӧ����0
% xCum4 = myCUM4EST (x, 50, 1000, 50, 'biased', 10, 10);
% xCum4Size = size(xCum4);
% % y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)  ���Ľ�ͳ����
% % y              : ������������(��)
% % maxlag         : �����Ƭ������Ƶ��w���гɲ�ͬ���Ӵ���������Ƶ�� = �Ӵ���*2+1��
% % �ֶ���߽��ۻ�����Ȼ��Խ����������
% % samp _ seg     : ÿ��������
% % overlap        : �ֶ�ʱ������ص�����ռÿ���ֶγ��ȵİٷֱ�
% % flag           : 'biased'��������ƫ���Ĺ���
% %                : ��unbiased����������ƫ���ơ�
% % k1��k2         : C3 ( m��k1 )��C4 ( m��k1��k2 )�еĹ̶��ͺ󣻼�����
% % y _ cum        : ���Ƶ��Ľ��ۻ�����Ƭ
% % C4 ( m��k1��k2 ) -����ͺ�< = m < =����ͺ�
% % ע��           : ����ָ�����в���
%  
% % �����ۻ�����Ӧ��˹�ֲ��ķ���a^2��������w=0����Ҳ���ǳ�����ͼ�к���ĵ� maxlag + 1 ���㴦
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
