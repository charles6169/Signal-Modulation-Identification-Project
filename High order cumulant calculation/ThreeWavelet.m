function y=ThreeWavelet(x)
% N = length(x);
[c,l]=wavedec(x,3,'sym4');                     %����С���任99
[thr,sorh,keepapp]=ddencmp('den','wv',x);    % ��ȡ�ź�Ĭ��ֵ��'den'��ʾȥ�룬'wv'��ʾС����
s2=wdencmp('gbl',c,l,'sym4',3,thr,sorh,keepapp);   %ȥ��
aa =cwt(s2,25,'haar');   %coefs�д�ŵ���С��ϵ��
s1=hilbert(aa);           %hilbertϣ�����ر任

y=s1;
end