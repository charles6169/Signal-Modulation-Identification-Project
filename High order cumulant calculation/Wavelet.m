function y=Wavelet(x)
a =cwt(x,25,'haar');   %coefs�д�ŵ���С��ϵ��
%��֪���Ƿ�Ӧ�ü������������仰��
% a2s=hilbert(a);               %hilbertϣ�����ر任
% fuzhi=medfilt1(abs(a2s),2);  %˲ʱ��ֵ��ֵ�˲�  
% y=fuzhi;
y=a;

end