function y=ThreeWavelet(x)
% N = length(x);
[c,l]=wavedec(x,3,'sym4');                     %三层小波变换99
[thr,sorh,keepapp]=ddencmp('den','wv',x);    % 获取信号默认值（'den'表示去噪，'wv'表示小波）
s2=wdencmp('gbl',c,l,'sym4',3,thr,sorh,keepapp);   %去噪
aa =cwt(s2,25,'haar');   %coefs中存放的是小波系数
s1=hilbert(aa);           %hilbert希尔伯特变换

y=s1;
end