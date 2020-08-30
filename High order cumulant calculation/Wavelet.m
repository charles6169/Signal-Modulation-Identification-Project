function y=Wavelet(x)
a =cwt(x,25,'haar');   %coefs中存放的是小波系数
%不知道是否应该加上下面这两句话？
% a2s=hilbert(a);               %hilbert希尔伯特变换
% fuzhi=medfilt1(abs(a2s),2);  %瞬时幅值中值滤波  
% y=fuzhi;
y=a;

end