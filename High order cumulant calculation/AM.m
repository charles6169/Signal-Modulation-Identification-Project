function y=AM(snr_in)
t = 0:0.0001:1;
A0 = 10;
A1 = 4;
A2 = 1;
f = 3000;
w0 = 2*f*pi;
m = 0.15;
mes = A1*cos(0.001*w0*t);
Uam = A2*(1+m*mes).*cos((w0).*t);
sn1 = snr_in;
db1 = A2^2/(2*(10^(sn1/10)));
n1 = sqrt(db1)*randn(size(t));
Uam = Uam+n1;
Y3 = fft(Uam);
Ft = 2000;
fpts = [100 120];
mag = [1 0];
dev = [0.01 0.05];
[n21,wn21,beta,ftype] = kaiserord(fpts,mag,dev,Ft);
b21 = fir1(n21,wn21,kaiser(n21+1,beta));
% [h,w] = freqz(b21,1);
y = Uam;
end


