function y=FM(snr_in);
dt = 0.001;
t = 0:dt:1.5;
sn1 = snr_in;
am = 1;
fm = 5;
mt = am*cos(2*pi*fm*t);
fc = 50;
ct = cos(2*pi*fc*t);
kf = 10;
int_mt(1) = 0;
for i=1:length(t)-1
    int_mt(i+1)=int_mt(i)+mt(i)*dt;
end
sfm=am*cos(2*pi*fc*t-2*pi*kf*int_mt);
%解调
for i=2:length(t)-1
    diff_nsfm(i)=(sfm(i-1)-sfm(i))./dt;
end
diff_nsfmn = abs(hilbert(diff_nsfm));
zero = (max(diff_nsfm)-min(diff_nsfm))/2;
diff_nsfmn1 = diff_nsfmn-zero;
%时域到频域转换
ts = 0.001;
fs = 1/ts;
df = 0.25;

m = am*cos(2*pi*fm*t);
fs = 1/ts;

n1 = fs/df;

n2 = length(m);
n = 2^(max(nextpow2(n1),nextpow2(n2)));
M = fft(m,n);
m = [m,zeros(1,n-n2)];
df1 = fs/n;
M = M/fs;
f = [0:df1:df1*(length(m)-1)]-fs/2;
%对已调信号求傅里叶变换
fs = 1/ts;

n1 = fs/df;
n2 = length(sfm);
n = 2^(max(nextpow2(n1),nextpow2(n2)));
U = fft(sfm,n);
u = [sfm,zeros(1,n-n2)];
df1 = fs/n;
nsfm = sfm;
for i=1:length(t)-1
    diff_nsfm(i)=(nsfm(i+1)-nsfm(i))./dt;
end
diff_nsfm = abs(hilbert(diff_nsfm));
zero = (max(diff_nsfm)-min(diff_nsfm))/2;
diff_nsfm1 = diff_nsfmn-zero;

db1 = am^2/(2*(10^(sn1/10)));
n1 = sqrt(db1)*randn(size(t));
nsfm1 = n1+sfm;
for i=1:length(t)-1
    diff_nsfm(i)=(nsfm1(i+1)-nsfm1(i))./dt;
end
diff_nsfmn1 = abs(hilbert(diff_nsfm1));
zero = (max(diff_nsfm1)-min(diff_nsfm1))/2;
diff_nsfmn1 = diff_nsfmn1-zero;
y = nsfm1
end