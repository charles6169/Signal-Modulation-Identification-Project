%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[s] = psk2(d,fb,fc,fs,Ac)
%clear; d=randint(1,10,2);fb=500;fc=2000;Ac=1;fs=12000;
% N = size(d,2);
N=length(d);
M = fs/fb;
tb = 1/fb;
tc = 1/fc;
Nc = M*tc/tb;
d = d*2-1;
s = zeros(1,N*M);
for j = 1:N
    for i = 1:M
        s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/Nc)*d(j)+1i*Ac*sin(2*pi*(i-1)/Nc)*d(j);
    end
end