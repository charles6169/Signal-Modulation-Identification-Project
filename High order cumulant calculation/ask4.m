%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[s] =ask4(d,fb,fc,fs,Ac)
%clear;d=randint(1,10,4);fb=500;fc=2000;fs=12000;Ac=1;
N = length(d);
M = fs/fb;
Nc = fs/fc;
s = zeros(1,N*M);
unit = sqrt(2/7);
for j = 1:N
    for i = 1:M
        s((j-1)*M+i) = d(j)*Ac*cos(2*pi*(i-1)/Nc)*unit+1i*d(j)*Ac*sin(2*pi*(i-1)/Nc)*unit;
    end
end