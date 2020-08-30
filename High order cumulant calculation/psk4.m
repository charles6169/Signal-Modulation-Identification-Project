function[s] = psk4(d,fb,fc,fs,Ac)
%clear; d=randint(1,10,4);fb=500;fc=2000;Ac=1;fs=12000;
N = length(d);
M = fs/fb;
tb = 1/fb;
tc = 1/fc;
Nc = M*tc/tb;
s = zeros(1,N*M);
for j = 1:N
    if d(j)==0
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/Nc)+1i*Ac*sin(2*pi*(i-1)/Nc);
        end
    elseif d(j)==1
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/Nc+pi/2)+1i*Ac*sin(2*pi*(i-1)/Nc+pi/2);
        end
    elseif d(j)==2
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/Nc+pi)+1i*Ac*sin(2*pi*(i-1)/Nc+pi);
        end
    else
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/Nc-pi/2)+1i*Ac*sin(2*pi*(i-1)/Nc-pi/2);
        end
    end
end