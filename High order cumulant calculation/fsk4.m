%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[s] =fsk4(d,fb,f1,f2,f3,f4,fs,Ac)
%clear;d=randint(1,10,4);fb=500;f1=500;f2=1000;f3=1500;f4=2000;Ac=1;fs=12000;
N = length(d);
M = fs/fb;
tb = 1/fb;
t1 = 1/f1;
t2 = 1/f2;
t3 = 1/f3;
t4 = 1/f4;
N1 = M*t1/tb;
N2 = M*t2/tb;
N3 = M*t3/tb;
N4 = M*t4/tb;
s = zeros(1,N*M);
for j = 1:N
    if d(j)==0
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/N1)+1i*Ac*sin(2*pi*(i-1)/N1);
        end
    elseif d(j)==1
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/N2)+1i*Ac*sin(2*pi*(i-1)/N2);
        end
    elseif d(j)==2
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/N3)+1i*Ac*sin(2*pi*(i-1)/N3);
        end
    else
        for i = 1:M
            s((j-1)*M+i) = Ac*cos(2*pi*(i-1)/N4)+1i*Ac*sin(2*pi*(i-1)/N4);
        end
    end
end