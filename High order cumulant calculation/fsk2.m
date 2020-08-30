function[s] = fsk2(d,fb,f1,f2,fs,Ac)
%clear;d=randint(1,10,2);fb=500;f1=500;f2=2000;Ac=1;fs=12000;
N = length(d);%N=size(d,2);%N = number of data bits.
M = fs/fb;%M = number of samples per bit duration.

s = zeros(1,N*M);
for j = 1:N
	if d(j) == 1
	   for i = 1:M
		s((j-1)*M+i) = Ac*cos(2*pi*f1*(i-1)/fs)+1i*Ac*sin(2*pi*f1*(i-1)/fs);
	   end
	else
	   for i = 1:M
        s((j-1)*M+i) = Ac*cos(2*pi*f2*(i-1)/fs)+1i*Ac*sin(2*pi*f2*(i-1)/fs);
	   end
	end
end