% function y=QAM16(qam_work,bits_in,snr)
% Nbin = 192;
% NOFDM = 10;
% data=randi([0,1],NOFDM,Nbin*NOFDM); %基带信号
% M=16;
% y1=modulate(qammod(data,M));
% symbol = round(rand(1,100)*15);
% qam_symbol = qammod(symbol,16);
% y1=qam_symbol;
% if qam_work == 1
%    full_len = length(bits_in);
%    m=1;
%    for k=-3:2:3
%       for l=-3:2:3
%          table(m) = (k+j*l)/sqrt(10); % power normalization
%          m=m+1;
%       end
%    end
%    table=table([0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10]+1); % Gray code mapping pattern for 8-PSK symbols
%    inp=reshape(bits_in,4,full_len/4);
%    qammod=table([8 4 2 1]*inp+1);  % maps transmitted bits into 16QAM symbols
% 
% y1=qammod;
% snr=snr+10*log10(4);
% y=awgn(y1,snr,'measured');
% end

function y=QAM16(snr)
x=randi([0,1],1,10000); %基带信号
% y1=modulate(qammod('M',16,'InputType','Bit'),x);
M=16;

snr=snr+10*log10(4);
y=awgn(y1,snr,'measured');
end

