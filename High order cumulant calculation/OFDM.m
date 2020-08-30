function y=OFDM(SNR_in)
IFFT_bin_length=1024;
carrier_count=200;
bits_per_symbol=2;
symbols_per_carrier=50;
SNR=SNR_in;
baseband_out_length=carrier_count*symbols_per_carrier*bits_per_symbol;
carriers=(1:carrier_count)+(floor(IFFT_bin_length/4)-floor(carrier_count/2));
conjugate_carriers=IFFT_bin_length-carriers+2;

baseband_out=round(rand(1,baseband_out_length));
convert_matrix=reshape(baseband_out,bits_per_symbol,length(baseband_out)/bits_per_symbol);
for k=1:(length(baseband_out)/bits_per_symbol)
    modulo_baseband(k)=0;
    for i=1:bits_per_symbol
        modulo_baseband(k)=modulo_baseband(k)+convert_matrix(i,k)*2^(bits_per_symbol-i);
    end
end
carrier_matrix=reshape(modulo_baseband,carrier_count,symbols_per_carrier)';
carrier_matrix=[zeros(1,carrier_count);carrier_matrix];
for i=2:(symbols_per_carrier+1)
    carrier_matrix(1,:)=rem(carrier_matrix(i,:)+carrier_matrix(i-1,:),2^bits_per_symbol);
end
carrier_matrix=carrier_matrix*((2*pi)/(2^bits_per_symbol));
[X,Y]=pol2cart(carrier_matrix,ones(size(carrier_matrix,1),size(carrier_matrix,2)));
complex_carrier_matrix=complex(X,Y);
IFFT_modulation=zeros(symbols_per_carrier+1,IFFT_bin_length);
IFFT_modulation(:,carriers)=complex_carrier_matrix;
IFFT_modulation(:,conjugate_carriers)=conj(complex_carrier_matrix);

time_wave_matrix=ifft(IFFT_modulation');
time_wave_matrix=time_wave_matrix';

for f=1:carrier_count
    temp_bins(1:IFFT_bin_length)=0+0j;
    temp_bins(carriers(f))=IFFT_modulation(2,carriers(f));
    temp_bins(conjugate_carriers(f))=IFFT_modulation(2,conjugate_carriers(f));
    temp_time=ifft(temp_bins');
end
for i=1:symbols_per_carrier+1
    windowed_time_wave_matrix(i,:)=real(time_wave_matrix(i,:)).*hamming(IFFT_bin_length)';
    windowed_time_wave_matrix(i,:)=real(time_wave_matrix(i,:));
end

ofdm_modulation=reshape(windowed_time_wave_matrix',1,IFFT_bin_length*(symbols_per_carrier+1));
temp_time=IFFT_bin_length*(symbols_per_carrier+1);

symbols_per_average=ceil(symbols_per_carrier/5);
avg_temp_time=IFFT_bin_length*symbols_per_average;
averages=floor(temp_time/avg_temp_time);
average_fft(1:avg_temp_time)=0;
for a=0:(averages-1)
    subset_ofdm=ofdm_modulation(((a*avg_temp_time)+1:((a+1)*avg_temp_time)));
    subset_ofdm_f=abs(fft(subset_ofdm));
    average_fft=average_fft+(subset_ofdm_f/averages);
end
average_fft_log=20*log10(average_fft);
Tx_data=ofdm_modulation;
Tx_signal_power=var(Tx_data);
liner_SNR=10^(SNR/10);
noise_sigma=Tx_signal_power/liner_SNR;
noise_scale_factor=sqrt(noise_sigma);
noise=randn(1,length(ofdm_modulation));
copy1=zeros(1,length(ofdm_modulation));
for i=2:length(ofdm_modulation)
    copy1(i)=ofdm_modulation(i-1);
end
Rx_data=Tx_data+noise*noise_scale_factor;
%noise=abs(fft(noise))/length(ofdm_modulation)*2;
%Rx_data=abs(fft(Rx_data))/length(ofdm_modulation)*2;
y=Rx_data;
end

