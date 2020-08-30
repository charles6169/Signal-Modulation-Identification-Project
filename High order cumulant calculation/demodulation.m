function output_frame = demodulation(input_modu, index)
% demodulation for IEEE802.11a
% Input:    input_modu, complex values representing constellation points
%           index
% Output:   output_frame, output bit stream (data unit is one bit)

% In this version, increase the quatilization levels into 8.
% note: Matlab index starts from 1
QAM_input_I = real(input_modu);
QAM_input_Q = imag(input_modu);
output_frame = zeros(1,length(input_modu)*index);
switch index
case 1,
    BPSK_Demodu_I = [0 1];      %f(m)=(m+1)/2 + 1, so I=-1 ---> 1, I=1 ---> 2
    idx = find(QAM_input_I>1);
    QAM_input_I(idx) = 1;
    idx = find(QAM_input_I<-1);
    QAM_input_I(idx) = -1;
    output_frame = BPSK_Demodu_I(round((QAM_input_I+1)/2) + 1);
case 2,
    QPSK_Demodu_IQ = [0 1];     %f(m)=(m+1)/2 + 1, so I=-1 ---> 1, I=1 ---> 2
    idx = find(QAM_input_I>1/sqrt(2));
    QAM_input_I(idx) = 1;
    idx = find(QAM_input_I<-1/sqrt(2));
    QAM_input_I(idx) = -1;
    idx = find(QAM_input_Q>1/sqrt(2));
    QAM_input_Q(idx) = 1;
    idx = find(QAM_input_Q<-1/sqrt(2));
    QAM_input_Q(idx) = -1;
    output_frame(1:2:end) = QPSK_Demodu_IQ(round((QAM_input_I+1)/2) + 1);
    output_frame(2:2:end) = QPSK_Demodu_IQ(round((QAM_input_Q+1)/2) + 1);
case 4,
    QAM_input_I = QAM_input_I.*sqrt(10);
    QAM_input_Q = QAM_input_Q.*sqrt(10);
    QAM_16_Demodu_IQ = [0 1 3 2];   %f(m)=(m+3)/2 + 1, so I=-3 ---> 1, I=1 ---> 3
    idx = find(QAM_input_I>3);
    QAM_input_I(idx) = 3;
    idx = find(QAM_input_I<-3);
    QAM_input_I(idx) = -3;
    idx = find(QAM_input_Q>3);
    QAM_input_Q(idx) = 3;
    idx = find(QAM_input_Q<-3);
    QAM_input_Q(idx) = -3;
    tmp = round((QAM_input_I+3)/2) + 1;
    output_frame(1:4:end) = bitget(QAM_16_Demodu_IQ(tmp),2);
    output_frame(2:4:end) = bitget(QAM_16_Demodu_IQ(tmp),1);
    tmp = round((QAM_input_Q+3)/2) + 1;
    output_frame(3:4:end) = bitget(QAM_16_Demodu_IQ(tmp),2);
    output_frame(4:4:end) = bitget(QAM_16_Demodu_IQ(tmp),1);
case 6,
    QAM_input_I = QAM_input_I.*sqrt(42);
    QAM_input_Q = QAM_input_Q.*sqrt(42);
    QAM_64_Demodu_IQ = [0 1 3 2 6 7 5 4];   %f(m)=(m+7)/2 + 1, so I=-7 ---> 1, I=1 ---> 5
    idx = find(QAM_input_I>7);
    QAM_input_I(idx) = 7;
    idx = find(QAM_input_I<-7);
    QAM_input_I(idx) = -7;
    idx = find(QAM_input_Q>7);
    QAM_input_Q(idx) = 7;
    idx = find(QAM_input_Q<-7);
    QAM_input_Q(idx) = -7;
    tmp = round((QAM_input_I+7)/2) + 1;
    output_frame(1:6:end) = bitget(QAM_64_Demodu_IQ(tmp),3);
    output_frame(2:6:end) = bitget(QAM_64_Demodu_IQ(tmp),2);
    output_frame(3:6:end) = bitget(QAM_64_Demodu_IQ(tmp),1);
    tmp = round((QAM_input_Q+7)/2) + 1;
    output_frame(4:6:end) = bitget(QAM_64_Demodu_IQ(tmp),3);
    output_frame(5:6:end) = bitget(QAM_64_Demodu_IQ(tmp),2);
    output_frame(6:6:end) = bitget(QAM_64_Demodu_IQ(tmp),1);
end
