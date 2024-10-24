clear; % 清除所有变量
clc;
close all; % 关闭所有图形窗口
load('dataset_training_aug_0hz')
figure()
plot(n,data_iq);title('data');

Fs=1*10e6;
[S,F,T,P] = spectrogram(data_iq,256,128,256,Fs);
S = abs(S)/ max(abs(S(:)));
% 绘制 STFT 频谱图
figure()
imagesc(T, F,S);
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;
save('test_aug_0hz')
