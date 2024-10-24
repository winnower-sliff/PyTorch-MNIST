clear; % 清除所有变量
clc;
close all; % 关闭所有图形窗口
load('dataset_training_no_aug')

% figure()
% plot(n(1:1024),data_iq(1:1024));title('data');
% xlim([0 1024])

%% 进行STFT
Fs=1e6;
smrg=8192;% 测试范围
fin=16384;% 频率精细度
window=64;
noverlap=63;
[S,F,T,P] = spectrogram(data_iq(1:smrg),window,noverlap,fin,Fs);
S = abs(S);

%% 绘制 STFT 频谱图
figure()
imagesc(T,F,S/max(S(:)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;
% save('test_no_aug')

%% 求STFT峰值频率格F1
[~,F1]=max(S);

%% 求平均周期时间
ind=[]; % 周期分割点
for i = 1:(length(F1)-1)
    if F1(i)-F1(i+1)>(fin/2)
        ind=[ind,i];
    end
end
Tc=mean(diff(ind))...
    /floor((smrg-noverlap)/(window-noverlap))...
    *T(end);

%% 求斜率
p_all=zeros(1,2*length(ind)-2);
pr=0.08;
for i = 1:(length(ind)-1)
    x=T(ind(i)+1:ind(i+1));
    y=F1(ind(i)+1:ind(i+1));
    len=size(x,2);
    loc1=floor(len*(0.5*pr));
    loc2=floor(len*(0.5*(1-pr)));
    loc3=floor(len*(0.5*pr+0.5));
    loc4=floor(len*(0.5*(1-pr)+0.5));
    p1=polyfit(x(loc1:loc2),y(loc1:loc2),1);
    p2=polyfit(x(loc3:loc4),y(loc3:loc4),1);
    p_all(2*i-1:2*i)=[p1(1),p2(1)];
end
p_mn=abs(mean(p_all));

%% 求带宽B
B_p=p_mn*Tc/fin*Fs;
B_t=2^7/Tc;
disp([B_p,B_t]);

%% 保存数据
save("special-1-SLF", ...
    'B_p','B_t','Fs','Tc', ...
    'data_iq',"Fs", ...
    "smrg","window","noverlap","fin", ...
    'S','F','T','P');
