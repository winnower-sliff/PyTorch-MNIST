clear; % 清除所有变量
clc;
close all; % 关闭所有图形窗口

%% 读取数据
% data1 = h5read('dataset_training_no_aug.h5','/CFO');
% data2 = h5read('dataset_training_no_aug.h5','/RSS');
data3 = h5read('dataset_training_no_aug.h5','/data');
% data4 = h5read('dataset_training_no_aug.h5','/label');
data_size = 16384/2;
data_imag = 1:1:data_size;
data_real = data_size+1:1:data_size*2;

%% 提取范围

% 定义起始和结束范围  
start = 1;  
end_val = 15000;  
step_size = 500;  
extract_count = 10;  
  
% 初始化结果数组  
result = [];  
  
% 循环提取  
for i = 0:step_size:(end_val - step_size)  
    % 计算当前段的起始和结束索引  
    start_idx = i + 1;  
    end_idx = min(i + extract_count, end_val);  
      
    % 提取当前段的前50个数（或者更少，如果接近末尾）  
    segment = start_idx:end_idx;  
    segment = segment+20;
    result = [result, segment];  
end  
  
% 显示结果  
disp(result);

iwant=result;
leng=size(iwant,2);
% n=0:leng*data_size-1;
for i=1:leng
    data_iq((i-1)*data_size+1:i*data_size) = ...
        complex(data3(data_real,iwant(i)),data3(data_imag,iwant(i)));
end

%% 循环
% figure()
tic
save_data=zeros(2*leng,4070);
h = waitbar(0, 'Processing...'); % 创建一个进度条，初始值为0，并显示文本“Processing...”  
  
for i_leng=1:leng
    %% 进行STFT
    Fs=1e6;
    smrg=data_size;% 测试范围
    fin=16384;% 频率精细度
    window=64;
    noverlap=63;
    [S,F,T,P] = spectrogram(data_iq((i_leng-1)*smrg+1:(i_leng-1)*smrg+smrg),window,noverlap,fin,Fs);
    S = abs(S);
     
    %% 频谱搬移
    md=floor(fin/2);
    S_fz=[S(md+1:end,:);S(1:md,:)];
    
    F_fz=[F(md+1:end)-Fs;F(1:md)];
    
    % figure()
    % imagesc(T,F_fz,S_fz/max(S_fz(:)));
    % axis xy;
    % xlabel('Time (s)');
    % ylabel('Frequency (Hz)');
    % title('STFT Spectrogram');
    % colorbar;
    
    %% 求上下包络
    UDidx=zeros(2,smrg-noverlap);
    [MxValue,MxIdx]=max(S_fz);
    mnMxV=min(MxValue);
    % mMxV=5;
    alpha=3;
    
    for i = 1:smrg-noverlap
        UDidx(:,i)=MxIdx(i);
        while S_fz(UDidx(1,i),i)>mnMxV*0.5 && UDidx(1,i)>1
            UDidx(1,i)=UDidx(1,i)-1;
        end
        while S_fz(UDidx(2,i),i)>mnMxV*0.5 && UDidx(2,i)<fin
            UDidx(2,i)=UDidx(2,i)+1;
        end
    end
    
    % 瞬时频率函数（示例）
    inst_freq_M = MxIdx/fin*Fs-Fs/2;
    inst_freq_U = UDidx(2,:)/fin*Fs-Fs/2;
    inst_freq_D = UDidx(1,:)/fin*Fs-Fs/2;
    
    if_UM=inst_freq_U-inst_freq_M; % 上包络-原时频信号
    if_DM=inst_freq_D-inst_freq_M; % 下包络-原时频信号

    ifs=[inst_freq_M; inst_freq_U; inst_freq_D];
    
    %% 绘制上下包络频率
    % figure()
    % plot(T,inst_freq_M)
    % hold on
    % plot(T,inst_freq_U)
    % hold on
    % plot(T,inst_freq_D);
    
    %% 绘制时频相减值

    % % subplot(leng,1,i_leng+1);
    % plot(T, if_UM);
    % hold on
    % % plot(T, if_DM);
    % title('瞬时频率');
    % xlabel('时间 (s)');
    % ylabel('频率 (Hz)');
    
    %% 恢复时域信号
    % 假设信号的幅度为1  
    amplitude = 1;  
    % 假设初始相位为0
    initial_phase = 0 * 2 * pi;  
    % 初始化时域信号（复数形式，包含相位信息）  
    xs=zeros(3,size(T,2)) + 1i * zeros(3,size(T,2)); 
    xs(:,1)=amplitude * exp(1i * initial_phase);

    for i_ifs=1:3
        % 生成信号（简化版，使用线性插值频率）  
        for i = 2:length(T)  
            % 假设在两个相邻点之间频率是线性的  
            freq_avg = (ifs(i_ifs,i) + ifs(i_ifs,i-1)) / 2;  
            phase_increment = 2 * pi * freq_avg * (T(i) - T(i-1));  
            xs(i_ifs,i) = xs(i_ifs,i-1) * exp(1j * phase_increment); 
        end
    end

    %% 绘制恢复信号  
  
    % plot(T, real(xs(1,:)));  
    % hold on
    % plot(T, real(xs(2,:)));      
    % hold on
    % plot(T, real(xs(3,:)));  
    % title('生成的时域信号（实部）');  
    % xlabel('时间 (s)');  
    % ylabel('幅度');

    %% 上包络-原信号；下包络-原信号
    x_UM=xs(2,:)-xs(1,:);
    x_DM=xs(3,:)-xs(1,:);

    %% 对得到的新信号进行WT
    numLevels=5; % 分解层数
    wavelet='coif2'; % 小波类型
    [C_UM,L_UM] = wavedec(real(x_UM),numLevels,wavelet);
    [C_DM,L_DM] = wavedec(real(x_DM),numLevels,wavelet);

    %% 画出分解后的信号图

    % subplot(numLevels+1,1,1)
    % plot(real(x_UM));
    % hold on
    % plot(real(x_DM));
    % for i = 1:numLevels
        % subplot(numLevels+1,1,(numLevels+1+1-i))

        % subplot(5,2,i_leng+1)
        % 
        % D_UM = detcoef(C_UM, L_UM, 1);
        % plot(D_UM);
        % hold on
        % D_DM = detcoef(C_DM,L_DM, 1);
        % plot(D_DM);
    % end
    
    %% 得到分解结果，存入数据
    D_UM = detcoef(C_UM, L_UM, 1);
    D_DM = detcoef(C_DM,L_DM, 1);
    save_data(2*i_leng-1:2*i_leng,:)=[D_UM;D_DM];

    %% 更新进度条的进度  
    progress = i_leng / leng; % 计算当前进度（0到1之间）  
    waitbar(progress, h, sprintf('Progress: %d%%', round(progress * 100))); % 更新进度条并显示当前进度百分比  

end
toc
disp(['运行时间: ',num2str(toc)]);

%% 保存数据
% labelname='labels.csv';
% dataname='datas.csv';
% labels=floor((iwant-1)/500)+1;
% writematrix(labels, labelname);
% writematrix(save_data,dataname);
labelname='labels_2.mat';
dataname='datas_2.mat';
labels=floor((iwant-1)/500)+1;
save(labelname,"labels");
save(dataname,"save_data");
