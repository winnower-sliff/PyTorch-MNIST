%% 提取范围

% 定义起始和结束范围  
start = 1;  
end_val = 5000;  
step_size = 500;  
extract_count = 450;  
  
% 初始化结果数组  
result1 = [];  
result2=[];
  
% 循环提取  
for i = 0:step_size:(end_val - step_size)  
    % 计算当前段的起始和结束索引  
    start_idx = i + 1;  
    end_idx = min(i + extract_count, end_val);  
      
    % 提取当前段的前50个数（或者更少，如果接近末尾）  
    segment1 = start_idx:end_idx;  
    segment1 = segment1;
    result1 = [result1, segment1];  
    segment2 = (start_idx*2-1):(end_idx*2);  
    segment2 = segment2;
    result2 = [result2, segment2];  
end  
  
% 显示结果  
disp(result1);
disp(result2);

%%
iwant_lb=result1;
iwant_dt=result2;

load("datas_wt_BIG.mat")
load("labels_wt_BIG.mat")

labels=labels(iwant);
save_data=save_data(1:10000,:);

labelname='labels_wt_5.mat';
dataname='datas_wt_5.mat';

save(labelname,"labels");
save(dataname,"save_data");