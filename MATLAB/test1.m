clear; % 清除所有变量
clc;
close all; % 关闭所有图形窗口
%h5disp('dataset_training_no_aug.h5');
data1 = h5read('dataset_training_no_aug.h5','/CFO');
data2 = h5read('dataset_training_no_aug.h5','/RSS');
data3 = h5read('dataset_training_no_aug.h5','/data');
data4 = h5read('dataset_training_no_aug.h5','/label');
% figure()
% subplot(1,2,1);plot(data1(1:500));title('CF0');
% subplot(1,2,2);plot(data2(1:500));title('RSS');
date_size = 16384/2;
data_imag = 1:1:date_size;
data_real = date_size+1:1:date_size*2;
leng=5;
n=0:leng*date_size-1;
for i=1:leng
    data_iq((i-1)*date_size+1:i*date_size) = complex(data3(data_real,i),data3(data_imag,i));
end

save('dataset_training_no_aug',"data_iq","date_size")