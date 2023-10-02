% 初始化Wi-Fi设备和数据收集
wifiDevice = wlanVHTConfig('ChannelBandwidth', 'CBW20', 'MCS', 5);
%
% rx = wlanRecoveryConfig('EqualizationMethod', 'ZF', 'ChannelEstimationMethod', 'Ideal');

% 创建绘图
figure;
hPlot = plot(0, 0); % 创建初始曲线
xlabel('时间 (秒)'); % 修改X轴标签
ylabel('CSI值');
title('实时CSI信号曲线');
grid on;

% 实时数据处理和绘图
startTime = now; % 记录开始时间
while true
    % 收集CSI数据，这里使用假数据代替
    csiData = rand(1, 30); % 假设每次收集30个CSI值

    % 计算经过的时间（秒）
    currentTime = (now - startTime) * 24 * 60 * 60;

    % 更新绘图
    set(hPlot, 'XData', [get(hPlot, 'XData'), currentTime], 'YData', [get(hPlot, 'YData'), mean(csiData)]);
    drawnow;

    % 这里可以进行其他实时数据处理
end
