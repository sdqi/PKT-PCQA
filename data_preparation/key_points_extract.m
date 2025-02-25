% 参数设置
points_number = 1024; % 提取的关键点个数

root = fullfile('F:\deep_learning\PKT_PCQA\kaiyuan\data\datasets\test');
out_root = fullfile('F:\deep_learning\PKT_PCQA\kaiyuan\data\key_points\test_key1024');
content = strings(0); % 创建空的字符串数组，用于存储文件名
if ~exist(out_root, 'dir')
    mkdir(out_root);
end

% 获取root路径下所有的ply文件
rootOutput = dir(fullfile(root, '*.ply'));
folder_size_row = size(rootOutput);
root_folder_file_number = folder_size_row(1);

for j = 1:root_folder_file_number
    % 读取点云文件路径
    root_folder_file = fullfile(root, rootOutput(j, 1).name)
    
    % 提取关键点
    [M, color, coordinate] = key_points(root_folder_file, points_number);
    
    % 保存文件名（去除后缀）
    [~, filename, ~] = fileparts(rootOutput(j, 1).name);
    content = [content; filename]; % 将文件名添加到content中
    
    % 处理点云数据
    lo = [coordinate(M, 1), coordinate(M, 2), coordinate(M, 3)];
    co = [color(M, 1), color(M, 2), color(M, 3)];
    co = uint8(co);  % 转换颜色为 uint8 类型
    
    % 创建点云对象
    pt = pointCloud(lo, 'Color', co);
    
    % 输出路径
    out_root_folder_file = fullfile(out_root, strcat(filename, '.ply'));
    % 保存提取的点云
    pcwrite(pt, out_root_folder_file, 'PLYFormat', 'binary');
end

% 将content数组保存到Excel文件中
xlswrite('F:\deep_learning\PKT_PCQA\kaiyuan\data\excel\test_content.xlsx', {'content'}, 'Sheet1', 'A1'); % 写入标题
xlswrite('F:\deep_learning\PKT_PCQA\kaiyuan\data\excel\test_content.xlsx', content, 'Sheet1', 'A2');    % 写入文件名内容
xlswrite('F:\deep_learning\PKT_PCQA\kaiyuan\data\excel\test_content.xlsx', {'MOS'}, 'Sheet1', 'B1');    % 在B1单元格写入'MOS'
