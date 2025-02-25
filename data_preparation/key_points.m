function [M,color,coordinate] = key_points(file_name,points_number)
    clear M lo co;
    pc=pcread(file_name);
    coordinate=pc.Location;
    color=single(pc.Color);
    i_sample = 4;%the length of filter
       
    attribute=[coordinate,color];
    attribute=sortrows(attribute,[3 1 2]);%首先基于第三排的元素，然后第一排的元素，然后第二排的元素进行排列
    coordinate=attribute(:,1:3);
    color=attribute(:,4:6);
    tic; score = computeVariation(coordinate, 50); toc;%tic,toc 记录时间

    N = size(score,1);
    M = datasample(1:N, points_number, 'Replace', true, 'Weights',  score(:,i_sample) );%根据权重大小随机取点
  
end

