function [M,color,coordinate] = key_points(file_name,points_number)
    clear M lo co;
    pc=pcread(file_name);
    coordinate=pc.Location;
    color=single(pc.Color);
    i_sample = 4;%the length of filter
       
    attribute=[coordinate,color];
    attribute=sortrows(attribute,[3 1 2]);%���Ȼ��ڵ����ŵ�Ԫ�أ�Ȼ���һ�ŵ�Ԫ�أ�Ȼ��ڶ��ŵ�Ԫ�ؽ�������
    coordinate=attribute(:,1:3);
    color=attribute(:,4:6);
    tic; score = computeVariation(coordinate, 50); toc;%tic,toc ��¼ʱ��

    N = size(score,1);
    M = datasample(1:N, points_number, 'Replace', true, 'Weights',  score(:,i_sample) );%����Ȩ�ش�С���ȡ��
  
end

