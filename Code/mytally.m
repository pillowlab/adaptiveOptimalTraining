function [uniq,cnt] = mytally(input)
% counts the repetition number of unique rows
% 4/24/2016 JHB

[uniq,~,ib] = unique(input,'rows');
cnt = accumarray(ib,1);

end