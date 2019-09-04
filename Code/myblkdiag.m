function y = myblkdiag(C)
% extracted/modified from builtin function BLKDIAG, for cell array input
% Apr 2016 JHB

y = [];
for k=1:numel(C)
    x = C{k};
    [p1,m1] = size(y);
    [p2,m2] = size(x);
    y = [y zeros(p1,m2); zeros(p2,m1) x]; %#ok
end

end

%%%%% original header:
%BLKDIAG  Block diagonal concatenation of matrix input arguments.
%
%                                   |A 0 .. 0|
%   Y = BLKDIAG(A,B,...)  produces  |0 B .. 0|
%                                   |0 0 ..  |
%
%   Class support for inputs:
%      float: double, single
%      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
%      char, logical
%
%   See also DIAG, HORZCAT, VERTCAT

% Copyright 1984-2013 The MathWorks, Inc.