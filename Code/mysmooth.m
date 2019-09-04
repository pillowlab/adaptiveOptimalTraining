function output = mysmooth(input,width)
% smooth input signal with given kernel width
% modification from the builtin smooth function: deals with endpoints
% Apr 2016 JHB

%% unpack input

wpad = floor(width); % integer

%% v1 using builtin smooth

% upper = bsxfun(@times,input(1,:),ones(wpad,1));
% lower = bsxfun(@times,input(end,:),ones(wpad,1));
% auxinput = [upper; input; lower]; % pad upper and lower boundaries
% 
% auxoutput = smooth(auxinput,width);
% output = auxoutput(wpad+1:end-wpad,:);

%% v2 simple rolling average, manually

%nw = 1+2*wpad;
%cache = zeros(1,nw);
output = zeros(size(input));
N = size(input,1);
for t = 1:N
    inds = max(1,t-wpad):min(N,t+wpad);
    output(t,:) = mean(input(inds,:),1);
end

end