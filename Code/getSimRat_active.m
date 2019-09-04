function [setby,wSim,simdat,polgrad,drExp] = getSimRat_active(params,dims,xall,wInit,varargin)
% get a simulated learning behavior following policy-gradient update
% with actively selected stimulus

% for NIPS paper: Adaptive optimal training of animal behavior (May 2016)
% rearranged to share (Apr 2017)

% 2016-2017 Ji Hyun Bak

%% unpack input

%%% stimulus space & input distribution

[xset,xcnt] = mytally(xall);
nxset = size(xset,1);
%px0 = ones(nxset,1)/nxset; % uniform input distribution
px0 = xcnt/sum(xcnt); % empirical input distribution

N = size(xall,1);
K = (dims.y)*(dims.g);
allys = 1:(1+dims.y);


% check/inherit parameters

if( K~=numel(wInit) )
    error('parameter dimension mismatch: wInit');
end

alpha = params.alpha; % parameter learning rate 
if(or(numel(alpha)==1, isequal(size(alpha),[K K])))
    % OK: (either scalar or diagonal matrix)
elseif(numel(alpha)==K*K)
    alpha = reshape(alpha,[K K]);
else
    error('parameter dimension mismatch: alpha');
end

if(isfield(params,'kappa'))
    kappa = params.kappa; % skewness
else
    kappa = 1;
end

if(isfield(params,'decay'))
    decay = params.decay;
else
    decay = 1; % no decay
end

if(isfield(params,'sigma'))
    sigma = params.sigma; 
else
    sigma = 0;
end

if(isfield(params,'tback'))
    tback = params.tback;
else
    tback = 0;
end

if(isfield(params,'AT'))
    activeTraining = params.AT; 
else
    activeTraining = false;
end


%% simulate time-varying choice behavior

setby = zeros(N,2);

if(activeTraining)
    xall(2:end,:) = NaN(N-1,size(xall,2)); % reset all future stimuli
    thetaGoal = [0 -10 10 0];
end

wSim = zeros(N,K);
polgrad = zeros(N,K);
drExp = -Inf(N,nxset);

yall = zeros(N,1);

mytheta = wInit(:)'; % initialize

for t = 1:N
    if(t>1)
        
        % generate choice from multinomial prob distribution
        myv = [1 xall(t,:)]*mytheta(:);
        myp = 1/(1+exp(-myv));
        yall(t) = mnrnd(1,[1-myp myp])*allys(:);
        
        if(~all(xall(t,:)<Inf))
            % ---- reset x[t] -- update for the active version (5/7/2016)
            
            % determine optimal stimulus 
            allgrads = zeros(K,nxset);
            for xi = 1:nxset
                pxdelta = ((1:nxset)'==xi); % delta p(x)
                thegrad = getPolGrad_discrimTask(xset,pxdelta,allys,yall(t),mytheta,kappa);
                allgrads(:,xi) = thegrad;
                %drExp(t,xi) = (thetaGoal-decay*mytheta)*alpha*thegrad(:);
            end
            drExp(t,:) = (thetaGoal-decay*mytheta)*alpha*allgrads;
            [dmax,imax] = max(drExp(t,:));
            
            % check for history variable match (see up to two steps ahead)
            myz = sign(diff(xall(t-1,1:2)));
            bestz = xset(imax,3);
            if(myz==bestz)
                % history relation match
                xall(t,:) = xset(imax,:);
                setby(t,:) = [1 dmax];
            else
                % history mismatch (should make it up in two steps)
                crit = and(xset(:,3)==myz,sign(diff(xset(:,1:2),[],2))==bestz);
                xmatch = xset(crit,:);
                [dm1,im1] = max(drExp(t,crit));
                xall(t,:) = xmatch(im1,:);
                if(t<N)
                    xall(t+1,:) = xset(imax,:);
                end
                setby(t,:) = [1 dm1];
                setby(t+1,:) = [2 dmax];
            end
            % ------------------------------------------------------------
        end
        
    end
    
    % set up dynamic input distribution
    if(tback>=0)
        recent = max(1,t-tback):t;
        [~,ct] = mytally([xall(recent,:);xset]);
        pxt = (ct-1)/sum(ct-1); % "recent" input distribution
    else
        pxt = px0; % uniform distribution if tback<0
    end
    
    % policy gradient update
    mypg = getPolGrad_discrimTask(xset,pxt,allys,yall(t),mytheta,kappa);
    mytheta = decay*mytheta + mypg*alpha' ... % transpose is important! 
        + randn(1,K)*sigma; % noise: 5/13/2016 update
    
    wSim(t,:) = mytheta;
    polgrad(t,:) = mypg;
    
end

% check causality
if(activeTraining)
    if(~isequal(sign(diff(xall(1:end-1,1:2),[],2)),xall(2:end,3)))
        error('something wrong in stimulus choice process.');
    end
end

% success at each trial
sall = (sign(diff(xall(:,1:2),[],2))==sign(yall-1.5));

% pack output
simdat = struct('x',xall,'y',yall,'s',sall,'allys',allys);

end