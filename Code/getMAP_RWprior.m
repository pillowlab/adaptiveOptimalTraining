function [wMode,Hess,logEvd,varargout] = getMAP_RWprior(dat,mysigma,myOpts)
% estimates non-stationary weight parameters with random walk prior

% for NIPS paper: Adaptive optimal training of animal behavior (May 2016)
% rearranged to share (Apr 2017)

% 2016-2017 Ji Hyun Bak

%% initialize

% unpack input
if(isfield(dat,'m'))
    m = dat.m;
    if(unique(sum(m,2))~=1) % each row in m should have only one 1, otherwise 0
        error('getMAP: input format mismatch: dat.m');
    end
elseif(isfield(dat,'y')) % allow alternative input
    y = dat.y;
    if(size(y,2)~=1) % y should be a single column vector
        error('getMAP: input mismatch: dat.y');
    end
    if(isfield(dat,'allys'))
        allys = dat.allys;
    else
        allys = unique(y); %[0 1 2]; %unique(y);
    end
    m = bsxfun(@eq,allys(:)',y(:));
    dat.m = m; 
else
    error('getMAP: no response given.');
end

N = size(m,1); % number of trials
ydim = size(m,2)-1; % number of independent outcomes

if(isfield(dat,'x'))
    x = dat.x;
    if(size(x,1)==N)
        gdim = size(x,2)+1; % for this specific g(x) = [1 x]
    else
        error('getMAP: trial number mismatch (x,y)');
    end
else
    % if x is not given
    x = zeros(N,0); 
    dat.x = x; 
    gdim = 1;
end

K = ydim*gdim; % number of weight parameters (without lapse)

% pass initializations if given

firstPrsInit = zeros(K,1);
if(isfield(myOpts,'prsInit'))
    prsInput = myOpts.prsInit;
    if(numel(prsInput)==K)
        firstPrsInit = prsInput(:);
        % otherwise leave as column of zeros
    end
end

sigInit = mysigma; % default is constant sigma
if(isfield(myOpts,'sigInit'))
    sigInit = myOpts.sigInit;
end

showopt = 0;
if(isfield(myOpts,'showopt'))
    showopt = myOpts.showopt;
end

maxIter = 1000;
if(isfield(myOpts,'maxIter'))
    maxIter = myOpts.maxIter;
end


%%% unpack policy update parameters 
if(all([isfield(myOpts,'alpha') isfield(myOpts,'eta') isfield(myOpts,'kappa')]))
    moreParams = struct('alpha',myOpts.alpha,...
        'eta',myOpts.eta,'kappa',myOpts.kappa);
    if(isfield(myOpts,'tback'))
        moreParams.tback = myOpts.tback;
    end
    moreParams.drift = true;
else
    moreParams = [];
end


%% MAP estimate

display('Obtaining MAP estimate...');

prsInitArray = bsxfun(@times,ones(N,1),firstPrsInit(:)'); % replicate N times
prs_init = prsInitArray(:); % stack of columns (of each weight type)
lossfun = @(prs) ...
    negLogPost_MNLRW(prs,dat,mysigma,firstPrsInit,sigInit,moreParams); 

if(showopt>0)
    opts = optimset('display','iter','gradobj','on','Hessian','on','MaxIter',maxIter); % for fminunc
else
    opts = optimset('display','off','gradobj','on','Hessian','on','MaxIter',maxIter); % for fminunc
end
[wMode,~,flag,~,~,Hess] = fminunc(lossfun,prs_init,opts);

if flag < 0
    warning('MAP estimate: fminunc did not converge to optimum');
end


%% Evidence (Marginal likelihood)

if(showopt>0)
    display(' ');
end
display('Calculating evidence...');


% prior and likelihood at wMode

if(showopt>0)
    display(' - prior and likelihood at wMode...');
end

[pT,lT] = getLP_MNLogistic_RWprior(wMode,dat,mysigma,firstPrsInit,sigInit,moreParams);
logterm_prior = pT.logprior;
logterm_li = lT.logli;



%%% --- posterior term (with Laplace approx)

if(showopt>0)
    display(' - posterior with Laplace approx...');
end

logterm_post = (1/2)*sparse_logdet(Hess); 

% log evidence
logEvd = logterm_li + logterm_prior - logterm_post;


%%% deal with varargout 
if(nargout>3)
    llstruct.logli = logterm_li;
    llstruct.logprior = logterm_prior;
    llstruct.logpost = logterm_post;
    varargout{1} = llstruct;
end

end