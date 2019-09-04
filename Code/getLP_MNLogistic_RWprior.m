function [priorTerms,liTerms] = getLP_MNLogistic_RWprior(prmSeq,dat,sigma,prs0,sigInit,varargin)
% given a sequence of parameters formatted as an N*K matrix,
% calculates random-walk log priors & multinomial logistic log likelihoods
% and their derivatives

% for NIPS paper: Adaptive optimal training of animal behavior (May 2016)
% rearranged to share (Apr 2017)

% 2016-2017 Ji Hyun Bak


%% unpack input

if(isfield(dat,'x') && or(isfield(dat,'y'),isfield(dat,'m')))
    x = dat.x;
    if(isfield(dat,'y'))
        y = dat.y;
        % construct m (choice data in 1's and 0's)
        if(isfield(dat,'allys'))
            allys = dat.allys;
        else
            allys = unique(y);
        end
        M = bsxfun(@eq,allys(:)',y(:));
    elseif(isfield(dat,'m'))
        % construct y (5/17/2016 update)
        M = dat.m;
        ydim = size(M,2)-1;
        allys = 0:ydim;
        y = M*allys(:);
    end
else
    error('getLP_MNLogistic_RWprior: insufficient input');
end

if(min(size(y))~=1) % y should be a single vector
    error('input mismatch: dat.y');
end



%%% get dimensions right

if(size(x,1)==numel(y)) 
    N = numel(y); % number of trials
else
    error('negLogPost: trial number mismatch');
end

% dims: specific to current model
ydim = numel(allys)-1; % one weight will be fixed
gdim = size(x,2)+1; % for the current model (linear carrier g(x)=[1 x])
dims = struct('y',ydim,'g',gdim);

K = ydim*gdim; % number of weight parameters



%%% allow sigma to be either a scalar or a K-vector (v2 update)
if(numel(sigma)==1)
    sigmavec = sigma*ones(K,1); % isotropic variability (original ver.)
elseif(numel(sigma)==K)
    sigmavec = sigma(:);
end


%%% deal with lapse

% for now, only work with lapse-free model
mylapse = zeros(1,ydim+1);
withLapse = 0;


%%% reshape parameter sequence
% prs should be rearranged as a matrix, with N rows and K columns
% (N: number of trials, K: number of weights)

if(numel(prmSeq)~=N*K)
    error('parameter dimension mismatch (#trials * #weights)');
end
prmArray = reshape(prmSeq(:),[N K]);


%%% unpack varargin (5/12/2016 update)
alpha = 0;
eta = 1;
kappa = 1;
tback = 0;
driftingWeight = false;
if(nargin>5)
    moreParams = varargin{1};
    if(isfield(moreParams,'alpha'))
        alpha = moreParams.alpha; % alpha is either a scalar or a K-vector
    end
    if(isfield(moreParams,'eta'))
        eta = moreParams.eta;
    end
    if(isfield(moreParams,'kappa'))
        kappa = moreParams.kappa;
    end
    if(isfield(moreParams,'tback'))
        tback = moreParams.tback;
    end
    if(isfield(moreParams,'drift'))
        driftingWeight = moreParams.drift;
    end
end


%% set up policy update "drift" (5/12/2016 addition)


% set trivial (uniform) input distribution
xset = unique(x,'rows');
nxset = size(xset,1);
px0 = ones(nxset,1)/nxset;

vdrift = zeros(N,K); % default
dvdrift = zeros(N,K,K);
ddvdrift = zeros(N,K,K,K);

% add drift term
if(driftingWeight) % if options were passed
    
    if(numel(allys)~=2)
        error('for now polgrad can only be calculated for binomial model.');
    end
    
    %for t = 1:N
    for t = 1:(N-1) % v2 update (one-off index shift)
        
        % set up dynamic input distribution
        if(tback>=0)
            recent = max(1,t-tback):t;
            [~,ct] = mytally([x(recent,:);xset]);
            pxt = (ct-1)/sum(ct-1); % "recent" input distribution
        else
            pxt = px0; % uniform distribution if tback<0
        end
        
        % get trial-specific policy gradient and the derivatives
        [pg,dpg,ddpg] = ...
            getPolGrad_discrimTask(xset,pxt,allys,y(t),prmArray(t,:),kappa);

        % one-off index shift (t --> t+1) -- v2 update
        % leave the first entries as zero 
        vdrift(t+1,:) = alpha(:).*pg(:); % [v]
        dvdrift(t+1,:,:) = bsxfun(@times,alpha(:),dpg); % [v w]
        ddvdrift(t+1,:,:,:) = bsxfun(@times,alpha(:),ddpg); % [v w w]
        
    end
    
end


%% calculate random-walk prior (with v2 update)


%%% set up difference matrix Dmat

Dmat = spdiags([ones(N,1) [-eta*ones(N-1,1);0]],[0 -1],N,N);


%%% constructing prior p(w)

% gather all sigma vectors (while allowing different sigma for each w)
sig_diag_array = [sigInit*ones(1,K); ...
    bsxfun(@times,sigmavec(:)',ones(N-1,1))];
sig_diag_vec = sig_diag_array(:); % vectorize column-wise
invSmat_full = spdiags(1./sig_diag_vec.^2, 0, N*K, N*K); % N*K diagonal matrix

% prior covariance matrix
Dmat_cell = cell(K,1);
for ndd = 1:K
    Dmat_cell{ndd} = Dmat;
end
D_full = myblkdiag(Dmat_cell); % N*K block diagonal matrix (still sparse)
invCprior_full = D_full'*invSmat_full*D_full; 

% % the N-vector for the specified weight, across all N trials
w0_vec = reshape( bsxfun(@times,prs0(:)',ones(N,1)), N*K, 1);
drift_vec = D_full\vdrift(:);
wrem_full = prmArray(:) - w0_vec - drift_vec; % all NK*1 vectors

% RW prior (with drift)
logprior = (1/2)*sparse_logdet(invCprior_full) - (1/2)*wrem_full'*invCprior_full*wrem_full;


%%% first derivative

dvdw_cell = cell(K,K);
for vidx = 1:K
    for widx = 1:K
        dvdw_cell{vidx,widx} = spdiags(squeeze(dvdrift(2:end,vidx,widx)),-1,N,N); % trial-specific diagonal shifted
    end
end
dvdw_full = cell2mat(dvdw_cell); % sparse NK*NK matrix
I_full = speye(N*K);
dlogprior = -wrem_full'*invCprior_full*(I_full-D_full\dvdw_full);
dlogprior = dlogprior(:); % make a column vector


%%% second derivative

term1_full = - (I_full-D_full\dvdw_full)'*invCprior_full*(I_full-D_full\dvdw_full); % diagonal
frontpart_full = wrem_full'*invCprior_full*(D_full\I_full); % 1*NK row vector (corresponding to vidx)
term2_cell = cell(K,K);
for widx1 = 1:K
    for widx2 = 1:K
        auxcell = cell(K,1);
        for vidx = 1:K
            auxcell{vidx} = spdiags(squeeze(ddvdrift(2:end,vidx,widx1,widx2)),-1,N,N); % trial-specific diagonal shifted
        end
        auxmat = cell2mat(auxcell); % NK * N matrix
        auxdiagvec = frontpart_full*auxmat;
        term2_cell{widx1,widx2} = spdiags(auxdiagvec(:),0,N,N);
    end
end
ddlogprior = term1_full + cell2mat(term2_cell); % sparse NK*NK matrix

priorTerms = struct('logprior',logprior,'dlogprior',dlogprior,...
    'ddlogprior',ddlogprior');
% ddlogprior is a sparse matrix


%% calculate log likelihood under multinomial logistic model

%%% Log likelihood with lapse

funs_MNLogistic;

%%% calculate separately for each trial, then combine

lliList = zeros(N,1);
dlliList = zeros(N,K);
HlliList = zeros(N,K,K);

for mytrial = 1:N
    
    wgt = prmArray(mytrial,:)'; % transpose: make column vector
    
    m = M(mytrial,:);
    
    % get probability
    wrem = getProb0XW(x(mytrial,:),wgt,dims); % probs without lapse
    p = getProbLapse(wrem,mylapse); % with lapse
    
    % auxiliary vectors
    r = (1-sum(mylapse))*wrem./p;
    t = m'.*r;
    s = m'.*r.*(1-r);
    
    %%% evaluate log likelihood L
    lli = sum(m'.*log(p)); % no sum across datapoints (single trial)
    
    %%% first derivatives {dL/dV, dL/dg} at each x
    dLdV = t - sum(t)*wrem;
    dLdc = (m'./p) - sum(t)/(1-sum(mylapse));
    dcdg = mylapse(:).*(1-mylapse(:)); % c = exp(g)/(1+exp(g)) is the lapse rate
    dLdg = dcdg.*dLdc; %bsxfun(@times,dcdg,dLdc);
    
    %%% second derivatives {ddL/dVdV, ddL/dgdg, ddL/dgdV} at each x
    ddLdVdV = -sum(t)*(diag(wrem)-(wrem*wrem')) + (diag(s)-(wrem*s'+s*wrem')+sum(s)*(wrem*wrem'));
    ddLdcdc = -diag(m'./(p.^2)) ...
        + bsxfun(@plus,m'.*wrem./(p.^2),(m'.*wrem./(p.^2))') ...
        - (sum(s)-sum(t))/(1-sum(mylapse))^2;
    ddLdgdg = ddLdcdc.*(dcdg*dcdg') ...
        + diag((1-2*mylapse(:)).*dLdg);
    ddLdVdc = - bsxfun(@plus,diag(t./p), s/(1-sum(mylapse))) ...
        + bsxfun(@plus, wrem*(t./p)', wrem*sum(s)/(1-sum(mylapse)));
    ddLdVdg = bsxfun(@times,ddLdVdc,dcdg');
    
    
    %%% gradient and Hessian (change variables to weights)
    
    myXmat = getXmat(gfun(x(mytrial,:)),ydim); 
    dlogli_w = myXmat'*dLdV(2:end);
    Hlogli_ww = myXmat'*ddLdVdV(2:end,2:end)*myXmat;
    Hlogli_wg = myXmat'*ddLdVdg(2:end,:);
    dlogli_g = dLdg;
    Hlogli_gg = ddLdgdg;
    
    if(withLapse==1)
        dlli = [dlogli_w; dlogli_g];
        Hlli = [Hlogli_ww Hlogli_wg; Hlogli_wg' Hlogli_gg];
    else
        dlli = dlogli_w;
        Hlli = Hlogli_ww;
    end
    
    lliList(mytrial) = lli;
    dlliList(mytrial,:) = dlli; % as a row vector (for each trial)
    HlliList(mytrial,:,:) = Hlli;
    
end


% combine

logli = sum(lliList);
dlogli = dlliList(:); % vectorize (stacks of columns)

Hcell = cell(K,K); % each cell has [N,N] diagonal matrix
for i = 1:K
    for j = 1:K
        Htij = squeeze(HlliList(:,i,j)); % single vector
        Hcell{i,j} = sparse(diag(Htij)); % sparse
    end
end
ddlogli = cell2mat(Hcell); % concatenate

liTerms = struct('logli',logli,'dlogli',dlogli,'ddlogli',ddlogli);
% ddlogli is a sparse matrix 

end