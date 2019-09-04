% funs_MNLogistic.m

% 2015-2017 Ji Hyun Bak

% ----------------------------------------------------------------------

% design matrix
gfun = @(x) [ones(size(x,1),1) x]'; % add a row of 1's on top 
getXmat = @(argStim,argYdim) kron(eye(argYdim),argStim'); 

% parameter array & linear predictor vector
getWarray = @(argParam,argDims) reshape(argParam(:),argDims.g,argDims.y);
getV = @(argStim,argParam,argDims) vertcat(zeros(1,size(argStim,1)), ...
    getWarray(argParam,argDims)'*gfun(argStim)); % corrected 9/12/2015

% probability vector
getZ = @(argV) sum(exp(argV),1);
getProb0 = @(argV) bsxfun(@times,exp(argV),1./getZ(argV)); % can deal with arrays
getProb0XW = @(argStim,argParam,argDims) getProb0(getV(argStim,argParam,argDims));

% probability - new method (for precision issues)
% because Matlab gives exp(large)=Inf while exp(small) is ok,
% and Inf/Inf = NaN, evaluating p[j] = 1./{sum_[h] exp(V[h]-V[j])} helps.
getProb = @(argV) 1./sum(exp(bsxfun(@plus,-argV,argV')),2);

% another version (10/27/2015 with Jonathan)
getProb2 = @(arvV) getProb0(bsxfun(@minus,argV,max(argV,[],1)));

% dealing with lapse rate
getProbLapse = @(argP,argLapse) bsxfun(@plus,(1-sum(argLapse))*argP, argLapse(:));
getLapse = @(gamma) exp(gamma)./(1+exp(gamma)); % logistic transformation


% ----------------------------------------------------------------------
% for derivatives of LL
getDelta = @(argX,argY,argP) argX'*(argY-argP);
getGamma = @(argP) diag(argP)-argP*argP';
getLambda = @(argX,argP) argX'*getGamma(argP)*argX;