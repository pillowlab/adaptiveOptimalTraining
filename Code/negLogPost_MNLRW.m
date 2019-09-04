function [negL,dL,ddL] = negLogPost_MNLRW(prmVec,dat,sigma,prsInit,sigInit,varargin)
% calculates posterior using multinomial logistic model & random-walk prior
% returns negative log posterior (and its derivatives)

% 2016 Ji Hyun Bak

% unpack policy update parameters 
if(nargin>5)
    moreParams = varargin{1};
else
    moreParams = [];
end

% get prior and likelihood
[priorTerms,liTerms] = ...
    getLP_MNLogistic_RWprior(prmVec,dat,sigma,prsInit,sigInit,moreParams);

% negative log posterior
negL = - priorTerms.logprior - liTerms.logli;
dL = - priorTerms.dlogprior - liTerms.dlogli;
ddL = - priorTerms.ddlogprior - liTerms.ddlogli;

end