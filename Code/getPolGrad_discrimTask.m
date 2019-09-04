function [mypg,varargout]= getPolGrad_discrimTask(xset,pxt,allys,myy,mywgt,kappa)
% compute policy gradient for the specific task structure

% 2016 Ji Hyun Bak

%% compute trial-specific policy gradient (and its gradient wrt weight)

nxset = size(xset,1);
K = numel(mywgt);

mypg = 0;

dpg = zeros(K,K);
ddpg = zeros(K,K,K);

for xi = 1:nxset
    
    %%% prepare ingredients
    myx = xset(xi,:);
    myg = [1 myx];
    myv = myg*mywgt(:);
    myp = 1/(1+exp(-myv));
    mykappa = (allys==myy) + kappa*(allys~=myy);
    myeps = [-1 1];
    myr = [diff(myx(1:2))<0 diff(myx(1:2))>0];
    myf = sum(mykappa(:).*myeps(:).*myr(:));
    
    %%% the PG
    coeff1 = myp*(1-myp);
    mypg = mypg + pxt(xi)*myf*coeff1*myg;
    
    %%% first derivative
    coeff2 = myp*(1-myp)*(1-2*myp);
    gg = (myg'*myg);
    dpg = dpg + pxt(xi)*myf*coeff2*gg;
    
    %%% second derivative
    coeff3 = myp*(1-myp)*(1-6*myp+6*myp^2);
    ggg = bsxfun(@times,(myg'*myg),shiftdim(myg(:),-2)); % g*g*g tensor product
    ddpg = ddpg + pxt(xi)*myf*coeff3*ggg;
end

% pack varargout
if(nargout>1)
    varargout{1} = dpg;
    if(nargout>2)
        varargout{2} = ddpg;
    end
end

end