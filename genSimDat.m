% genSimDat.m

% generate a simulated dataset (for code testing)
% Apr 2017 Ji Hyun Bak

%% initialize

clear all;
clc;

setpaths;

% set directory
basedir = 'Test/';
if(~exist(basedir,'dir'))
    mkdir(basedir);
end


%%% set overall parameters

% set data size
N = 2000; %500; % number of trials

% set "true" hyperparameters
alpha = 2^-6; % learning rate
sigma = 2^-6; % noise strength


%% generate simulated data

%%% set up stimulus spaces 

% stimulus space
xgrid1D = (55:10:95);
xx = combvec(xgrid1D,xgrid1D)';
xx = xx(diff(xx,[],2)~=0,:); % exclude diagonal

% scaled stimulus space
xcenter = mean(xgrid1D); % typical scale of stimulus
xstd = std(xgrid1D);
xsetTrue = (unique(xx,'rows')-xcenter)/xstd;
nxsetTrue = size(xsetTrue,1);


%%% prepare stimuli

% simulate full session
iall = randsample(nxsetTrue,N,'true');
xall0 = xsetTrue(iall,:);
xprev0 = sign(diff(xsetTrue(randsample(nxsetTrue,1),:)));
zall = [xprev0; sign(diff(xall0(1:end-1,:),[],2))];
xall = [xall0 zall]; % with single-step-back history term
xset = unique(xall,'rows'); % extended stimulus set


%%% run simulation

wInit = [-1 0 0 1];
K = numel(wInit); % number of parameter types
dims_sim = struct('y',1,'g',K); % binary response

params = struct('alpha',alpha,'sigma',sigma,'AT',false); % AT false: random stimuli
[~,wSim,simdat,~,~] = getSimRat_active(params,dims_sim,xall,wInit);

% wrap like a real dataset
alldat = simdat;
alldat.z = simdat.x(:,3);
alldat.x = simdat.x(:,1:2);
dims = struct('y',numel(alldat.allys)-1,'g',size(alldat.x,2)+1);


%%% save to file
filename = [basedir,'testdat.mat'];
save(filename,'alldat','dims','alpha','sigma');
