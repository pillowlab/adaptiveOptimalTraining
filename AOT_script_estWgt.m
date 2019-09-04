% AOT_script_estWgt.m
% time-varying psychophysical weight estimate, from past observations
% with learning hyperparameters

% for NIPS paper: Adaptive optimal training of animal behavior (May 2016)
% rearranged to share (Apr 2017)

% 2016-2017 Ji Hyun Bak

%% initialize

clear all;
clc;

setpaths;
setcolors;

basedir = 'Test/';
datadir = [basedir,'Saved/'];
if(~exist(datadir,'dir'))
    mkdir(datadir);
end


%% -------- fit with random walk --------

%% (A-B) single-weight simulation: w(t) estimate, repetition test

simname = [datadir,'simtest-1D.mat'];
if(exist(simname,'file'))
    disp('file already exists.');
    
else
    
    % set parameters
    N = 2000; % number of trials
    sigbin = -6;
    sigma = 2^sigbin; % random walk width
    
    % generate w(t) with random walk (change in w is Gaussian random variable)
    diffw = randn(N,1)*sigma;
    w = cumsum(diffw);
    
    % use a simple model p(t) = 1/(1+exp(-w(t)))
    mylogistic = @(wgt) 1./(1+exp(-wgt));
    p = mylogistic(w);
    fullP = [1-p p];
    
    numRep = 5;
    repCell = cell(numRep,2);
    
    % vary sigma
    sigmaList = 2.^((sigbin-2):0.5:(sigbin+2));
    
    setGoodRand;
    
    for nr = 1:numRep
        
        disp(' ');
        disp(['rep ',num2str(nr)]);
        
        % Simulate binary choices
        m = mnrnd(1,fullP); % binary vector
        y = m*[0;1]; % response
        
        %%% MAP estimate & marginal likelihood maximization
        
        wModeList = zeros(N,numel(sigmaList)); % MAP estimate of weight parameter
        evdList = zeros(numel(sigmaList),1); % evidence (marginal likelihood)
        
        showopt = 0;
        
        for ns = 1:numel(sigmaList)
            
            mysigma = sigmaList(ns);
            
            display(['sigma 2^',num2str(log2(mysigma))]);
            
            %%% MAP estimate
            prsInit = 0;
            sigInit = mysigma;
            fullfitOpts = struct('showopt',showopt,'prsInit',prsInit,'sigInit',sigInit);
            newdat = struct('y',y);
            [wMode,~,logEvd,~] = getMAP_RWprior(newdat,mysigma,fullfitOpts);
            
            wModeList(:,ns) = wMode;
            evdList(ns) = logEvd;
        end
        
        disp('done.');
        
        repCell{nr,1} = wModeList;
        repCell{nr,2} = evdList;
        
    end
    
    % %%% save data
    save(simname,'repCell','sigmaList','N','sigbin','w');
    
end

%% (C-D) rat data fit / max-evd

% load dataset
tempVar = load([basedir,'testdat.mat']);
alldat = tempVar.alldat;
dims = tempVar.dims;
% truealpha = tempVar.alpha;
truesigma = tempVar.sigma;
clear tempVar
setTag = '_sim';

sigInit = 1;
hterm = 1; % single-step-back history

allsigbin = -8:-4;
logEvdList = -Inf(numel(allsigbin),1);
wModeList = cell(numel(allsigbin),1);

for ns = 1:numel(allsigbin)
    mysigbin = allsigbin(ns);
    
    filename = [datadir,'fit',setTag,'_h',num2str(hterm),...
        '_sig2n',num2str(abs(mysigbin)),'_siginit',num2str(sigInit),'.mat'];
    
    if(exist(filename,'file'))
        
        %%% load pre-estimated w(t)
        fitVar = load(filename);
        logEvd = fitVar.logEvd;
        wArray = fitVar.wArray;
        disp(filename);
        
    else
        %%% estimate w(t) including history effect
        
        % append history variable
        if(hterm==0)
            newdat = struct('x',alldat.x,'y',alldat.y,'s',alldat.s,'allys',alldat.allys);
        elseif(hterm==1)
            % include z as input
            newdat = struct('x',[alldat.x alldat.z],'y',alldat.y,'s',alldat.s,'allys',alldat.allys);
        else
            error('option hterm not recognized.');
        end
        
        % set dimensions
        ydim = numel(newdat.allys)-1;
        gdim = size(newdat.x,2)+1;
        K = ydim*gdim;
        N = numel(newdat.y);
        
        % set fit options
        showopt = 1;
        prsInit = zeros(K,1);
        fullfitOpts = struct('showopt',showopt,'prsInit',prsInit,'sigInit',sigInit);
        
        % run estimate
        disp(['2^',num2str(mysigbin)]);
        mysigma = 2^mysigbin;
        [wMode,Hess,logEvd] = getMAP_RWprior(newdat,mysigma,fullfitOpts);
        wArray = reshape(wMode,[N K]);
        
        save(filename,'alldat','dims','mysigbin','sigInit','wArray','Hess','logEvd');
        disp('done.');
        
    end
    
    logEvdList(ns) = logEvd;
    wModeList{ns} = wArray;
end
clear fitVar

[~,nsmax] = max(logEvdList);
bestsigbin = allsigbin(nsmax);
wArray = wModeList{nsmax};
clear wModeList


%%% store data for plot
ratFit = struct('evd',logEvdList,'wArray',wArray,...
    'allsigbin',allsigbin,'bestsigbin',bestsigbin);

%% (E) rat data BIC

N = numel(alldat.y);

maxtau = 5;
tauList = 0:maxtau;
KnumList = 4+tauList;

BIClist = zeros(numel(tauList),1);

filename = [datadir,'HTdat',setTag,'_maxT',num2str(maxtau),'.mat'];
if(exist(filename,'file'))
    
    bVar = load(filename);
    logliList = bVar.logliList;
    clear bVar
    
    disp('loaded from existing file.');
    
else
    
    %%% Model selection: how many history terms to include?
    
    % find MAP estimate for the time-varying weight sequence
    % (with varying number of history terms)
    
    xall = alldat.x;
    xdiff = sign(xall(:,2)-xall(:,1));
    
    sigInit = 1;
    sigbinList = log2(truesigma); % fow now, only at the true sigma
    
    %%% try up to tau-back terms
    
    wArray_tauvar = cell(numel(sigbinList),numel(tauList));
    evdList = zeros(numel(sigbinList),numel(tauList));
    logliList = zeros(numel(sigbinList),numel(tauList));
    
    
    for nt = 1:numel(tauList)
        
        tau = tauList(nt);
        
        disp(' ');
        
        % set up stimulus history
        xhist = zeros(N,tau); % if tau==0, empty matrix
        for nh = 1:tau % if tau==0, xhist is left empty
            xhist(:,nh) = [zeros(nh,1); xdiff(1:(end-nh))];
        end
        
        % include stimulus history as input
        newdat = struct('x',[alldat.x xhist],...
            'y',alldat.y,'s',alldat.s,'allys',alldat.allys);
        ydim = numel(newdat.allys)-1;
        gdim = size(newdat.x,2)+1;
        K = ydim*gdim; % changes with tau
        
        % prepare for MAP estimate
        showopt = 1;
        prsInit = zeros(K,1);
        fullfitOpts = struct('showopt',showopt,'prsInit',prsInit,'sigInit',sigInit);
        
        for ns = 1:numel(sigbinList)
            
            mysigbin = sigbinList(ns);
            mysigma = 2^mysigbin;
            display(['tau ',num2str(tau),'; sigma ',num2str(mysigma)]);
            
            [wMode,~,logEvd,llstruct] = getMAP_RWprior(newdat,mysigma,fullfitOpts);
            
            wArray = reshape(wMode,[N K]);
            wArray_tauvar{ns,nt} = wArray;
            evdList(ns,nt) = logEvd;
            logliList(ns,nt) = llstruct.logli;
        end
        
    end
    
    disp(' ');
    disp('done.');
    
    save(filename,'newdat','sigbinList','sigInit','tauList',...
        'wArray_tauvar','evdList','logliList');
    disp('saved to file.');
    
end

myBIC = log(N)*KnumList - 2*logliList;
BIClist(:) = myBIC;

%% plot together


figure(1)
set(gcf,'Position',[50 50 1200 500])
clf;

% set subplot sizes
hnum = 4.5; % number of columns
vnum = 2; %2; % number of rows
hmarg0 = 0.08; % left/right margin
vmarg0 = 0.10; %0.06; % bottom/top margin
hmarg1 = 0.05; %0.05; % column spacing
vmarg1 = 0.14; % row spacing
hsize = (1-hmarg0*2-hmarg1*(hnum-1))/hnum; %0.4; % panel width
vsize = (1-vmarg0*2-vmarg1*(vnum-1))/vnum; %0.4; % panel height



%%% (A) simulated dataset test

sigma = 2^-6;
numRep = 5;

simname = [datadir,'simtest-1D.mat'];
sVar = load(simname);
repCell = sVar.repCell;
sigmaList = sVar.sigmaList;
w = sVar.w;

N = size(w,1);

% -- weight estimate ----

nr = 5;
wModeList = repCell{nr,1};
evdList = repCell{nr,2};
[~,kmax] = max(sum(evdList,2));

wrange = [min([w;wModeList(:,kmax)]) max([w;wModeList(:,kmax)])];
wrange = [floor(10*wrange(1))/10 ceil(10*wrange(2))/10];
wrange = wrange + diff(wrange)*[-0.3 0.1];

axes('Position',[hmarg0 vmarg0+vsize+vmarg1 hsize vsize]);
plot(1:N,w,'k-','LineWidth',2)
hold on
%plot(1:N,y,'k.','MarkerSize',10,'LineWidth',1.5)
plot(1:N,wModeList(:,kmax),'r--','LineWidth',2.5)
hold off
xlim([0 N])
ylim(wrange)
% title('model parameter')
legend('true weight','best fit','Location','SouthEast')
legend('boxoff')
xlabel('trials')
ylabel('weight w')
axis square

% -- evidence maximization --
axes('Position',[hmarg0+hsize+hmarg1 vmarg0+vsize+vmarg1 hsize vsize]);
plot(log(sigmaList)/log(2),evdList-evdList(kmax),'ko-','LineWidth',2)
hold on
ylm0 = ylim;
plot((log(sigmaList(kmax))/log(2))*[1 1],ylm0,'r--','LineWidth',2)
plot((log(sigma)/log(2))*[1 1],ylm0,'r:','LineWidth',2)
hold off
ylim(ylm0)
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(rem(ytk0,4)==0));
xlabel('log_{2} \sigma')
ylabel('log evd. (rel.)')
% title('log evidence')
legend('log evd','max-evd','true \sigma','Location','South')
%legend('boxoff')
axis square


%%% repetition test

% -- repeated w --
axes('Position',[hmarg0 vmarg0 hsize vsize]);
plot(1:N,w,'k-','LineWidth',2)
hold on
for nr = 1:numRep
    wModeList = repCell{nr,1};
    evdList = repCell{nr,2};
    [~,kmax] = max(evdList);
    plot(1:N,wModeList(:,kmax),'--','color',0.5*[1 1 1],'LineWidth',2)
    hold on
end
plot(1:N,w,'k-','LineWidth',2)
hold off
xlim([0 N])
ylim(wrange)
xlabel('trials')
ylabel('weight w')
legend('true weight','repeated fits','Location','SouthEast')
legend('boxoff')
axis square

% -- repeated dw
axes('Position',[hmarg0+hsize+hmarg1 vmarg0 hsize vsize]);
dwRep = zeros(N-1,2);
nrList = [2 3];
for nr = nrList
    wModeList = repCell{nr,1};
    evdList = repCell{nr,2};
    [~,kmax] = max(evdList);
    dwRep(:,nrList==nr) = diff(wModeList(:,kmax));
end
lms = [min(dwRep(:)) max(dwRep(:))];
plot(dwRep(:,1),dwRep(:,2),'.','MarkerSize',7,'color',0.5*[1 1 1])
hold on
plot(lms,lms,'k-')
hold off
xlim(lms)
ylim(lms)
xlabel('{\Delta}w (rep 1)      ')
ylabel('{\Delta}w (rep 2)')
axis square



%%% rat data

% -- w(t) estimate --
%axes('Position',[hmarg0+2*(hsize+hmarg1)+hmarg1/4 vmarg0 hsize*1.75 2*vsize+vmarg1]);
h1 = hmarg0+2*(hsize+hmarg1)+hmarg1/4;
hlim = hsize*1.7;
v1 = vmarg0;
vlim = 2*vsize+vmarg1;
vcut = 0.03;
vdiv = (vlim-2*vcut)/3;

wArray = ratFit.wArray;
N = size(wArray,1);
npanList = {1,[2 3],4};
npanLabels = {'bias b','sensitivity a','stickiness h'};
%wlabels = {'b','a1','a2','h'};
wlabels = {'bias b','sensitivity a1','sensitivity a2','history dependence h'};

for nrow = 1:3
    axes('Position',[h1,v1+(3-nrow)*(vdiv+vcut),hlim,vdiv])
    for np = npanList{nrow}
        plot(1:N,wArray(:,np),'color',sevenColors(np,:),'LineWidth',2.5)
        hold on
    end
    plot([0 N],[0 0],'k:')
    hold off
    if(nrow<3)
        set(gca,'XTick',[]);
    end
    xlim([0 N])
    legend(wlabels(npanList{nrow}),'Location','NorthWest')
    legend('boxoff')
end
xlabel('trials')


% -- max evidence
axes('Position',[1-(hsize+hmarg1) vmarg0+vsize+vmarg1 hsize vsize]);
logEvdList = ratFit.evd;
plot(allsigbin,logEvdList-max(logEvdList),'ko-','LineWidth',2)
ylm0 = ylim;
hold on
plot(bestsigbin*[1 1],ylm0,'r--','LineWidth',2)
hold off
ylim(ylm0)
xlim([min(allsigbin) max(allsigbin)])
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(rem(ytk0,100)==0));
xtk0 = get(gca,'XTick');
set(gca,'XTick',xtk0(rem(xtk0,2)==0))
xlabel('log_{2} \sigma')
ylabel('log evd. (rel.)')
legend('log evd','max-evd','Location','SouthWest')
axis square



% -- BIC model selection
axes('Position',[1-(hsize+hmarg1) vmarg0 hsize vsize]);

BIClist = BIClist - max(BIClist(:));

plot(tauList,BIClist,'ko-','LineWidth',2)
myrange = [min(BIClist(:)) max(BIClist(:))];
ylim([myrange(1)-diff(myrange)/10 myrange(2)])
xlim([0 max(tauList)])
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(rem(ytk0,100)==0))
set(gca,'XTick',tauList)
xlabel('d (trials back)')
ylabel('BIC (rel.)')
axis square


set(findall(gcf,'-property','Fontsize'),'Fontsize',18)
set(gcf,'PaperPositionMode','auto') % match print size to screen


%%% final touch
axes('Position',[0 0 1 1])
axis off
hold on
panelfont = 24;
text(0.04,0.90,'A','FontWeight','Bold','FontSize',panelfont)
text(0.04,0.43,'B','FontWeight','Bold','FontSize',panelfont)
text(0.45,0.90,'C','FontWeight','Bold','FontSize',panelfont)
text(0.755,0.90,'D','FontWeight','Bold','FontSize',panelfont)
text(0.755,0.43,'E','FontWeight','Bold','FontSize',panelfont)
hold off



%% -------- with learning component --------

%% testing on simulated dataset, with true weights known

%%% check for existing file
sname = [datadir,'evdmax-test.mat'];
if(exist(sname,'file'))
    % skip
    disp('file already exists.');
    
else
    
    %%% simulate a policy-gradient-updating rat
    
    % set parameters
    
    N = 2000;
    truealpha = 2^-7;
    truesigma = 2^-7;
    
    
    % stimulus space
    xgrid1D = (55:10:95);
    xx = combvec(xgrid1D,xgrid1D)';
    nx = size(xx,1);
    
    xcenter = mean(xgrid1D); % typical scale of stimulus
    xstd = std(xgrid1D);
    
    xsetTrue = xx(or(diff(xx,[],2)==10,diff(xx,[],2)==-10),:);
    xsetTrue = (unique(xsetTrue,'rows')-xcenter)/xstd;
    nxsetTrue = size(xsetTrue,1);
    
    %%% draw responses
    iall = randsample(nxsetTrue,N,'true');
    xall = xsetTrue(iall,:);
    
    wInit = [0.4 -0.05 0.05];
    
    hterms = 1;
    if(hterms==1)
        zall = [1; sign(diff(xall(1:end-1,:),[],2))];
        xall = [xall zall];
        wInit = [wInit 1];
    end
    
    %%% generate simulated rat
    
    tback = 0;
    decay = 1;
    kappa = 1;
    
    params = struct('alpha',truealpha,'sigma',truesigma,...
        'tback',tback,'decay',decay,'kappa',kappa,'AT',false); % AT=false by default
    dims = struct('y',1,'g',size(xall,2)+1);
    
    [~,wSim,simdat,~] = getSimRat_active(params,dims,xall,wInit);
    
    
    %%% fit learning parameters (from simulated learner)
    
    K = size(wSim,2);
    
    % set learning parameter space
    
    alphabin = -9:0.5:-6;
    eta = 1;
    kappa = 1;
    tback = 0;
    
    showopt = 1;
    maxIter = 25; % 5/16/2016 adition
    
    % set prior width
    sigInit = 3;
    prsInit = zeros(K,1);
    
    allsigbin = -7; %[-8 -9 -10];
    allEvd = -Inf(numel(allsigbin),numel(alphabin));
    
    for ns = 1:numel(allsigbin)
        
        mysigbin = allsigbin(ns);
        mysigma = 2^mysigbin;
        wModeList = cell(size(alphabin,1),1);
        
        disp(' ');
        for nphi = 1:numel(alphabin)
            alpha = 2^alphabin(nphi);
            disp(['sigma 2^',num2str(mysigbin),' alpha 2^',num2str(alphabin(nphi))]);
            fullfitOpts = struct('showopt',showopt,'maxIter',maxIter,...
                'prsInit',prsInit,'sigInit',sigInit,...
                'alpha',alpha,'eta',eta,'kappa',kappa,'tback',tback);
            [wMode,Hess,logEvd,llstruct] = getMAP_RWprior(simdat,mysigma,fullfitOpts);
            
            allEvd(ns,nphi) = logEvd; %llstruct.logli + llstruct.logprior - llstruct.logpost;
            wModeList{nphi} = reshape(wMode,[N K]);
        end
        disp('done.');
        
    end
    
    [nsmax,nphimax] = find(allEvd==max(allEvd(:)));
    display([allsigbin(nsmax) alphabin(nphimax)]);
    
    %%% save data
    save(sname,'wModeList','allEvd',...
        'alphabin','allsigbin','truealpha','truesigma','sigInit','prsInit',...
        'simdat','wSim');
    
end

%% on external dataset [also simulated for now]

%%% load dataset
tempVar = load([basedir,'testdat.mat']);
alldat = tempVar.alldat;
dims = tempVar.dims;
% truealpha = tempVar.alpha;
% truesigma = tempVar.sigma;
clear tempVar
setTag = '_sim';

K0 = (dims.y)*(dims.g);

% add history variable
hterm = 1; % single-step-back history
if(hterm==0)
    newdat = struct('x',alldat.x,'y',alldat.y,'s',alldat.s,'allys',alldat.allys);
elseif(hterm==1)
    % include z as input
    newdat = struct('x',[alldat.x alldat.z],'y',alldat.y,'s',alldat.s,'allys',alldat.allys);
else
    error('option hterm not recognized.');
end

% set dimensions
ydim = numel(newdat.allys)-1;
gdim = size(newdat.x,2)+1;
K = ydim*gdim;
N = numel(newdat.y);

if(hterm ~= (K-K0))
    error('dimension mismatch: K');
end


%%% fit learning hyperparameters (from rat dataset)

% set learning hyperparameter space
alphabin = -7:-5; %-8:-4;
allsigbin = -7:-5; % -8:-4;
eta = 1;
kappa = 1;
tback = 0;

% set prior width
sigInit = 1;
prsInit = zeros(K,1);

showopt = 1;
maxIter = 25;

%%% check previous file
sname = [datadir,'evdmax',setTag,'_N',num2str(N),'.mat'];
if(exist(sname,'file'))
    % skip
    disp('file already exists.');
    
else
    
    allEvd = -Inf(numel(allsigbin),numel(alphabin));
    
    for ns = 1:numel(allsigbin)
        
        mysigbin = allsigbin(ns);
        mysigma = 2^mysigbin;
        
        for nphi = 1:numel(alphabin)
            
            alpha = 2^alphabin(nphi);
            
            disp(' ');
            disp(['sigma 2^',num2str(mysigbin),' alpha 2^',num2str(alphabin(nphi))]);
            if(allEvd(ns,nphi)>-Inf)
                disp('skipped');
                continue;
            end
            
            fullfitOpts = struct('showopt',showopt,'maxIter',maxIter,...
                'prsInit',prsInit,'sigInit',sigInit,...
                'alpha',alpha,'eta',eta,'kappa',kappa,'tback',tback);
            
            [wMode,~,logEvd,llstruct] = getMAP_RWprior(newdat,mysigma,fullfitOpts);
            
            allEvd(ns,nphi) = logEvd;
            wModeList{ns,nphi} = reshape(wMode,[N K]);
            
        end
        disp('done.');
        
    end
    
    [nsmax,nphimax] = find(allEvd==max(allEvd(:)));
    display([allsigbin(nsmax) alphabin(nphimax)]);
    
    %%% save data
    save(sname,'wModeList','allEvd',...
        'alphabin','allsigbin','sigInit','prsInit',...
        'newdat','setTag','N');
    
end

%% plot together

figure(2)
clf;
set(gcf,'Position',[100 100 1000 350])

hmarg = 0.08;
vmarg = 0.2;
hsize = (1-4*hmarg)/3;
vsize = (1-2*vmarg);

%%% simulated model

sname = [datadir,'evdmax-test'];
sVar = load(sname);

allEvd = sVar.allEvd;
alphabin = sVar.alphabin;
truealpha = sVar.truealpha;
wModeList = sVar.wModeList;
wSim = sVar.wSim;
clear sVar

N = size(wSim,1);
[~,nphimax] = find(allEvd==max(allEvd(:)));

%subplot(1,3,1)
axes('Position',[hmarg vmarg hsize vsize])
% ---- plot true simulated model ---
ax0 = gca;
plot(1:N,wSim,'-','color',0.5*[1 1 1],'LineWidth',2)
hold on
%set(gca,'ColorOrderIndex',1)
plot([0 N],[0 0],'k:')
hold off
xlm0 = xlim;
ylm0 = ylim;
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(ytk0==floor(ytk0)))
xlabel('trials')
ylabel('model weights')
legend('true','Location','SouthWest')
legend('boxoff')
% ---- plot estimated model ---
axes('position',ax0.Position)
axis off
hold on
plot(1:N,wModeList{nphimax},'--','color',0*[1 1 1],'LineWidth',2)
hold off
xlim(xlm0);
ylim(ylm0);
%legend(strcat('estimated\_',wlabels),'Location','South')
legend('estimated','Location','SouthEast')
legend('boxoff')

%figure(6)
%subplot(1,3,2)
axes('Position',[2*hmarg+hsize vmarg hsize vsize])
plot(alphabin,allEvd'-max(allEvd(:)),'ko-','LineWidth',2)
ylm0 = ylim;
hold on
plot(log2(truealpha)*[1 1],ylm0,'r:','LineWidth',2)
plot(alphabin(nphimax)*[1 1],ylm0,'r--','LineWidth',2)
hold off
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(ytk0==floor(ytk0)))
xlabel('log_{2} \alpha')
ylabel('log evidence (rel.)')
legend('log evidence','true \alpha','max-evd','Location','SouthWest')
legend('boxoff')


%%% --- max evd for rat data ---

N = numel(alldat.y);

axes('Position',[3*hmarg+2*hsize vmarg hsize*1.1 vsize])

sname = [datadir,'evdmax',setTag,'_N',num2str(N),'.mat'];
sVar = load(sname);
allEvd = sVar.allEvd;
alphabin = sVar.alphabin;
allsigbin = sVar.allsigbin;
clear sVar

[nsmax,nphimax] = find(allEvd==max(allEvd(:)));

colormap gray
imagesc(alphabin,allsigbin,allEvd-max(allEvd(:)))
set(gca,'YDir','normal')
set(gca,'YTick',allsigbin)
set(gca,'XTick',alphabin)
hold on
plot(alphabin(nphimax),allsigbin(nsmax),'r*','MarkerSize',16)
hold off
xtk0 = get(gca,'XTick');
set(gca,'XTick',xtk0(xtk0==floor(xtk0)))
ytk0 = get(gca,'YTick');
set(gca,'YTick',ytk0(ytk0==floor(ytk0)))
ylabel('log_{2} \sigma')
xlabel('log_{2} \alpha')
%axis square
c0 = colorbar; % EastOutside;
c0.Label.String = 'log evidence (rel.)';
c0.Location = 'EastOutside';


set(findall(gcf,'-property','fontsize'),'fontsize',18)
set(gcf,'paperpositionmode','auto')


%%% final touch
axes('Position',[0 0 1 1])
axis off
hold on
text(0.03,0.8,'A','FontWeight','Bold','FontSize',22)
text(0.34,0.8,'B','FontWeight','Bold','FontSize',22)
text(0.65,0.8,'C','FontWeight','Bold','FontSize',22)
hold off

