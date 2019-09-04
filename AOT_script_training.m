% AOT_script_training.m
% animal behavior training simulation

% for NIPS paper: Adaptive optimal training of animal behavior (May 2016)
% rearranged to share (Apr 2017)

% 2016-2017 Ji Hyun Bak

%% initialize

clear all;
clc;

setpaths;
setcolors;
setGoodRand;

%% simulation setup

%%% set up stimulus spaces 

% stimulus space
xgrid1D = (55:10:95);
xx = combvec(xgrid1D,xgrid1D)';
xx = xx(diff(xx,[],2)~=0,:); % exclude diagonal
nx = size(xx,1);

%%% stimulus spaces

% full and narrows
xsetFull = xx;
xsetNarrow = xx(or(diff(xx,[],2)==10,diff(xx,[],2)==-10),:);

% scaled stimulus space
xcenter = mean(xgrid1D); % typical scale of stimulus
xstd = std(xgrid1D);
xscaled = (xgrid1D-xcenter)/xstd;


%%% set overall parameters

wInit = [-1 0 0 1];

K = numel(wInit); % number of parameter types
dims = struct('y',1,'g',K); % binary response

% set data size
N = 500; % number of trials
numRep = 3; % repetition

% set hyperparameters
alpha = 0.05; % learning rate
sigma = 0.01; % noise strength

%%% set training options

stimPatternList = {'full','narrow'}; % stimulus space
AToptionList = [true false]; % active training option


%% training with a simulated policy-gradient learner 
% based on the version 4/27/2015, in ratDataView_copy.m

wSimCell = cell(numRep,1);
simdatCell = cell(numRep,1);

for nr = 1:numRep
    
    disp(' ');
    disp(['** rep ',num2str(nr),' **']);
    
    wSimTwoWays = cell(numel(AToptionList),numel(stimPatternList));
    simdatTwoWays = cell(numel(AToptionList),numel(stimPatternList));
    
    for np = 1:numel(stimPatternList)
        
        %%% set stimulus space
        
        stimPattern = stimPatternList{np};
        if(strcmp(stimPattern,'narrow'))
            xsetTrue = xsetNarrow;
        elseif(strcmp(stimPattern,'full'))
            xsetTrue = xsetFull;
        else
            error('unknown stimulus pattern.');
        end
        xsetTrue = (unique(xsetTrue,'rows')-xcenter)/xstd;
        nxsetTrue = size(xsetTrue,1);
        
        
        %%% prepare stimuli
        
        % simulate full session
        iall = randsample(nxsetTrue,N,'true');
        xall0 = xsetTrue(iall,:);
        xprev0 = sign(diff(xsetTrue(randsample(nxsetTrue,1),:)));
        zall = [xprev0; sign(diff(xall0(1:end-1,:),[],2))];
        xall = [xall0 zall]; % with single-step-back history term
        xset = unique(xall,'rows'); % extended stimulus set
        
        
        %%% run both trainings (active and not)
        
        for na = 1:numel(AToptionList)
            
            myAT = AToptionList(na);
            display(['stimSpace=',stimPattern,', activeTraining=',num2str(myAT)]);
            
            params = struct('alpha',alpha,'sigma',sigma,'AT',myAT);
            % when AT is true, xall is only used for extracting
            % the total # trials & the stimulus space
            
            % run simulation
            [~,wSim,simdat,~,~] = getSimRat_active(params,dims,xall,wInit);
            wSimTwoWays{na,np} = wSim;
            simdatTwoWays{na,np} = simdat;
        end
        
    end
    
    wSimCell{nr} = wSimTwoWays;
    simdatCell{nr} = simdatTwoWays;
    
end

disp('done.');

%% plot training performances (single run)

nr = 1;
wSimTwoWays = wSimCell{nr};
simdatTwoWays = simdatCell{nr};


%%% prepare performance comparison data

ssp = N/10;

perfData = cell(numel(stimPatternList),3);

probmax = NaN;
probmin = NaN; 
divmax = NaN;
divmin = NaN;

for np = 1:numel(stimPatternList)
    
    successRate = zeros(N,2);
    expReward = zeros(N,2);
    divKL = zeros(N,2);
    
    %%% set stimulus space
    
    stimPattern = stimPatternList{np};
    if(strcmp(stimPattern,'narrow'))
        xsetTrue = xsetNarrow;
    elseif(strcmp(stimPattern,'full'))
        xsetTrue = xsetFull;
    else
        error('unknown stim pattern.');
    end
    xsetTrue = (unique(xsetTrue,'rows')-xcenter)/xstd;
    xset = combvec(xsetTrue',[-1 1])'; % extend by adding history variable
    
    % input distribution (uniform, assuming a "general" stimulus train)
    nxset = size(xset,1);
    px0 = ones(nxset,1)/nxset;
    
    for na = 1:numel(AToptionList) % (myAT+1)
        
        simdat = simdatTwoWays{na,np};
        wSim = wSimTwoWays{na,np};
        
        % success rate
        successRate(:,na) = mysmooth(simdat.s,ssp);
        
        
        %%% expected reward & KL divergence
        
        rho = zeros(N,1); % expected reward
        dkl = zeros(N,1); % KL divergence
        for t = 1:N
            rsum = 0;
            dklsum = 0;
            for xi = 1:nxset % "expected" reward, irrespective of actual stimulus
                myx = xset(xi,:);
                myv = [1 myx]*wSim(t,:)';
                myp = 1./(1+exp(-myv));
                myr = (myx(2)>myx(1))*myp + (myx(2)<myx(1))*(1-myp);
                rsum = rsum + px0(xi)*myr;
                mydkl = (myx(2)>myx(1))*log(1/myp) + (myx(2)<myx(1))*log(1/(1-myp));
                dklsum = dklsum + mydkl;
            end
            rho(t,:) = rsum;
            dkl(t,:) = dklsum/nxset;
        end
        expReward(:,na) = rho;
        divKL(:,na) = dkl;
        
    end
    
    % merge performance data
    perfData(np,:) = {successRate,expReward,divKL};
    
    % extract bounds
    probdata = [expReward(:);successRate(:)];
    probmax = max(probmax,max(probdata));
    probmin = min(probmin,min(probdata));
    
    divmax = max(divmax,max(divKL(:)));
    divmin = min(divmin,min(divKL(:)));
    
end


%%% plot weight evolution data

np = 1;
stimPattern = stimPatternList{np};

lineStyleList = {'-','--'};
npanList = {1,[2 3],4};
nrowLabels = {'bias b','sensitivity a','stickiness h'};
wlegs = {'b',{'a1','a2'},'h'};

figure(1)
clf;
set(gcf,'Position',[100 100 1050 450])
set(gcf,'DefaultAxesColorOrder',[0 0 0; 0.3*[1 1 1]])

totCols = 4;
hmarg0 = 0.05;
hmarg1 = 0.10; % between A and B/C
hcut = 0.06; % within B/C
vmarg0 = 0.10;
vmarg1 = 0.15; % between B and C
vcut = 0.03; % within A

%%% --- weights ---

hsize = (1-2*hmarg0-hmarg1-2*hcut)/4.5;
vsizeA = (1-2*vmarg0-2*vcut)/3;

for nrow = 1:3
    axes('Position',[hmarg0 vmarg0+(3-nrow)*(vsizeA+vcut) hsize*1.6 vsizeA])
    yrange = [NaN NaN];
    for na = 1:numel(AToptionList)
        wgtData = wSimTwoWays{na,np};
        for npan = fliplr(npanList{nrow})
            plot(1:N,wgtData(:,npan),lineStyleList{na},'LineWidth',2,...
                'color',sevenColors(npan,:))
            hold on
            yrange = [min(yrange(1),min(wgtData(:,npan))) ...
                max(yrange(2),max(wgtData(:,npan)))];
        end
    end
    plot([0 N],[0 0],'k:')
    hold off
    xlim([0 N])
    yrange = [min(0,yrange(1)) max(0,yrange(2))];
    if(nrow==2)
        ymargin = 0.1;
    else
        ymargin = 0.5;
    end
    
    yrange = yrange + diff(yrange)*ymargin*[-1 1];
    ylim(yrange)
    ylabel(nrowLabels{nrow})
    ytk0 = get(gca,'YTick');
    if(nrow==2)
        set(gca,'YTick',ytk0(rem(ytk0,2)==0))
        myloc = 'East';
    else
        set(gca,'YTick',ytk0(ytk0==floor(ytk0)))
        myloc = 'Best';
    end
    legend(fliplr(wlegs{nrow}),'Location',myloc)
    legend('boxoff')
    if(nrow<3)
        set(gca,'XTick',[])
    end
end
xlabel('trials')



%%% plot performance comparison data

plotmax = ceil(10*probmax)/10;
plotmargin = plotmax-probmax;
plotmin = max(0,probmin-plotmargin);

plotmin = plotmin - (plotmax-plotmin)*0.05;
plotmax = plotmax + (plotmax-plotmin)*0.05;

dplotmax = ceil(10*divmax)/10;
dplotmin = floor(10*divmin)/10;

legsAT = {'AlignMax','Random'};
titlesAT = {'success rate','expected reward','KL divergence'};

vsizeBC = (1-2*vmarg0-vmarg1)/2;

for np = 1:2
    for ncol = 1:3
        h0 = 1-hmarg0-hsize-(3-ncol)*(hsize+hcut);
        v0 = vmarg0+(2-np)*(vsizeBC+vmarg1);
        axes('Position',[h0 v0 hsize vsizeBC]);
        plotData = perfData{np,ncol};
        plot(1:N,plotData(:,1),'-','LineWidth',2.5)
        hold on
        plot(1:N,plotData(:,2),'--','LineWidth',2)
        if(ncol==1 || ncol==2)
            plot([0 N],[0.5 0.5],'k:')
            ylim([plotmin plotmax])
        elseif(ncol==3)
            ylim([dplotmin dplotmax])
            ytk0 = get(gca,'YTick');
            set(gca,'YTick',ytk0(rem(ytk0,1)<0.01))
        end
        hold off
        xlim([0 N])
        ylim([0 1])
        %title(titlesAT{ncol})
        ylabel(titlesAT{ncol})
        if(np==2)
            xlabel('trials')
        end
        leglocs = {'NorthWest','SouthEast','NorthEast'};
        legend(legsAT,'Location',leglocs{ncol})
        % legend exceptions
        if(np==2 && ncol==3)
            legend(legsAT,'Location','SouthWest')
        elseif(np==1 && ncol==1)
           legend(legsAT,'Location','East')
        end
        legend('boxoff')
    end
end

set(findall(gcf,'-property','fontsize'),'fontsize',16)
set(gcf,'PaperPositionMode','auto') % match print size to screen


%%% final touch
axes('Position',[0 0 1 1])
axis off
hold on
text(0.05,0.945,'A. model weights','FontWeight','Bold','FontSize',20)
text(0.34,0.945,'B. full stimulus space','FontWeight','Bold','FontSize',20)
text(0.34,0.47,'C. reduced stimulus space','FontWeight','Bold','FontSize',20)
hold off

%% plot how AlignMax works (take average)

% take average

wSimSum = zeros(N,4);
xSimSum = zeros(N,2);
sSimSum = zeros(N,2);

for nr = 1:numRep

    simdatTwoWays = simdatCell{nr};
    wSimTwoWays = wSimCell{nr};
    
    %%% active on full set
    na = 1; np = 1;
    xset = xsetFull;
    
    mywSim = wSimTwoWays{na,np};
    mysimdat = simdatTwoWays{na,np};
    myxSim = mysimdat.x;
    myxSim = myxSim(:,1:2); % rearrange xSim: 10/17/2016 correction
    mysSim = mysimdat.s;
    
    wSimSum = wSimSum + mywSim;
    xSimSum = xSimSum + myxSim;
    sSimSum = sSimSum + mysSim;
    
end

wSim = wSimSum/numRep;
xSim = xSimSum/numRep;
sSim = sSimSum/numRep;


figure(2)
clf;
set(gcf,'Position',[100 50 600 650])

subplot(14,1,1:3)
plot(1:N,wSim(:,4),'color',sevenColors(4,:),'LineWidth',2)
hold on
plot(1:N,wSim(:,1),'color',sevenColors(1,:),'LineWidth',2)
plot([0 N],[0 0],'k:')
hold off
axis tight
ylm0 = ylim;
ylm0 = ylm0 + 0.05*diff(ylm0)*[-1 1];
ylim(ylm0)
set(gca,'XTick',[])
set(gca,'YTick',[-1,0,1],'YTickLabel',{'-1','   0','1'})
ylabel('weights')
legend('stickiness h','bias b','Location','SouthEast')
legend('boxoff')
title('A. weights driven by input statistics')

subplot(14,1,4:6)
plot(mysmooth(0.5+sign(diff(xSim,[],2))/2,100),'k-','LineWidth',2)
hold on
plot(mysmooth([0.5; ...
    double(diff(xSim(1:end-1,:),[],2)==diff(xSim(2:end,:),[],2))],100)...
    ,'-','color',sevenColors(7,:),'LineWidth',2)
plot([0 N],0.5*[1 1],'k:')
hold off
axis tight
ylm0 = [0 1];
ylm0 = ylm0 + 0.05*diff(ylm0)*[-1 1];
ylim(ylm0)
legend('prob x increasing','prob x staying','Location','SouthEast')
legend('boxoff')
xlabel('trials')
ylabel('input statistics')

subplot(14,1,9:11)
plot(1:N,diff(xSim,[],2)/2,'.','color',0.3*[1 1 1],'LineWidth',1)
hold on
plot([0 N],[0 0],'w:')
hold off
set(gca,'XTick',[])
ylabel('(x2-x1)/2')
ylim([-2 2])
title('B. AlignMax choice of optimal stimuli')

subplot(14,1,12:14)
plot(1:N,mean(xSim,2),'.','color',0.3*[1 1 1],'LineWidth',1)
hold on
plot([0 N],[0 0],'w:')
hold off
ylim([-2 2])
xlabel('trials')
ylabel('(x2+x1)/2')

set(findall(gcf,'-property','fontsize'),'fontsize',14)
