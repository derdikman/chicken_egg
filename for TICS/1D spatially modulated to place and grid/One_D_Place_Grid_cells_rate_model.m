function [ ] = One_D_Place_Grid_cells_rate_model( trials )
% 
% Simulating the joint dynamics of place cells and grid cells with symmetric connections between them
% This version is modified for the SCAN model, and 
% uses 1D spatially modulated input cells into the place cells,
% Code generated for Morris & Deridkman, TICS, 2023
% (The code is modified from original code for Agmon and Burak, eLife,
% 2020)
%

if nargin<1
    trials=0;
end
rng(trials);

N=4800; % # of Place cells
Ratios=[3,4,5]; % ratios of arena size with grid cells modules - such that mod(PC,Ratios)=0.
Lambdas=N./Ratios; % the corresponding grid spacings

Online_bump_estimate=0; % continuously estimate the place & grid cells bumps vs time (for pc path integration & gc coupling modules) set to 1
Online_GC_rate_register_initial_cond=0; % continuously register the rate of grid cells in the intial condition and the maximal firing rate vs time (for blobs variability)? set to 1
Online_Rates_register=0; % continuously register the rate of all grid cells and place cells vs time? set to 1
Online_maps_scoring=0; % continuously register the scores for each map? set to 1
%% Load the W matrix (connectivity within grid cells)
load('Grids_RF_Analytic.mat'); % loading the grid cells files
Grid_Bumps=Grids_RF_Analytic.Bump_2pop; % bump on most left grid cell rate
Grid_Locations=Grids_RF_Analytic.Grid_Locations; % the locations of the grid cells
W=Grids_RF_Analytic.W_2pop; % the connectivity matrix, identical to all modules
W=.05.*W; % same scaling to W as in 6 maps original simulation

Even_indices = 2:2:size(W,1); % indices of left population grid cells
Odd_indices = 1:2:size(W,1)-1; % indices of right population grid cells

mo=size(Grid_Locations,1); % # of modules
n=size(Grid_Locations,2); % # of grid cells in each module
%% Loading and permutaing (if necessary) the correlation matrix
load('Ms.mat'); % loading the correlation matrices
M_Corr=Ms{1}; % 1st module correlation matrix
M_Corr_2=Ms{2}; % 2nd module correlation matrix
M_Corr_3=Ms{3}; % 3rd module correlation matrix

M_Corr=M_Corr.*(1/(max(max(M_Corr)))); % Normalizing the correlation matrix to values in range [0,1];
M_Corr_2=M_Corr_2.*(1/(max(max(M_Corr_2)))); % Normalizing the correlation matrix to values in range [0,1];
M_Corr_3=M_Corr_3.*(1/(max(max(M_Corr_3)))); % Normalizing the correlation matrix to values in range [0,1];

M_Corr=M_Corr./6; % pre-scaling factor to M to fit the 6 maps original simulation value
M_Corr_2=M_Corr_2./6; % pre-scaling factor to M to fit the 6 maps original simulation value
M_Corr_3=M_Corr_3./6; % pre-scaling factor to M to fit the 6 maps original simulation value

M_Original=M_Corr; % M_Original is used in case Maps>0
M_Original_2=M_Corr_2; % M_Original is used in case Maps>0
M_Original_3=M_Corr_3; % M_Original is used in case Maps>0

D_1 = [mean(sum(M_Corr,1)), mean(sum(M_Corr_2,1)), mean(sum(M_Corr_3,1)) ]; %  coefficient D1 - sum of columns
D_2 = [mean(sum(M_Corr,2)), mean(sum(M_Corr_2,2)), mean(sum(M_Corr_3,2)) ]; %  coefficient D2 - sum of rows
%% Calculating the distances for the Place cells top left neuron
D_most_left=zeros(1,N); % connectivity matrix for the most left neuron
D=1;

for d=1:N
    XDist=d-D; % d>=D
    if XDist>N/2
        XDist=N-d+D;
    end
    D_most_left(d)=XDist; % The distances of the most left neuron from all other neurons
end
%% Genela2  
% create weakly spatially modulated input layer

load('steady_1d_bump_analytic_with_Grids_2pop.mat'); % bump rate on most left place cell

% 
% The input SM layer will be of size 960 (N), and the connectivity matrix
% U (SM -> PC) will be of size NxN, based on the correlation between rate
% maps. SM is a set of lpf white noise cells.
%
% 

% Generate broad spatially modulated cells

% spatial_gaussian_width=500;
% y1 = wgn(N,n,0);
% winwin = gausswin(spatial_gaussian_width);
% y = filter(winwin,1,y1);
% y(y<0)=0;
% S_SM = y;
% 
% % correlate with all circshift of generic place cell
% 
% for i=1:N
%     if (i/100 == floor(i/100))
%         i
%     end
%     PC_shifted = circshift(steady_1d_bump_analytic_with_Grids_2pop(:),i);
%     for j=1:n
%          tmp = corrcoef(PC_shifted,S_SM(:,j)');
%          M_PC_SM(j,i) = tmp(2,1);
%     end
% end
% 
% save PC_SM_DATA M_PC_SM S_SM spatial_gaussian_width
load PC_SM_DATA

% rectify weights

M_PC_SM_full = M_PC_SM;
M_PC_SM(M_PC_SM<0) = 0;

% divide by 6 (why not)

%M_PC_SM = M_PC_SM / 6;

%% Calculating the connectivity fucntion (1 Gaussian with global inhibition)
sig=30*4; %30*5;
a=50*4; %50*2;

Gaus=a .* (1/(sqrt(2*pi) *sig)) .* exp( (-D_most_left.^2)./(2*sig^2)); % Gaussian

C_0=-800*4800; % conectivity parmeter
Total_J_top_left=C_0/N; % =C
Excite=sum(Gaus)-Gaus(1); % total excitation (after substracting the connection from the neuron to itself)

inhibition=Total_J_top_left-Excite;% leftover for inhibition
h=inhibition/(N-1); % individual neuron inhibition parameter distributed equally among all other neurons
%% Calculating the connectivity matrix J^0
J_most_left=Gaus+h;
J_most_left(D)=0;

J=zeros(N); % the full connectivity matrix
for i=1:N    
    J(i,:)=circshift(J_most_left,i-1,2); % connections of the i,j neuron    
end

J=J.*(1/8); % pre-scaling factor to J to fit the 6 maps original simulation value
Total_J_top_left_fixed=mean(sum(J)); % sum of a row/column in the matrix J
%% Interpolating for the case we use a bump initial condition for the grid cells
GC_RFS=zeros(mo,N); % will contain the grid cells interpolated bumps in the place cells resulotion
for k=1:mo % running on all modules
 
    lambda=Grid_Locations(k,end); % the grid spacing of the k module    
    gc_rf=Grid_Bumps(:,k); % the defult receptive field of most left grid cell
    
    %% Finding the elements we need to interpolate and performing interpolation
    query=zeros(1,lambda-numel(gc_rf)); % # of elements we need to interpolate
    if numel(query)>0 % meaning we do need to interpolate             
        count=1;
        for j=1:lambda
            if sum(j==Grid_Locations(k,:))==0 % the j'th element does not exist
                query(count)=j; % so register it
                count=count+1;
            end
        end
        
        query_val=interp1(Grid_Locations(k,:),gc_rf,query); % interpolated values
    
        All_X=[Grid_Locations(k,:) query]; % all the places from 1 to lambda in 2 clusteres 
        All_Y=[gc_rf' query_val]; % all corresponding values
    
        gc_rf=zeros(1,lambda); % will contain the receptive filed of most left grid cell
        for j=1:lambda
            in=find(j==All_X); % the relevant index
            gc_rf(j)=All_Y(in); % constructing the interpolated receptive filed
        end
    
    end    
    
    GC_RF=zeros(1,N); % will contain the grid cell receptive filed over all the arena [same resulotion as place cells]
    for l=1:N/lambda
        GC_RF(1+(l-1)*lambda:l*lambda)=gc_rf;
    end
    GC_RFS(k,:)=GC_RF; % interpolated activity over neuron #1 (most left grid cell) for each module
end
%% Permute rows and cols by matrix multiplication
Maps=5; % # of different maps we want to generate except for the "original" one

J_Original=J; % the template to be used to permute on top, we don't save all the maps

PERMUTATIONS=zeros(Maps,N); % all the random permutations
shifts_loc=round(N.*rand(mo,Maps)); % the location in the arena [cm] where each modules is moved to for each map
Grids_permutations=zeros(mo,Maps); % will contaion the coordinated neurons permutations [0,n] (shifts) for all modules and maps
% rng(trials); % differnt maps will be sumed up

alpha=0.0103;%  % pre-factor - M is already divided by 6 but we could divide alpha instead to adjust to the original 6 maps simulation value
beta=(-0.004/6); % global inhibition - adjusted to the original 6 maps simulation value

alpha_2=alpha; % pre-factor
beta_2=beta; % global inhibition
alpha_3=alpha; % pre-factor
beta_3=beta; % global inhibition

M=(alpha.*M_Corr)+beta; % the 0 map connectivity matrix 1st module 
M_2=(alpha_2.*M_Corr_2)+beta_2; % the 0 map connectivity matrix 2nd module
M_3=(alpha_3.*M_Corr_3)+beta_3; % the 0 map connectivity matrix 3rd module

M_PC_SM = (alpha.*M_PC_SM); %+beta;  % The non-grid spatially modulated cells to PC connectivity matrix (Genela)

strength_1=0; % do we strength only the first map connections (to show mixed states would net emerge now during depolarizaion)
% if yes then set to 1, if no then set to 0

Permute_Matrices=zeros(N,N,Maps); % will contain all the permute matrices to transfer from the 0 base to any other base
for i=1:Maps
    
    rng(i); % the same maps as before + 1 map addition sumed up
    
    Permutation=randperm(N); % vector of the random permutations
    PERMUTATIONS(i,:)=Permutation;
    
    permute_matrix=zeros(N);
    for m=1:N
        permute_matrix(m,Permutation(m))=1;
    end
    
    J_permuted=permute_matrix*J_Original*permute_matrix'; % because inverse_permute_matrix=permute_matrix'
    
    % performing the grid cells permutations (independent shifts per module)
    bump_ic=circshift(GC_RFS(1,:)',shifts_loc(1,i)-1);
    bump_ic_2=circshift(GC_RFS(2,:)',shifts_loc(2,i)-1);
    bump_ic_3=circshift(GC_RFS(3,:)',shifts_loc(3,i)-1);

    Bump_IC=[bump_ic bump_ic_2 bump_ic_3];
    S_GC_Interpolated=zeros(n,mo); % will contain the activities of grid cells of the relevant modules
    loc_max=zeros(1,mo); % will contain the relevant coordinated shifts for all modules for this current map
    for k=1:mo % running on all modules 
        S_GC_Interpolated(:,k)=Bump_IC(Grid_Locations(k,:),k);
        [~,loc_max(k)]=max(S_GC_Interpolated(:,k)); % the neuron # with the maximal firing rate around location shifts_loc
    end
    Grids_permutations(:,i)=loc_max; % registering all the coordinated shifts of all modules for the i'th map

    M_permuted=circshift(M_Original,[loc_max(1),0]);
    M_permuted_2=circshift(M_Original_2,[loc_max(2),0]);
    M_permuted_3=circshift(M_Original_3,[loc_max(3),0]);
    
    % performing the place cells permutations on the mutual connectivity matrix
    M_permuted=M_permuted*permute_matrix';  % Genela3 - think how to permute our SM input
    M_permuted_2=M_permuted_2*permute_matrix';
    M_permuted_3=M_permuted_3*permute_matrix';
    
    M_permuted = (alpha.*M_permuted)+beta; % the permuted map connectivity matrix 1st module 
    M_permuted_2 = (alpha_2.*M_permuted_2)+beta_2; % the permuted map connectivity matrix 2nd module 
    M_permuted_3 = (alpha_3.*M_permuted_3)+beta_3; % the permuted map connectivity matrix 3rd module
    
    J=J+J_permuted;% summing the maps one on top
    
    % strengthing only the first map to show that now depolarization has almost no effect
    if i==1 && strength_1==1 % we set the first map to have double effect then all the other maps
        M=2.*M;
        M_2=2.*M_2;
        M_3=2.*M_3;
        strength_1=2;
    end
    
    M=M+M_permuted;% summing the maps one on top
    M_2=M_2+M_permuted_2;% summing the maps one on top
    M_3=M_3+M_permuted_3;% summing the maps one on top
%     disp(i);

    Permute_Matrices(:,:,i)=permute_matrix;
end

if strength_1==2 % if we strenghthen first map we need to normalize it now (with the same factor 2 for 6 total maps -> (6/7) 
    M=M.*(6/7);
    M_2=M_2.*(6/7);
    M_3=M_3.*(6/7);
end
%% Dynamic equation parametesrs

gamma_pc_to_gc = 50; % 25;%50; % how much grid cells weigh the place cells input
gamma_gc_to_pc = 4; % 6;%4; % how much place cells weigh the grid cells inputs

Tau_GC=0.015;  % synaptic time constant of grid cells [Sec]
Tau_PC=Tau_GC; % synaptic time constant of place cells [Sec]

deltat=0.0002; % time interval (factor of 75 from tau) [Sec]
Tt=1;%2.5;% % for control with signal velocity; % +(6.2-1.5); % 6.4 time in seconds we wish to simulate [Sec] for one period of path integration with depolarization (after one sec)
% 0.5+2.05*3; % time in seconds we wish to simulate [Sec] (2.05sec is the time takes for one circshift period with path_jumps=10 speed)
Time=round(Tt/deltat); % Total time of the realization [iterations] 


mean_r_PC = mean(steady_1d_bump_analytic_with_Grids_2pop); % mean place cells firing rate from a single map [Hz]

mean_r_GC = mean(Grid_Bumps(:,1));% mean grid cells 1st module firing rate from a single map [Hz] - identical (good approximation) for all modules in a single map
mean_r_GC_2 = mean(Grid_Bumps(:,2)); % mean grid cells 2nd module firing rate from a single map [Hz] - identical (good approximation) for all modules in a single map
mean_r_GC_3 = mean(Grid_Bumps(:,3)); % mean grid cells 3rd module firinfg rate from a single map [Hz] - identical (good approximation) for all modules in a single map
% mean_r_GC is actually the mean of second module firing rate from a single map

g_I = gamma_gc_to_pc * (alpha * D_1(1) + beta*n ) * mean_r_GC; % the mean field values of input to place cells from grid cells - 1st module
g_I_2 = gamma_gc_to_pc * (alpha_2 * D_1(2) + beta_2*n ) * mean_r_GC_2; % the mean field values of input from grid cells - 2nd module
g_I_3 = gamma_gc_to_pc * (alpha_3 * D_1(3) + beta_3*n ) * mean_r_GC_3; % the mean field values of input from grid cells - 3rd module

% Inputs that we want, taken from the 6 maps original simulation
% I_GC=260-150; % external input to all GC 1st module
% I_GC_2=185-150; % external input to all GC 2nd module
% I_GC_3=150-150;
% I_PC=250;

%I_0_PC= -10; %-20 % works well in a single map and generate close value to the 6 maps original simulation value written 2 lines above
I_0_PC = 10;  % Genela
I_PC=I_0_PC - (Maps+1-1)*( Total_J_top_left_fixed * mean_r_PC + g_I +g_I_2 + g_I_3 ); % L=Maps+1, Total_J_top_left=C,

I_0_GC = -5; %-70; % works well in a single map and generate close value to the 6 maps original simulation value written 8 lines above
I_0_GC_2 = -5; %-70; % works well in a single map and generate close value to the 6 maps original simulation value written 8 lines above
I_0_GC_3 = -5; %-70; % works well in a single map and generate close value to the 6 maps original simulation value written 8 lines above
I_GC = I_0_GC- gamma_pc_to_gc*(Maps+1-1) * (alpha * D_2(1) + beta*N ) * mean_r_PC ; % external input to all GC 1st module
I_GC_2 = I_0_GC_2- gamma_pc_to_gc*(Maps+1-1) * (alpha_2 * D_2(2) + beta_2*N ) * mean_r_PC ; % external input to all GC 2nd module
I_GC_3 = I_0_GC_3- gamma_pc_to_gc*(Maps+1-1) * (alpha_3 * D_2(3) + beta_3*N ) * mean_r_PC ; % external input to all GC 3rd module


S_GC=zeros(n,2); % Activity (rate) of the grid cells only for current and previous states (markov dynamics) - 1st module
S_GC_2=zeros(n,2); % Activity (rate) of the grid cells only for current and previous states (markov dynamics) - 2nd module
S_GC_3=zeros(n,2); % Activity (rate) of the grid cells only for current and previous states (markov dynamics) - 3rd module

S_PC=zeros(N,2); % Activity (rate) of the place cells only for current and previous states (markov dynamics)

%% Initial condition
rng(trials);
phases=round(N*rand(1,4)); % the neurons around we put the initial bump condition

C=1; % maxiaml inital synaptic activity

% Place cells bump initial condition (in the first map)
S_PC(:,1)=steady_1d_bump_analytic_with_Grids_2pop(:); % making it the initial condition
S_PC(:,1)=C.*(circshift(S_PC(:,1),phases(1)-1)); % or round(rand.*PC)

% Permuted initial condition
base_PC=0; % if we want to set the place cells inital conditon to a different base, can be [0,Maps]
if Maps>0 && base_PC>0
 
    permute_matrix = Permute_Matrices(:,:,base_PC);
    
    S_PC(:,1)=permute_matrix*S_PC(:,1); % inital activity at the 2nd base
end

% Settting Grid bumps
bump_ic=circshift(GC_RFS(1,:)',phases(1)-1);
bump_ic_2=circshift(GC_RFS(2,:)',phases(1)-1);
bump_ic_3=circshift(GC_RFS(3,:)',phases(1)-1);

Bump_IC=[bump_ic bump_ic_2 bump_ic_3];

S_GC_Interpolated=zeros(n,mo); % will contain the activities of grid cells of the relevant modules
for k=1:mo % running on all modules 
    S_GC_Interpolated(:,k)=Bump_IC(Grid_Locations(k,:),k);
end
    
S_GC(:,1)=C.*(S_GC_Interpolated(:,1)); % making it the initial condition - 1st module
S_GC_2(:,1)=C.*(S_GC_Interpolated(:,2)); % making it the initial condition - 2nd module
S_GC_3(:,1)=C.*(S_GC_Interpolated(:,3)); % making it the initial condition - 3rd module

base_GC=base_PC; % if we want to set the grid cells initial condition to a different base
if Maps>0 && base_GC>0
    S_GC(:,1)=circshift(S_GC(:,1),[Grids_permutations(1,base_GC),0]);
    S_GC_2(:,1)=circshift(S_GC_2(:,1),[Grids_permutations(2,base_GC),0]);
    S_GC_3(:,1)=circshift(S_GC_3(:,1),[Grids_permutations(3,base_GC),0]);
end



%% The map in which we plot the rates
plot_in_base=0; % the base we want to plot the rates in, number from [0,Maps]

if Maps>=plot_in_base && plot_in_base>0 % if we want a base which is not the 0 base
    
     plot_in_base_matrix=Permute_Matrices(:,:,plot_in_base); % picking the relevant permutationg matrix
end
%% Generating movie
generate_movie=1; % want to generate movie of a single base? set to 1: want to generate movie of all bases simultaneuosly? set to 11
if generate_movie==1 || generate_movie==11
    fig=figure(1);
%     set(gcf,'units','normalized','outerposition',[0.25 0.25 .5 .5]); % make it full screen [0 0 1 1]
%     set(0, 'DefaultFigureVisible', 'off');
    vidfile=VideoWriter('Control.mp4','MPEG-4');
    vidfile.FrameRate = 5;
    open(vidfile);
    spatial_factor=0.04; % distance [cm] between two adjacent place cells
    
    Grid_All_Centers_movie=cell(1,mo); % will contain all the location grid cell encodes
    for k=1:mo
        l=Ratios(k); % the ratio between arena size to grid spacing (= # of bumps)
        lambda=Lambdas(k); % the relevant grid spacing
        All_Centers=zeros(1,l*n); % will contain the ceneters of grid cells
        count=0;
        for i=1:l
            All_Centers(1+(i-1)*n:i*n)=Grid_Locations(k,:)+count*lambda;
            count=count+1;       
        end
        Grid_All_Centers_movie{k}=All_Centers;
    end
    
    %% Plotting initial condition
    
    clf(fig,'reset'); % clearing the figure
                 
    % Plotting place cells
    if plot_in_base==0 % if we plot in the 0 base
        plot((1:N).*spatial_factor,S_PC(:,1),'color',[0,0,0],'LineWidth',.35); % plotting place cells rates (r) in the 1st base
    end
    if plot_in_base>0 % if we plot in a different base
        plot((1:N).*spatial_factor,plot_in_base_matrix'*S_PC(:,1),'color',[0,0,0],'LineWidth',.35); % plotting place cells rates (r) in another base [1,.5,.1]
    end
    
    hold all;
%     plot((1:N).*0.1,permute_matrix'*s_PC_Total,'color',[0.8,0,0.8],'LineWidth',.1); % plotting place cells rates (r) in the 2nd base [1,.5,.1]
    box on;
    xlim([1,N.*spatial_factor]); % setting max limit x axis
%     ylim([0,20]); % setting max limit y axis
    % Plotting Grid cells
    S_GCS=[S_GC(:,1) S_GC_2(:,1) S_GC_3(:,1)]; % initial the activities S of all grid cells
    Rs=[S_GC(:,1) S_GC_2(:,1) S_GC_3(:,1)]; % initial rates r of all grid cells
            
    Modules_colors=[0,.5,0;.8,0,0;0,0,1]; % setting the modules colors
    for movie_mo=1:mo % running on all modules
        l=Ratios(movie_mo); % the ratio between arena size to grid spacing (= # of bumps)
                
        Activies=zeros(1,l*n); % will contain the single period activity of grid cell
        R_Activity=zeros(1,l*n); % will contain the single period rates of grid cell
                     
        for i=1:l
            S=S_GCS(:,movie_mo); % final activity of the k'th module
            R=Rs(:,movie_mo);
                    
            Activies(1+(i-1)*n:i*n)=S;
            R_Activity(1+(i-1)*n:i*n)=R;
        end
        if plot_in_base>0 % if we plot in differnt base then the 0 base
            R_Activity=circshift(R_Activity,[0,-Grids_permutations(movie_mo,plot_in_base)]);
        end
        hold all;
        plot(Grid_All_Centers_movie{movie_mo}.*spatial_factor,R_Activity,'color',Modules_colors(movie_mo,:),'LineWidth',1.85);
    end
    
    legend('Place cells','Grid M1','Grid M2','Grid M3','Location','eastoutside');
    ylabel('Firing rate [Hz]');
    xlabel(sprintf('Map #%d spatial location [cm]',plot_in_base+1));
    filename=('Time = 0 [ms]'); % initial timestep
    title(filename);
    set(gca,'fontsize',16,'xtick',0:24:192);     
%     ylim([0,30]); % setting max limit y axis

    Mo1=getframe(gcf); % movie of the first base
    writeVideo(vidfile,Mo1);
    
end
%% Simulating the dynamics
if Online_bump_estimate==1 % meaning we do estimate continuously the place and grid cells bump
    PC_estimate=zeros(1,Time); % will contain the estimation of location of place bump vs time
    GC_estimate=zeros(mo,Time); % will contain the estimation of location of grid cells vs time
    GC_estimate(:,1)=phases(1); % the first location of grid cells bump is the initial condition
end
if Online_GC_rate_register_initial_cond==1
    GC_rates_initial_cond=zeros(mo,Time); % will contain the gc rates at the initial condition vs time
    max_gc_rate=zeros(mo,Time); % will contain the maximal firing rate from each module vs time
end
if Online_Rates_register==1
    reso_fac=5; % factor determining once in how many iterations we register the rates
      
    Place_cells_Rates_vs_time=zeros(N,Time/reso_fac); % will contain the place cells rates vs time
    Grid_cells_Rates_vs_time=zeros(n,Time/reso_fac,mo); % will contain the grid rates vs time
    register_counter=1;
end
if Online_maps_scoring==1
    interval=50; % once every how many dt we compute the scores
    Scores=zeros(Maps+1,round(Time/interval)); % will contain the embedded map scoring vs time
    
    load('new_permutations'); % loading new permutations
    New_Scores=zeros(size(new_permutations,1),round(Time/interval)); % will contain the map scoring for new permutations (unembedded mpas) vs time
    Online_maps_scoring_Counter=1;
end
%% Observations (path integration, decoupling, paper perturbations and setting noise)
Induce_paper_Pertubation=0; % do we want to induce a similiar Pertubation to the grid cells as in the paper? 1,11 for depolarization and 2,22 for hyperpolarization (multiplicative and additive respectively)
if Induce_paper_Pertubation==1 || Induce_paper_Pertubation==2 % multiplicative change
    depor_factor=1.01; % factor affecting the amount of depolarization
    hyper_factor=5; % the factor by which we divide grid cell synaptic input if we do hyperpolarization
end
depor_add=0; % 0 addition in case we don't simulate depolarization
hyper_add=0; % 0 addition in case we don't simulate hyperpolarization

velocity=0;%60; % the velocity to be used when implemented path integration with velocity input
eps_velocity(1)=1.7; eps_velocity(2)=1.9; eps_velocity(3)=2.3; % the modulations per module


Eta_Noise_GC=-1; % want eta to grid cells noise? set to 1
eta=0;eta_2=eta;eta_3=0;
noise_factor_GC=1;

Eta_Noise_PC=-1; % want eta to place cells noise? set to 1
ETA=0;
noise_factor_PC=1;

Noise_start_time=0; % the time the noise will kick in [sec]

DeCouple=0; % want to bidirectionally decouple grid cells from place cells? set to 1
I_initi=0; % if we want to add to place cells current identiccal to their initial condition set to 1

Ipc_initial=0; % DON'T CHANGE! the additional extrnal input to place cells is 0 by defult
if I_initi==1 % if we use the initial condition external current
    Ipc_initial=S_PC(:,1); % additional external current identical to the initial condition
end

 % Spatially modulated cell input initialization (Genela)
 Current_pos = 100;


for k=2:Time % iterating time % Dori
    
    if DeCouple==1 && k==2
        gamma_pc_to_gc = 0; % how much grid cells weigh the place cells input
        gamma_gc_to_pc = 0; % how much place cells weigh the grid cells inputs
        % In the case of studying only the place cell model:
        % fixing external current to place cells when there is no noisy input from grid cells (but there is still noise from place cells) to the same external current as there is in a single map
%         I_PC=I_0_PC - (Maps+1-1)*( Total_J_top_left_fixed * mean_r_PC);
        % fixing external current to grid cells when there is no noisy input from place cells to the same external current as there is in a single map
%         I_GC=-5; I_GC_2=I_GC; I_GC_3=I_GC; 
    end
    
    if k==50 && Induce_paper_Pertubation==11 % activate additive depolarization k=500 -> 100ms
        depor_add=500; % addition to excite grid cells
    end
    if k==50 && Induce_paper_Pertubation==22 % activate additive hyperpolarization k=500 -> 100ms
        hyper_add=-100; % addition to substract to reduce exitability
    end
    
    
    sm_step_size = -1;
    Current_pos = mod(Current_pos + sm_step_size-1 ,N)+1;
    s_SM = S_SM(Current_pos,:);
    
    s_GC=S_GC(:,1); % synaptic activity of all grid cells at the previous time step - 1st module
    s_GC_2=S_GC_2(:,1); % synaptic activity of all grid cells at the previous time step - 2nd module
    s_GC_3=S_GC_3(:,1); % synaptic activity of all grid cells at the previous time step - 3rd module
    
    s_PC=S_PC(:,1); % synaptic activity of all place cells at the previous time step 
    
    s_W = (W*s_GC); % weighted GRID CELLS input, solely as a result of all other GRID CELLS activity - 1st module
    s_W_2 = (W*s_GC_2); % weighted GRID CELLS input, solely as a result of all other GRID CELLS activity - 2nd module
    s_W_3 = (W*s_GC_3); % weighted GRID CELLS input, solely as a result of all other GRID CELLS activity - 3rd module
    s_J = (J*s_PC); % weighted PLACE CELLS input, solely as a result of all other PLACE CELLS activity
    
    
    s_M_GC=gamma_pc_to_gc.*(M*s_PC); % weighted GRID CELLS input, solely as a result of all other PLACE CELLS activity - 1st module
    s_M_PC=gamma_gc_to_pc.*(M'*s_GC); % weighted PLACE CELLS input, solely as a result of all other GRID CELLS activity - 1st module
    
    s_M_GC_2=gamma_pc_to_gc.*(M_2*s_PC); % weighted GRID CELLS input, solely as a result of all other PLACE CELLS activity - 2nd module
    s_M_PC_2=gamma_gc_to_pc.*(M_2'*s_GC_2); % weighted PLACE CELLS input, solely as a result of all other GRID CELLS activity - 2nd module
    
    s_M_GC_3=gamma_pc_to_gc.*(M_3*s_PC); % weighted GRID CELLS input, solely as a result of all other PLACE CELLS activity - 3rd module
    s_M_PC_3=gamma_gc_to_pc.*(M_3'*s_GC_3); % weighted PLACE CELLS input, solely as a result of all other GRID CELLS activity - 3rd module
    
    %%%%%%% Genela %%%%%%%%%%%
        sm_factor = 2; % This modulates how much the bump follows the spatially modulated cells
                         % and also tends to attenuate the grid activity if
                         % too large, and thus has to be carefully tuned.
                         % Currently best value is ~3.
                         %
        s_M_PC_SM = sm_factor*gamma_gc_to_pc.*(M_PC_SM'*s_SM'); 
     %
    
    s_GC_Total=s_W+s_M_GC+I_GC + depor_add+hyper_add; % plus constant external input -> Total grid cells inputs - 1st module
    s_GC_Total(Even_indices) = s_GC_Total(Even_indices) - eps_velocity(1)*velocity; % adding velocity input to even grid cells
    s_GC_Total(Odd_indices) = s_GC_Total(Odd_indices) + eps_velocity(1)*velocity; % adding velocity input to odd grid cells
    
    s_GC_Total_2=s_W_2+s_M_GC_2+I_GC_2 + depor_add+hyper_add; % plus constant external input -> Total grid cells inputs - 2nd module
    s_GC_Total_2(Even_indices) = s_GC_Total_2(Even_indices) - eps_velocity(2)*velocity; % adding velocity input to even grid cells
    s_GC_Total_2(Odd_indices) = s_GC_Total_2(Odd_indices) + eps_velocity(2)*velocity; % adding velocity input to odd grid cells
     
    s_GC_Total_3=s_W_3+s_M_GC_3+I_GC_3 + depor_add+hyper_add; % plus constant external input -> Total grid cells inputs - 3rd module
    s_GC_Total_3(Even_indices) = s_GC_Total_3(Even_indices) - eps_velocity(3)*velocity; % adding velocity input to even grid cells
    s_GC_Total_3(Odd_indices) = s_GC_Total_3(Odd_indices) + eps_velocity(3)*velocity; % adding velocity input to odd grid cells
    
    %Genela
    genelpha = 0.9;
    s_PC_Total =s_J+(1-genelpha)*(s_M_PC+s_M_PC_2+s_M_PC_3)+genelpha*s_M_PC_SM+I_PC + Ipc_initial; % plus constant external input -> Total place cells inputs
    %s_PC_Total =s_J+s_M_PC+s_M_PC_2+s_M_PC_3+I_PC + Ipc_initial;
    
%     figure(2)
%     plot(s_M_PC_SM);
%     drawnow;
    figure(1)
    
    %%% Note: Subtract the mean of s_U from I_PC to maintain balance 
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    s_GC_Total(s_GC_Total<0)=0; % applying non linearity for grid cells (thershold 0) - 1st module
    s_GC_Total_2(s_GC_Total_2<0)=0; % applying non linearity for grid cells (thershold 0) - 2nd module
    s_GC_Total_3(s_GC_Total_3<0)=0; % applying non linearity for grid cells (thershold 0) - 3rd module
    
    s_PC_Total(s_PC_Total<0)=0; % applying non linearity for place cells(thershold 0)
    
    s_GC_Total(s_GC_Total>0)=sqrt(s_GC_Total(s_GC_Total>0)); % sub linear (sqrt) for the grid cells - 1st module
    s_GC_Total_2(s_GC_Total_2>0)=sqrt(s_GC_Total_2(s_GC_Total_2>0)); % sub linear (sqrt) for the grid cells - 2nd module
    s_GC_Total_3(s_GC_Total_3>0)=sqrt(s_GC_Total_3(s_GC_Total_3>0)); % sub linear (sqrt) for the grid cells - 3rd module
    
    s_PC_Total(s_PC_Total>0)=sqrt(s_PC_Total(s_PC_Total>0)); % sub linear (sqrt) for the place cells
    
    
    if Eta_Noise_GC==1 && k>Noise_start_time/deltat
        eta = (s_GC_Total.*deltat).^.5 .* randn(n,1); % gaussian variable with (r_i)*dt variance and mean 0 for each grid cell
        eta_2 = (s_GC_Total_2.*deltat).^.5 .* randn(n,1); % gaussian variable with (r_i)*dt variance and mean 0 for each grid cell
        eta_3 = (s_GC_Total_3.*deltat).^.5 .* randn(n,1); % gaussian variable with (r_i)*dt variance and mean 0 for each grid cell
    end  
    if Eta_Noise_PC==1 && k>Noise_start_time/deltat
        ETA = (s_PC_Total.*deltat).^.5 .* randn(N,1); % gaussian variable with (r_i)*dt variance and mean 0 for each place cell
    end 
    
    %%%% Dynamics of grid cells and place cells (Genela)%%%%
    
    S_GC(:,2)=s_GC-(s_GC./Tau_GC)*deltat + (s_GC_Total./Tau_GC)*deltat + (eta/Tau_GC)./noise_factor_GC; % updating the synaptic activity of all GRID cells at the current time step - 1st module
    S_GC_2(:,2)=s_GC_2-(s_GC_2./Tau_GC)*deltat + (s_GC_Total_2./Tau_GC)*deltat + (eta_2/Tau_GC)./noise_factor_GC; % updating the synaptic activity of all GRID cells at the current time step - 2nd module
    S_GC_3(:,2)=s_GC_3-(s_GC_3./Tau_GC)*deltat + (s_GC_Total_3./Tau_GC)*deltat + (eta_3/Tau_GC)./noise_factor_GC; % updating the synaptic activity of all GRID cells at the current time step - 3rd module
    
    S_PC(:,2)=s_PC-(s_PC./Tau_PC)*deltat + (s_PC_Total./Tau_PC)*deltat + (ETA/Tau_PC)./noise_factor_PC; % updating the synaptic activity of all PLACE cells at the current time step
    %step=1;
    %S_PC(:,2)=circshift(s_PC,step); % Genela
    %%%%%%%
    
    %% Registering and updating activities
    if Online_Rates_register==1 && mod(k,reso_fac)==0
        Place_cells_Rates_vs_time(:,register_counter)=s_PC_Total; % place cell rates at current time
        Grid_cells_Rates_vs_time(:,register_counter,1)=s_GC_Total; % grid cells 1st module rates at current time
        Grid_cells_Rates_vs_time(:,register_counter,2)=s_GC_Total_2; % grid cells 2nd module rates at current time
        Grid_cells_Rates_vs_time(:,register_counter,3)=s_GC_Total_3; % grid cells 3rd module rates at current time
        
        register_counter=register_counter+1;
    end
             
    S_GC(:,1)=S_GC(:,2); % we have calculated the current state of the grid cells - now it's their previous step - 1st module
    S_GC_2(:,1)=S_GC_2(:,2); % we have calculated the current state of the grid cells - now it's their previous step - 2nd module
    S_GC_3(:,1)=S_GC_3(:,2); % we have calculated the current state of the grid cells - now it's their previous step - 3rd module
      
    S_PC(:,1)=S_PC(:,2); % we have calculated the current state of the place cells - now it's their previous step
    
    if Online_bump_estimate==1 && Maps>0 % if we estimate bump location among multiple environments
        Rs=[s_GC_Total s_GC_Total_2 s_GC_Total_3]; % the rates r of all grid cells
        [pc_bump_loc, gc_bump_loc ] = Online_bump_estimation_Analytic( s_PC_Total,steady_1d_bump_analytic_with_Grids_2pop,Maps, Rs,Lambdas,Grid_Locations,GC_RFS,PERMUTATIONS,Grids_permutations,GC_estimate(:,k-1));
        PC_estimate(k)=pc_bump_loc;
        GC_estimate(:,k)=gc_bump_loc;
    end
    if Online_bump_estimate==1 && Maps==0 % if we estimate bump location in a single environment
        [ pc_bump_loc ] = PC_bump_estimation_0_Maps_Analytic( s_PC_Total,steady_1d_bump_2pop );
        PC_estimate(k-1)=pc_bump_loc;
    end
    
    if Online_GC_rate_register_initial_cond==1 % if we want to register the rate vs time of the intial condition
        if k<=2 % we compute once the location we need the grid cells activities for
            modules_ic_location=phases(1).*ones(mo,1); % will contain the first peiodicity grid cell location for each modules
            gc_numbers=cell(1,mo); % will contain the grid cell number (from 1 to 960) of the cell encoding the relevant location

            for m=1:mo
                while modules_ic_location(m)>Lambdas(m)
                    modules_ic_location(m)=modules_ic_location(m)-Lambdas(m); % getting the location in the first period
                end
                [val,loc]=min(abs(modules_ic_location(m)-Grid_Locations(m,:)));
                if val==0 % if a grid cell that encodes this exact location exsist
                    gc_numbers{m}=loc; % the exact grid cell # (from 1 to 960) that encodes this location (if exsits)
                end
                if val~=0 % if a grid cell that encodes this exact location doesn't exsist
                    gc_numbers{m}=find(abs(modules_ic_location(m)-Grid_Locations(m,:))==val); % # of grid cells who activities should be interpolated
                end  
            end
        end      
        % Registering the rates in the initial condition
        Rs=[s_GC_Total s_GC_Total_2 s_GC_Total_3]'; % the rates r of all grid cells
        
        if base_GC>1 % if we are not in the 0 map we need to shift back the grid cells
            Rs(1,:)=circshift(Rs(1,:),[0,-Grids_permutations(1,base_GC)]);
            Rs(2,:)=circshift(Rs(2,:),[0,-Grids_permutations(2,base_GC)]);
            Rs(3,:)=circshift(Rs(3,:),[0,-Grids_permutations(3,base_GC)]);
        end
    
        for m=1:mo % running on all modules
            Loc=gc_numbers{m}; % grid cell/s numbers (1 to 960) that encode the relevant location
            GC_rates_initial_cond(m,k) = mean(Rs(m,Loc)); % the rates of grid cells in the desired location
            
            max_gc_rate(m,k) = max(Rs(m,:)); % maximal firing rate from each module vs time
        end
    end
    
    if Online_maps_scoring==1 && mod(k,interval)==0 || Online_maps_scoring==1 && k==2 % scoring the place cell rates across all maps
        
        [scores] = mixed_states_check_while_path_integrated(PERMUTATIONS,s_PC_Total,steady_1d_bump_analytic_with_Grids_2pop,Maps); % evaluating scores for each embedded map
        Scores(:,Online_maps_scoring_Counter)=100.*scores; % scores for embedded maps, multiply by 100 for convienence for presenting the data
        
        [new_scores] = mixed_states_check_while_path_integrated_new_permutations(new_permutations,s_PC_Total,steady_1d_bump_analytic_with_Grids_2pop); % evaluating scores for each unembedded map
        New_Scores(:,Online_maps_scoring_Counter)=100.*new_scores; % scores for unembedded maps, multiply by 100 for convienence for presenting the data
        
        Online_maps_scoring_Counter=Online_maps_scoring_Counter+1;
    end
    %% Movie in a single base of choice
    if generate_movie==1
        if mod(k,50)==0
            clf(fig,'reset'); % clearing the figure
                 
            % Plotting place cells
            if plot_in_base==0 % if we plot in the 0 base
                plot((1:N).*spatial_factor,s_PC_Total,'color',[0,0,0],'LineWidth',.35); % plotting place cells rates (r) in the 1st base
            end
            if plot_in_base>0 % if we plot in a different base
                 plot((1:N).*spatial_factor,plot_in_base_matrix'*s_PC_Total,'color',[0,0,0],'LineWidth',.35); % plotting place cells rates (r) in another base [1,.5,.1]
            end
            
            hold all;
%             plot((1:N).*0.1,permute_matrix'*s_PC_Total,'color',[0.8,0,0.8],'LineWidth',.1); % plotting place cells rates (r) in the 2nd base [1,.5,.1]

            box on;
            xlim([1,N.*spatial_factor]); % setting max limit x axis
%             ylim([0,20]); % setting max limit y axis
            % Plotting Grid cells
            S_GCS=[S_GC(:,2) S_GC_2(:,2) S_GC_3(:,2)]; % the activities S of all grid cells
            Rs=[s_GC_Total s_GC_Total_2 s_GC_Total_3]; % the rates r of all grid cells
            
            Modules_colors=[0,.5,0;.8,0,0;0,0,1]; % setting the modules colors
            for movie_mo=1:mo % running on all modules
                l=Ratios(movie_mo); % the ratio between arena size to grid spacing (= # of bumps)
                
                Activies=zeros(1,l*n); % will contain the single period activity of grid cell
                R_Activity=zeros(1,l*n); % will contain the single period rates of grid cell
                     
                for i=1:l
                    S=S_GCS(:,movie_mo); % final activity of the k'th module
                    R=Rs(:,movie_mo);
                    
                    Activies(1+(i-1)*n:i*n)=S;
                    R_Activity(1+(i-1)*n:i*n)=R;
                end
                if plot_in_base>0 % if we plot in differnt base then the 0 base
                    R_Activity=circshift(R_Activity,[0,-Grids_permutations(movie_mo,plot_in_base)]);
                end
                hold all;
                plot(Grid_All_Centers_movie{movie_mo}.*spatial_factor,R_Activity,'color',Modules_colors(movie_mo,:),'LineWidth',1.85);
            end
           
            legend('Place cells','Grid M1','Grid M2','Grid M3','Location','eastoutside');
            ylabel('Firing rate [Hz]');
            xlabel(sprintf('Map #%d spatial location [cm]',plot_in_base+1));
            filename=sprintf('Time = %d [ms]',round(k*deltat*10^3)); % *10^3 to convert from sec to ms
            title(filename);
            set(gca,'fontsize',16,'xtick',0:24:192);     
%             ylim([0,30]); % setting max limit y axis

            Mo1=getframe(gcf); % movie of the first base
            writeVideo(vidfile,Mo1);
        end
    end
    %% Movie in all bases simultaneously
    if generate_movie==11
        
        if mod(k,50)==0
            clf(fig,'reset'); % clearing the figure
            for plot_maps=0:Maps
                
                subplot(2,3,plot_maps+1);
                
                if plot_maps==0 % plotting place cells the 0 base
                    plot((1:N).*spatial_factor,s_PC_Total,'color',[0,0,0],'LineWidth',.25); % plotting place cells rates (r) in the 1st base
                end
                if plot_maps>0 % plotting place cells in a different base
                    plot_in_base_matrix = Permute_Matrices(:,:,plot_maps);
                    plot((1:N).*spatial_factor,plot_in_base_matrix'*s_PC_Total,'color',[0,0,0],'LineWidth',.25); % plotting place cells rates (r) in another base
                end
                
                hold all;
                box on;
                xlim([1,N.*spatial_factor]); % setting max limit x axis
%                 ylim([0,20]); % setting max limit y axis
            
                % Plotting Grid cells
                S_GCS=[S_GC(:,2) S_GC_2(:,2) S_GC_3(:,2)]; % the activities S of all grid cells
                Rs=[s_GC_Total s_GC_Total_2 s_GC_Total_3]; % the rates r of all grid cells
            
                Modules_colors=[0,.5,0;.8,0,0;0,0,1]; % setting the modules colors
                for movie_mo=1:mo % running on all modules
                    l=Ratios(movie_mo); % the ratio between arena size to grid spacing (= # of bumps)
                
                    Activies=zeros(1,l*n); % will contain the single period activity of grid cell
                    R_Activity=zeros(1,l*n); % will contain the single period rates of grid cell
                     
                    for i=1:l
                        S=S_GCS(:,movie_mo); % final activity of the k'th module
                        R=Rs(:,movie_mo);
                    
                        Activies(1+(i-1)*n:i*n)=S;
                        R_Activity(1+(i-1)*n:i*n)=R;
                    end
                    if plot_maps>0 % if we plot in differnt base then the 0 base then we need to permute
                        R_Activity=circshift(R_Activity,[0,-Grids_permutations(movie_mo,plot_maps)]);
                    end
                    hold all;
                    plot(Grid_All_Centers_movie{movie_mo}.*spatial_factor,R_Activity,'color',Modules_colors(movie_mo,:),'LineWidth',.85);
                end
                
%                 legend('Place cells','Grid M1','Grid M2','Grid M3');
                ylabel('Firing rate [Hz]');
                xlabel(sprintf('Map #%d spatial location [cm]',plot_maps));
                legend('Place cells','Grid M1','Grid M2','Grid M3');
                filename=sprintf('Time = %d [ms]',k*deltat*10^3); % *10^3 to convert from sec to ms
                title(filename);
                set(gca,'fontsize',8,'xtick',0:24:192);               
%                 ylim([0,30]); % setting max limit y axis
            end
            
            Mo1=getframe(gcf); % movie of the first base
            writeVideo(vidfile,Mo1);
        end
    end
end
   
if generate_movie==1 ||generate_movie==11
    close(vidfile); % done writing to to video file
end
%% Covering the whole arena with grid cells as the place cells;
S_GCS=[S_GC(:,1) S_GC_2(:,1) S_GC_3(:,1)]; % the final synaptic activities (S) of all grid cells
Rs=[s_GC_Total s_GC_Total_2 s_GC_Total_3]; % the rates (r) of all grid cells

Grid_All_Centers=cell(1,mo); % will contain all the location grid cell encodes
Grid_Activies=cell(1,mo); % will contain all the synaptic activities (S) for the corresponding locations
Grid_Rates=cell(1,mo); % will contain all the rates (r) for the corresponding locations

for k=1:mo
    l=Ratios(k); % the ratio between arena size to grid spacing (= # of bumps)
    lambda=Lambdas(k); % the relevant grid spacing
    All_Centers=zeros(1,l*n); % will contain the ceneters of grid cells
    
    Activies=zeros(1,l*n); % will contain the single period activity of grid cell
    R_Activity=zeros(1,l*n); % will contain the single period rates of grid cell
    
    S=S_GCS(:,k); % final synaptic activity of the k'th module
    R=Rs(:,k); % final rates of grid cells 
    
    count=0;
    for i=1:l
        All_Centers(1+(i-1)*n:i*n)=Grid_Locations(k,:)+count*lambda;
        
        Activies(1+(i-1)*n:i*n)=S;
        R_Activity(1+(i-1)*n:i*n)=R;
        
        count=count+1;       
    end
    Grid_All_Centers{k}=All_Centers;
    Grid_Activies{k}=Activies; % the synaptic activit (S) of grid cells
    Grid_Rates{k}=R_Activity; % the rates (r) of grid cells
end
%% final place cells rates on all maps

ACTIVITY_s=zeros(Maps+1,N); % will hold the place cells synaptic activity in all the map patterns
ACTIVITY_s(1,:)=S_PC(:,2); % the first row is the plcae cells synaptic activity in the original base

ACTIVITY_r=zeros(Maps+1,N); % will hold the place cells rates (r) in all the map patterns
ACTIVITY_r(1,:)=s_PC_Total; % the first row is the plcae cells rates (r) in the original base

for i=1:Maps
%     Permutation=PERMUTATIONS(i,:); % the relevant permutation for this map
%     
    sp_end_s=S_PC(:,2); % duplicate the final synaptic activity from the original to perform permutation from it
    sp_end_r=s_PC_Total; % duplicate the final rates (r) from the original to perform permutation from it

%     permute_matrix=zeros(N);
% 
%     for m=1:N
%         permute_matrix(m,Permutation(m))=1;
%     end

    permute_matrix = Permute_Matrices(:,:,i);

    sp_end_s=permute_matrix'*sp_end_s;
    sp_end_r=permute_matrix'*sp_end_r;

    ACTIVITY_s(i+1,:)=sp_end_s; % the final synaptic activity in the i'th map
    ACTIVITY_r(i+1,:)=sp_end_r; % the final rates (r) in the i'th map
end

%% End of function - saving
Data=struct;

Data.Place_s=ACTIVITY_s; % final synaptic activity of place cells over all maps
Data.Place_r=ACTIVITY_r; % final rates (r) of place cells over all maps [Hz]
Data.Grid_All_Centers=Grid_All_Centers; % all centers of location of grid cells from all modules
Data.Grid_Activies=Grid_Activies; % final synaptic activities of corresponding grid cells
Data.Grid_Rates=Grid_Rates; % final rates (r) of corresponding grid cells [Hz]
Data.GC_RFS=GC_RFS; % interpolated activites of grid cells around cell #1 (for initial condition)
Data.phases=phases; % the phases used as bump inital condition
Data.shifts_loc=shifts_loc; % the location [cm] in the arena where all modules combine for each map
Data.PERMUTATIONS=PERMUTATIONS; % the place cell permuatations
Data.Grids_permutations=Grids_permutations; % the coordinated neurons permutations (shifts) for all modules for all maps



if Online_bump_estimate==1 % saving the grid and place cells bumps locations vs time (for pc path integration & grid cell modules coupling)
    Data.PC_estimate=PC_estimate;
    Data.GC_estimate=GC_estimate;
end
if Online_GC_rate_register_initial_cond==1 % saving the rates of grid cells in the initial condition (for blobs variability)
    Data.GC_rates=GC_rates_initial_cond;
    Data.max_gc_rate=max_gc_rate; % maximal firing rate from each module vs time
end
if Online_Rates_register==1  % saving the place cells and grid cells rates vs time
    Rates_vs_Time=cell(1,2);
    Rates_vs_Time{1}=Place_cells_Rates_vs_time;
    Rates_vs_Time{2}=Grid_cells_Rates_vs_time;
    Data.Rates_vs_Time=Rates_vs_Time;
end
if Online_maps_scoring==1 % saving scores vs time for each of the maps
    Data.Scores=Scores; % scores for embedded maps
    Data.New_Scores=New_Scores; % scores for new permutations 
    Data.interval=interval;  % once every how many dt we compute the scores
    Data.deltat=deltat; % time interval
end

filename=sprintf('Data=%d.mat',trials);
save(filename,'Data');
end