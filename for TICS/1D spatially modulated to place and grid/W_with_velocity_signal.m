close all;
clear;

PC=4800; % total # of Place cells resulotion of 0.04cm (arena = 1.92 meters)

Ratios=[3,4,5]; % ratios of arena size with grid cells modules - in purpose mod(PC,Ratios)=0.
mo=numel(Ratios); % total # of modules

Lambdas=PC./Ratios; % the corresponding grid spacings
n=min(Lambdas); % the # of grid cells in each module - each module has the same # of grid cells

Grid_Locations=zeros(mo,n); % will contain the location which the frid cells encodes
D_most_left=zeros(mo,n); % will contain the distances for the most left neuron on each module (circular boundary condition)
%% Spreading the grid cell and Calculating the distances for most left neuron in each module
for k=1:mo % running on all modules
    lambda=Lambdas(k); % grid spacig of the k'th module, the total distance this module encodes
    
    Locations=round(linspace(1,lambda,n)-1); % locations of the the grid cells
    Grid_Locations(k,:)=Locations; % locations that the grid cells encode
end
%% Generating distances matrices
DD=zeros(n,n,mo); % will contain distnaces matrices for all modules
del=0; % = differnece we induce between populations % Dori - was 300, now set to 0

for m=1:mo % running on all modules
    
    lambda=Lambdas(m); % the relevant lambda
    Locations=Grid_Locations(m,:); % the relevant locations of grid cells
    delta=del/Ratios(m); % amount of diff we insert to odd and even populations
    
    D=zeros(n); % will contain the distnaces between all neurons
    
    for i=1:n % the neruon we calculate the incoming synapses to
        loc=Locations(i); % location of relevant neuron
        if mod(i,2)==0 % meaning its a neuron from even population
            for j=1:n % all other neurons which we consider
                dist=Locations(j)-loc;
                
                dist=dist+delta;
                
                if dist>lambda/2
                    dist=dist-Locations(end)-1;
                end
                if dist<-lambda/2
                    dist=dist+Locations(end)+1;
                end
           
                D(i,j)=dist;
            end
        end

        if mod(i,2)~=0 % meaning its a neuron from odd population
            for j=1:n
                dist=Locations(j)-loc;
                
                dist=dist-delta;
                
                if dist>lambda/2
                    dist=dist-Locations(end)-1;
                end
                if dist<-lambda/2
                    dist=dist+Locations(end)+1;
                end
   
                D(i,j)=dist;
            end
        end
    end
    
    DD(:,:,m)=D;
end

%% Calculating the connectivity fucntions (1 Gaussian with global inhibition)
Sigmas=Lambdas./3;
A=Lambdas./20;

W=zeros(n,n,mo); % will contain gauusian for each distance matrix
for m=1:mo   
    W(:,:,m)=A(m) .* (1/(sqrt(2*pi) *Sigmas(m))) .* exp( (-DD(:,:,m).^2)./(2*Sigmas(m)^2)); % Gaussian
end

Total_W_top_left=-n/80; % sum of each row constraint (=alpha)
exci=zeros(1,mo);
for k=1:mo
    w=W(:,:,mo);
    exci(k)=mean(sum(w,2)) - mean(diag(w));
end
Excite = mean(exci); % total excitation (after substracting the connection from the neuron to itself)

inhibition=Total_W_top_left-Excite;% leftover for inhibition
h=inhibition/(n-1); % individual neuron inhibition parameter distributed equally among all other neurons

W=W+h;

for m=1:mo
    w=W(:,:,m);
    for k=1:n
        w(k,k)=0;
    end
    W(:,:,m)=w;
end

W=mean(W,3); % averaged gaussian for all module ditances

W=W./max(max(W));
W=W';
%% Parameters & initial condition
Time=1.7*10^4+1; % Total time of the realization [iterations]
deltat=1; % [ms]
Tau=50; % synaptic time constant [ms]
I=3;%6+5001/40000; % external input to all neurons

S=zeros(n,2); % synaptic activity of the grid cells only for current and previous states (markov dynamics)

% rng('shuffle');
% S(:,1)=rand(N,1);
S(end-4:end,1)=[.6:.1:1]';
S(1:5,1)=[1:-.1:.6]';

% w=find(W(lambda,:)>0); % initial condition around specific grid cell
% S(w,1)=1;
%% Simulating the dynamics
T=7000; % # of time steps we keep track of and save
s_final_count=1;
S_final=zeros(n,T); % the final saved data

for k=2:Time
    
    s=S(:,1); % synaptic activity of all neurons at the previous time step
    
    s_W=W*s; % inner product of weighted input to all neurons, vector for all neurons
    s_W=s_W+I; % plus constant external currnet, vector for all neurons 
    
%     s_W=roundn(s_W,-6); % rounding to compensate for differnet sum(J) and wrong values of the sqrt
    
    s_W(s_W<0)=0; % applying non linearity (thershold 0)
    s_W(s_W>=0)=sqrt(s_W(s_W>=0)); % sub linear (sqrt)

    S(:,2)=s-(s./Tau)*deltat + (s_W./Tau)*deltat; % updating the synaptic activity of all the neurons at the current time step
    
    if k>=Time-T
        S_final(:,s_final_count)=S(:,1);
        s_final_count=s_final_count+1;
    end
    
    %% Inducing pertubation
%     if k==1000
%         S(:,2)=S(:,2)+(rand(N,1)-0.5).*10^-7;
%     end
    %%
    S(:,1)=S(:,2); % we have calculated the current state - now it's the previous step

end
%% Covering the whole arena as the place cells;
Grid_All_Centers=cell(1,mo); % will contain all the location grid cell encodes
figure(1);
hold all;
for k=1:mo
    l=Ratios(k); % the ratio between arena size to grid spacing (= # of bumps)
    lambda=Lambdas(k); % the relevant grid spacing
    All_Centers=zeros(1,l*n); % will contain the ceneters of grid cells
    Activies=zeros(1,l*n); % will contain the single period activity of grid cell
    
    count=0;
    for i=1:l
        All_Centers(1+(i-1)*n:i*n)=Grid_Locations(k,:)+count*lambda;
        Activies(1+(i-1)*n:i*n)=S(:,2);
        count=count+1;       
    end
    Grid_All_Centers{k}=All_Centers;   
    plot(All_Centers,Activies);
end

figure(2);
hold all;
for k=1:n
    plot(S_final(k,:));
end