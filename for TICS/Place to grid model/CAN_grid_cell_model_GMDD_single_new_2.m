function our_model
%
% Code for implementing SCAN model - demonstrating how movement of place
% cells can control movement of grid cells forming an attractor manifold.
%
% Code generated for Morris & Deridkman, TICS, 2023
% The code is modified from code for Agmon and Burak, eLife, 2020
%

%% parameters of model
parms.nSpatialBins = 50; 
parms.ncells = parms.nSpatialBins*parms.nSpatialBins; % total number of cells in network
parms.aa = 14; 
aa = parms.aa;
parms.R= 12;  %size of voronoi zone
parms.Rgrid = 6; % This version of the code will use a multiplier on R for grid cells.
parms.W0_L5= -0.08; %amplitude of firing field  Dori -0.16
parms.W0_PC_L5 = 0.1; 
parms.mod_ratio = 2; 
parms.A0_PC = 20; %  Max. amplitude of place cell 
parms.g=0;   % was 1/aa; %threshold-linear function
parms.Weights_L5='W_L5.mat';
parms.Weights_PC_L5='W_PC_L5.mat';
parms.W_PC_L5_new='W_PC_L5_new.mat';
parms.Activity_PC = 'A_PC.mat';
parms.I=20; 
parms.std_noise=aa*0.4;
parms.skip = 100; % how much to skip for the mean vel. calculation
parms.simdur = 120000; % duration of simulation

%% Simulation parameters
dt =1; % time step, ms 
stabilizationTime = 400; % no-velocity time for pattern to form, ms
tind = 0; % counts iterations of the simulation
t = 0; % measure real time of the simulation
tau = 10; % grid cell synapse time constant, ms
livePlot=50; %mod of what the simulation will show
watchCell=1275;
% threshold for plotting a cell as having spiked
spikeThresh = parms.std_noise;
count=1;

parms.Cell_file_to_load='2-Cell_r11025_d010605_s11_t7_c2.mat';


%% Create vector with the preferred directions of all neurons (radians)
L5_Dir = [0 pi/2; pi 3*pi/2]; % block of 4 different directions
L5_Dir = repmat(L5_Dir,sqrt(parms.ncells)/2,sqrt(parms.ncells)/2);
L5_Dir = reshape(L5_Dir,1,[]);

%% x(i) and y(i) represent the possition of the neuron i in a sqrt(ncells) x sqrt(ncells) neuronal sheet
x = (0:(sqrt(parms.ncells)-1))-(sqrt(parms.ncells)-1)/2; 
[X,Y] = meshgrid(x,x);
x=reshape(X,1,[]);
y=reshape(Y,1,[]);


%% if there is allready Weights load them if not create them and save
if exist(parms.Weights_L5,'file')

    load(parms.Weights_L5);

else

    W_L5=Build_Conections(x,y,parms,parms.W0_L5,L5_Dir); % is_shift = 0
    
    %save(parms.Weights_L5,'W_L5');
    
end


%% if there is allready Weights load them if not create them and save
if exist(parms.Weights_PC_L5,'file')

    load(parms.Weights_PC_L5);

else
    
    W_PC_L5=Build_Conections_between_layers(x,y,parms,parms.W0_PC_L5);
    %save(parms.Weights_PC_L5,'W_PC_L5');
    
end

%% if there is allready Weights load them if not create them and save
if exist(parms.Activity_PC,'file')

    load(parms.Activity_PC);

else
    
    A_PC=Build_PC_activity(x,y,parms,parms.A0_PC);
    %save(parms.Activity_PC,'A_PC');
    
end

%%  show Weights

% W_L5 = full(W_L5);
% W_PC_L5 = full(W_PC_L5);



pos=Load_Trajectory_From_File(parms,dt);

I=sin(8*(0:2*pi/1000:2*pi*(length(pos)/1000)));
I=(I(1:length(pos))+1)/2;



%% Firing field plot variables
nSpatialBins = parms.nSpatialBins;
min_x =floor(min(pos(1,:)));
max_x=ceil(max(pos(1,:))); % m
min_y = floor(min(pos(2,:)));
max_y=ceil(max(pos(2,:))); % m
% divid the environment into spatial bins 
axis_x = min_x:(max_x-min_x)/nSpatialBins:max_x;
axis_y = min_y:(max_y-min_y)/nSpatialBins:max_y;

time_map2 = zeros(nSpatialBins+1);
spikes_map2 = zeros(nSpatialBins+1);
spikeCoords2 = [];
time_map5 = zeros(nSpatialBins+1);
spikes_map5 = zeros(nSpatialBins+1);
spikeCoords5 = [];


if ~isfield(parms,'simdur')
    simdur =max(pos(3,:)); % total simulation time, ms
else
    simdur = parms.simdur;
end

if livePlot
   h = figure('color','w','name','Activity of sheet of cells on brain''s surface');
   drawnow
   
end


%% Initial conditions
L5 = rand(1,parms.ncells); % activation of each cell;
PC = zeros(1,parms.ncells);
%% stabilization
close all

L5=Stabilization_Of_System...
    (stabilizationTime,L5_Dir,L5,W_L5,parms,h);

% build connectivity matrix between place cells and grid cells
if exist(parms.W_PC_L5_new,'file')

    load(parms.W_PC_L5_new);

else

    W_PC_L5_new = build_place_grid_connectivity(A_PC,L5,parms);
    
    save(parms.W_PC_L5_new,'W_PC_L5_new');
    
end

W_PC_L5 = W_PC_L5_new;

tind=0;
clear rat_Dir rat_speed;
 
%% Run Simulation 
while t< simdur
    
   if livePlot>0 && (livePlot==1 || mod(tind,livePlot)==1)
       t
   end
   
     
    
  tind = tind+1; %% count iterations of the simulation
  t = dt*tind; %% measure reall time 
  
   
  %% find place cell activity packet
 
  curr_x = pos(1,tind);
  curr_y = pos(2,tind);
  try
    ind_x = find(axis_x(1:end-1) <= curr_x & axis_x(2:end) > curr_x);
  catch
      ind_x = nSpatialBins;
  end
  try
    ind_y = find(axis_y(1:end-1) <= curr_y & axis_y(2:end) > curr_y);
  catch
      ind_y = nSpatialBins;
  end
  my_ind = (ind_y-1)*nSpatialBins + ind_x;
  
  PC = A_PC(my_ind,:);  % current place cell activity packet
  
  %figure(999); imagesc(reshape(PC,[50 50])); colorbar; % debug
  
  %% Compute Layer 5 
  
  %L5Inputs = (W_L5*L5'+ W_PC_L5*PC')'+parms.I;

  internal = (W_L5*L5' + parms.I)';
  external = (W_PC_L5*PC')';
  L5Inputs = internal+external;
  %L5Inputs = external;
  
  L5Inputs_b = L5Inputs;

  % Synaptic drive only increases if input cells are over threshold 1
  L5Inputs = L5Inputs.*(L5Inputs>parms.g);

    
  L5 = L5 + dt*(L5Inputs - L5)/tau;
   
  
  
  
%% ploting the figures
   if livePlot>0 && (livePlot==1 || mod(tind,livePlot)==1)
      keep = 1;
      [v_pc,v_gc,time_map5,spikes_map5,spikeCoords5]=Plot_Simulation...
        (keep,L5,PC, watchCell,spikeThresh,pos,axis_x,axis_y,tind,livePlot,dt,parms,t,time_map5...
         ,spikes_map5,spikeCoords5,internal,external,L5Inputs);
   end
  
end
v_pc_mean = mean(v_pc(parms.skip:end));
v_gc_mean = mean(v_gc(parms.skip:end));
 
keep = 0;
[v_pc,v_gc,time_map5,spikes_map5,spikeCoords5]=Plot_Simulation...
    (keep,L5,PC, watchCell,spikeThresh,pos,axis_x,axis_y,tind,livePlot,dt,parms,t,time_map5...
    ,spikes_map5,spikeCoords5,internal,external,L5Inputs);
disp('')



function W=Build_PC_activity(x,y,parms,strength)

W=zeros(parms.ncells);
  
    for i=1:parms.ncells
%       if mod(i,round(parms.ncells/10))==0
%         fprintf('Generating weight matrix. %d%% done.\n',round(i/parms.ncells*100))
%       end
     
      [dist_x,dist_y]=Find_Periodic_Boundary(x(i),y(i),x,y,parms);
              
       first_term=dist_x;
       second_term=dist_y;
       
       excit_width = 1/8; % 1/4; Dori
       
      tmpW=strength*( exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.R* excit_width).^2) ) ;
        
                 
       W(i,:) = tmpW;
    
    end
    
    %W=sparse(W);
figure;imagesc(reshape(tmpW,sqrt(parms.ncells),sqrt(parms.ncells)));
colorbar;

disp('')

function W_PC_L5 = build_place_grid_connectivity(A_PC,GC,parms)
%
% build connectivity between place cells and grid cells by looking at all
% shifted cross-correlations of both matrices
%
GC = reshape(GC,[parms.nSpatialBins parms.nSpatialBins]);

% build connectivity for all place cells with one grid cell
for ind = 1:parms.ncells
    ind
    PC = A_PC(ind,:);
    ind_g = 0;
    for ind_i = 1:parms.nSpatialBins
        GC_circ_i = circshift(GC,ind_i-1,1);
        for ind_j = 1:parms.nSpatialBins
            ind_g = ind_g+1;
            GC_circ = circshift(GC_circ_i,ind_j-1,2);
            coefs = corrcoef(PC,GC_circ);         
            corrvals(ind_g,ind) = coefs(2,1);
        end
    end
end

% build connectivity of place cells to all grid cells, by shifting grid
% cell position



W_PC_L5 = corrvals;


function W=Build_Conections_between_layers(x,y,parms,strength)

W=zeros(parms.ncells);
  
    for i=1:parms.ncells
%       if mod(i,round(parms.ncells/10))==0
%         fprintf('Generating weight matrix. %d%% done.\n',round(i/parms.ncells*100))
%       end
      mod_ratio = parms.mod_ratio;
      mod_base = sqrt(parms.ncells)/mod_ratio;
      x_mod = (mod(x+mod_base/2,mod_base)-mod_base/2)*mod_ratio;
      y_mod = (mod(y+mod_base/2,mod_base)-mod_base/2)*mod_ratio;
      [dist_x,dist_y]=Find_Periodic_Boundary(x(i),y(i),x_mod,y_mod,parms);
              
% 
% perform modulu on dist
% 
%        mod_base = sqrt(parms.ncells)/mod_ratio;
%        dist_x_mod = (mod(dist_x+mod_base/2,mod_base)-mod_base/2)*mod_ratio;
%        dist_y_mod = (mod(dist_y+mod_base/2,mod_base)-mod_base/2)*mod_ratio;
       %figure;plot(dist_x,dist_x_mod,'.');
       
       first_term=dist_x;
       second_term=dist_y;
       
      %inhib_width = 1/2; % 1 Dori
      excit_width = 1/2; % 1/2
      
      
      %a_excit = 2; % excitation twice inhibition
      %tmpW=-strength*( exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.R* inhib_width).^2) -  ...
      %                    a_excit*exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.R* excit_width).^2) ) ;
        tmpW=strength*exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.R* excit_width).^2);    
         
       W(i,:) = tmpW;
    
    end

figure;imagesc(reshape(tmpW,sqrt(parms.ncells),sqrt(parms.ncells)));
colorbar;

disp('')


function W=Build_Conections(x,y,parms,strength,L5_Dir)

W=zeros(parms.ncells);
    
    for i=1:parms.ncells
%       if mod(i,round(parms.ncells/10))==0
%         fprintf('Generating weight matrix. %d%% done.\n',round(i/parms.ncells*100))
%       end
      
      [dist_x,dist_y]=Find_Periodic_Boundary(x(i),y(i),x,y,parms);
              
       first_term=dist_x;
       second_term=dist_y;
       
      inhib_width = 1;
      excit_width = 1/2;
       
 
      tmpW=strength*( exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.Rgrid* inhib_width).^2) -  ...
                     exp(-(sqrt(first_term.^2+second_term.^2).^2)/2/(parms.Rgrid* excit_width).^2) ) ;
        
      W(i,:) = tmpW;
    
    end
    
    %W=sparse(W);
figure;imagesc(reshape(tmpW,sqrt(parms.ncells),sqrt(parms.ncells)));
colorbar;

disp('')



function L5=Stabilization_Of_System...
    (stabilizationTime,L5_Dir,L5,W_L5,parms,h)
tind = 0; % counts iterations of the simulation
t = 0; % measure real time of the simulation
dt =1; % time step, ms
tau = 10;
livePlot=20;
while t<stabilizationTime
      
  tind = tind+1; %% count iterations of the simulation
  t = dt*tind; %% measure real time 
  
  L5Inputs = (W_L5*L5')'+parms.I;

  % Synaptic drive only increases if input cells are over threshold 1 
  L5Inputs = L5Inputs.*(L5Inputs>parms.g);

  noise_std =1;
  noise_term = noise_std*randn([1 length(L5)]);
  
  
 %s = s + ds
  L5 = L5 + dt*(L5Inputs - L5)/tau + sqrt(dt)/tau*noise_term;
 
  
 
  % plot the stabilization
  if livePlot>0 && (livePlot==1 || mod(tind,20)==1)
      try
        fig=figure(h);
      catch
        h=figure;
      end
      set(h,'name','Activity of sheet of cells on brain''s surface');
      imagesc(reshape(L5,sqrt(parms.ncells),sqrt(parms.ncells)));
      axis square
      set(gca,'ydir','normal')
      title(sprintf('t = %.1f ms',t))
      drawnow
      
     % count=count+1;
  end
  
end
disp('')



function [pos,vels,I]=Load_Trajectory_From_File(parms,dt)


dat=load(parms.Cell_file_to_load);

  Cell= dat.S;
  clear dat;
  theta=Cell.theta;
  pos_x=Cell.pos.x;
  pos_y=Cell.pos.y;
  pos_t=Cell.pos.t;
  % sec to msec (sec*1000)
  pos_t = pos_t*1e3;
  
  if ~isempty(Cell.pos.x2)
     pos_x2=Cell.pos.x2;
     pos_y2=Cell.pos.y2;
     pos = [interp1(pos_t,pos_x,0:dt:pos_t(end));
         interp1(pos_t,pos_y,0:dt:pos_t(end));
         interp1(pos_t,pos_t,0:dt:pos_t(end));
         interp1(pos_t,pos_x2,0:dt:pos_t(end));
         interp1(pos_t,pos_y2,0:dt:pos_t(end))];
      
  end
  
  
  % interpolate down to simulation time step
  pos = [interp1(pos_t,pos_x,0:dt:pos_t(end));
         interp1(pos_t,pos_y,0:dt:pos_t(end));
         interp1(pos_t,pos_t,0:dt:pos_t(end))];
     
  pos(1:2,:) = pos(1:2,:); %m*1000=cm*10
  vels = [diff(pos(1,:)*10); diff(pos(2,:)*10)]/dt; %(m*1000/sec*1000) = (m/s)

  theta=theta-min(theta);
  theta=theta/max(theta);
  time_theta=((1:length(theta))*1/Cell.Fs-(1/Cell.Fs))*1000;
  I=interp1(time_theta',theta,0:dt:pos_t(end));



disp('')
 
function [v_pc,v_gc,time_map5,spikes_map5,spikeCoords5]=Plot_Simulation...
    (keep,s5,pc,watchCell,spikeThresh,pos,axis_x,axis_y,tind,...
    livePlot,dt,parms,t,time_map5,spikes_map5,spikeCoords5,internal,external,L5Inputs)

     
    
    
    % find the bin where the rat is at
    [min_val,xindex] =  min(abs(pos(1,tind)-axis_x));
    [min_val,yindex] =  min(abs(pos(2,tind)-axis_y));
    
    
 
     if s5(watchCell)>spikeThresh
      spikeCoords5 = [spikeCoords5; pos(1,tind) pos(2,tind)];
    end
    
    % find the bin where the rat is at
    [min_val,xindex] =  min(abs(pos(1,tind)-axis_x));
    [min_val,yindex] =  min(abs(pos(2,tind)-axis_y));
    
    % upsate the rate map
    time_map5(yindex,xindex) = time_map5(yindex,xindex)+dt;
    spikes_map5(yindex,xindex) = spikes_map5(yindex,xindex) + s5(watchCell);
    
     pc_image = reshape(pc,sqrt(parms.ncells),sqrt(parms.ncells))';   
     gc_image = reshape(s5,sqrt(parms.ncells),sqrt(parms.ncells))';
    
     % calc velocity of bumps
     
     [max_vals,max_pc_y_vec] = max(pc_image);
     [~,max_pc_x_ind] = max(max_vals);
     max_pc_y_ind = max_pc_y_vec(max_pc_x_ind);
     
     [max_vals,max_gc_y_vec] = max(gc_image);
     [~,max_gc_x_ind] = max(max_vals);
     max_gc_y_ind = max_gc_y_vec(max_gc_x_ind);
     
     persistent max_pc_x_ind_keep max_pc_y_ind_keep
     persistent max_gc_x_ind_keep max_gc_y_ind_keep
        
     tau = 100;
     try
         v_x_pc = max_pc_x_ind_keep(tau+1:end) - max_pc_x_ind_keep(1:end-tau);
         v_y_pc = max_pc_y_ind_keep(tau+1:end) - max_pc_y_ind_keep(1:end-tau);
         v_x_gc = max_gc_x_ind_keep(tau+1:end) - max_gc_x_ind_keep(1:end-tau);
         v_y_gc = max_gc_y_ind_keep(tau+1:end) - max_gc_y_ind_keep(1:end-tau);
     catch
         v_x_pc = 0; v_y_pc = 0; v_x_gc = 0; v_y_gc = 0;
     end
     
     if keep
         max_pc_x_ind_keep(end+1) = max_pc_x_ind;
         max_pc_y_ind_keep(end+1) = max_pc_y_ind;
         max_gc_x_ind_keep(end+1) = max_gc_x_ind;
         max_gc_y_ind_keep(end+1) = max_gc_y_ind;
     else
         max_pc_x_ind_keep = [];
         max_pc_y_ind_keep = [];
         max_gc_x_ind_keep = [];
         max_gc_y_ind_keep = [];
     end
     
     v_pc = sqrt(v_x_pc.^2+v_y_pc.^2);
     v_gc = sqrt(v_x_gc.^2+v_y_gc.^2);
     
     disp('');
     
     if livePlot>0 && (livePlot==1 || mod(tind,livePlot)==1)
         
         subplot(241);
         
         imagesc(pc_image);hold on;
         title('Place cell activity')
         axis square
         axis xy
         
         subplot(247);
         imagesc(reshape(internal,sqrt(parms.ncells),sqrt(parms.ncells))');hold on;
         colorbar;
         title('internal')
         axis square
         axis xy
         
         subplot(248);
         imagesc(reshape(external,sqrt(parms.ncells),sqrt(parms.ncells))');hold on;
         colorbar;
         title('external')
         axis square
         axis xy
         
         subplot(244);
         imagesc(reshape(L5Inputs,sqrt(parms.ncells),sqrt(parms.ncells))');hold on;
         colorbar;
         title('L5Inputs')
         axis square
         axis xy
         
         subplot(246);
         imagesc(reshape(pc,sqrt(parms.ncells),sqrt(parms.ncells))');hold on;
         title('Place cell activity')
         axis square
         axis xy
         
         subplot(245);
         
         imagesc(gc_image);hold on;
         [j,i]=ind2sub([sqrt(parms.ncells),sqrt(parms.ncells)],watchCell);
         plot(j,i,'*g','linewidth',5);
         axis square
         title('Population activity L5')
         set(gca,'ydir','normal')
         
         if tind>1
             
             
             %% plot ongoing velocity of bumps
             subplot(242);
             %imagesc(spikes_map5./time_map5); (was rate map)
             
             plot(1:length(v_pc),v_pc,'b');
             plot(1:length(v_gc),v_gc,'r');
             hold on;
             disp('');
             
             
             %% plot trajectory and spikes
             subplot(243);
             plot(pos(1,1:tind),pos(2,1:tind));hold on;
             % plot spikes
             if ~isempty(spikeCoords5)
                 plot(spikeCoords5(:,1),spikeCoords5(:,2),'r.')
             end
             xlim([min(axis_x) max(axis_x)]);
             ylim([min(axis_y) max(axis_y)]);
             axis square;
             title(sprintf(' Neuron Rate=%0.3f',s5(watchCell)));
             %title({'Trajectory (blue)','and spikes (red)'})
             drawnow
             
             
             
         end
         
     end
disp('')

function [x_dist,y_dist]=Find_Periodic_Boundary(s_x,s_y,x,y,parms)


%%
dist_x(1,:)=s_x*ones(1,parms.ncells)-x;
dist_y(1,:)=s_y*ones(1,parms.ncells)-y;
dist_all(1,:)=dist_x(1,:).^2+dist_y(1,:).^2;
%%
dist_x(2,:)=s_x*ones(1,parms.ncells)-x-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(2,:)=s_y*ones(1,parms.ncells)-y;
dist_all(2,:)=dist_x(2,:).^2+dist_y(2,:).^2;
%%
dist_x(3,:)=s_x*ones(1,parms.ncells)-x+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(3,:)=s_y*ones(1,parms.ncells)-y;
dist_all(3,:)=dist_x(3,:).^2+dist_y(3,:).^2;
%%
dist_x(4,:)=s_x*ones(1,parms.ncells)-x;
dist_y(4,:)=s_y*ones(1,parms.ncells)-y-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(4,:)=dist_x(4,:).^2+dist_y(4,:).^2;
%%
dist_x(5,:)=s_x*ones(1,parms.ncells)-x;
dist_y(5,:)=s_y*ones(1,parms.ncells)-y+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(5,:)=dist_x(5,:).^2+dist_y(5,:).^2;
%%        
dist_x(6,:)=s_x*ones(1,parms.ncells)-x+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(6,:)=s_y*ones(1,parms.ncells)-y+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(6,:)=dist_x(6,:).^2+dist_y(6,:).^2;
          
%%        
dist_x(7,:)=s_x*ones(1,parms.ncells)-x-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(7,:)=s_y*ones(1,parms.ncells)-y+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(7,:)=dist_x(7,:).^2+dist_y(7,:).^2;
%%                  
dist_x(8,:)=s_x*ones(1,parms.ncells)-x+sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(8,:)=s_y*ones(1,parms.ncells)-y-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(8,:)=dist_x(8,:).^2+dist_y(8,:).^2;        
%%
dist_x(9,:)=s_x*ones(1,parms.ncells)-x-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_y(9,:)=s_y*ones(1,parms.ncells)-y-sqrt(parms.ncells)*ones(1,parms.ncells);
dist_all(9,:)=dist_x(9,:).^2+dist_y(9,:).^2;
        
        
        % Select respective least distances:
        [min_dist,ind] = min(dist_all);
        
        for i=1:length(ind)
          
            x_dist(i)=dist_x(ind(i),i);
            y_dist(i)=dist_y(ind(i),i);
        end
        
%         x=(tmp_x-s_x*ones(1,parms.ncells))*-1;
%         y=(tmp_y-s_y*ones(1,parms.ncells))*-1;
%figure;imagesc(reshape(min_dist,sqrt(parms.ncells),sqrt(parms.ncells)));
        
disp('')