%% parameters
int=0;                                                 % Put zero if you want the P2 element interpolation, 1 for to truncate transient, 2 for nothing 
m=20000;                                               % Put a relevent space point for time signal  mUV=20001 for U and V because not the same element than P
mP=208;                                                % Put a relevent space point for time signal  mP= 2800 for P
n=1 ;                                                  % Frequency of save data, taken from FreeFem
dt=n*0.01;                                             % Time step in order to compute the frequency Samples and the frequency vector
Re=100;                                                % Reynolds number 

%% Run this just once to place the cylinder at the origin
% points_opt(1,:) = points_opt(1,:)-20;
% points_opt(2,:) = points_opt(2,:)-15;

%% Load Mesh 
% % This is the P1 mesh used only for pressure  
% % [points, seg, tri]=importfilemesh('CylinderWakeP1.msh'); 
% 
% % This is the P2 mesh used for U,V,W  
%  [points_opt, seg_opt, tri_opt]=importfilemesh('CylinderWakeP2.msh'); 
%  
%  figure(3)
%  pdeplot(points_opt, seg_opt, tri_opt);

%% Load data 
[datW_opt]=LoadDat('w_Noise.bb',int,'matInterp.txt',m,n);
[datU_opt]=LoadDat('Um_Noise.bb',int,'matInterp.txt',m,n);

%% Plot
figure(3)
pdeplot(points_opt,seg_opt,tri_opt,'xydata',datW_opt,'mesh','off','colormap','jet');
    axis equal
    caxis([-1 1])
    hcb=colorbar;
    xlabel('x','FontSize',25,'FontWeight','bold','FontName','Times New Roman');
    ylabel('y','FontSize',25,'FontWeight','bold','FontName','Times New Roman');
    set(get(hcb,'Title'),'String','', 'FontSize', 25, 'fontName','Times','FontWeight','bold','Interpreter','Latex')
    set(gca, 'FontSize', 25, 'fontName','Times');

xlim([-2 10])
pbaspect([1 1 1])

%%
figure(2)
%pdecont(points_opt,tri_opt,datW_opt,10)
pdeplot(points_opt,seg_opt,tri_opt,'xydata',datU_opt,'mesh','off','Contour','on','Levels',100)
