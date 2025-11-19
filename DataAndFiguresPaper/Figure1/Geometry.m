figure
plot([0.9 6.1],[1 1],'k','LineWidth',3)
hold on
plot([1 1],[1 3.1],'k','LineWidth',3)
plot([0.9 6.1],[3 3],'k','LineWidth',3)
plot([6 6],[1 3],'k','LineWidth',3)

xticks('')
yticks('')
axis([0.99 6.01 0.99 3.01])

%%
%% Create array of XY coordinates to plot a cylinder
a = importdata('a_Initial.txt'); % Change this to the desired txt file
N = length(a);
M = 4001;
theta = (linspace(0, 2*pi, M))';
R = zeros(M,1);

RR=0
for i = 1:N   
        R = R + a(i)*cos((i-1)*theta);
end

x = R.*cos(theta);
y = R.*sin(theta);

% Plot
figure
h1 = plot(x,y,'k','linewidth',2)
set(gca,'FontSize',18,'LineWidth',2)``
axis equal
hold on
axis([-2 5 -1.7 1.7])
xticks('')
yticks('')

txt = '{\bf {\partial \Omega}}_{inlet}';
text(-1.75,0.4,txt,'Rotation',-90,'FontSize',19,'Interpreter','tex')

txt = '{\bf {\partial \Omega}}_{outlet}';
text(4.8,0.4,txt,'Rotation',-90,'FontSize',19,'Interpreter','tex')

txt = '{\bf {\partial \Omega}}_{bottom}';
text(3,-1.45,txt,'Rotation',0,'FontSize',19,'Interpreter','tex')

txt = '{\bf {\partial \Omega}}_{top}';
text(3,1.5,txt,'Rotation',0,'FontSize',19,'Interpreter','tex')

txt = '{\bf {\partial \Omega}}_{wall}';
text(0.7,0.25,txt,'Rotation',0,'FontSize',19,'Interpreter','tex')

txt = '${U(t)}$';

text(-2,3,txt,'Rotation',0,'FontSize',19,'Interpreter','latex')
