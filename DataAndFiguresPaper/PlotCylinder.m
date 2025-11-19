%% Create array of XY coordinates to plot a cylinder
a = importdata('a_e.txt'); % Change this to the desired txt file
N = length(a);
M = 500000;
theta = (linspace(0, 2*pi, M))';
R = zeros(M,1);

for i = 1:N   
        R = R + a(i)*cos((i-1)*theta);
end

x = R.*cos(theta);
y = R.*sin(theta);

% Plot
figure
h1 = plot(x,y,'-','linewidth',2)
set(gca,'FontSize',18,'LineWidth',2)
xlabel('$x$','interpreter','latex','FontSize',38)
ylabel('$y$','interpreter','latex','FontSize',38)
grid on
hold on
xlim([-1, 1.5])
ylim([-1.25,1.25])
pbaspect([1 1 1])