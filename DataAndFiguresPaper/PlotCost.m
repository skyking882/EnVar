%% Plot evolution of cost function with iterations
J = importdata('J.txt'); % Introduce text file here
%subplot(1,2,1)
%yyaxis left
plot([0:(length(J)-1)],J/J(1,1),'-*','LineWidth',2,'MarkerSize',20,'MarkerEdgeColor','black')
grid on
set(gca,'FontSize',18,'LineWidth',2)
xlabel('$i$','interpreter','latex','FontSize',38)
ylabel('$\frac{{J}}{{J_{0}}}$','interpreter','latex','FontSize',38)
xticks(0:5:length(J))
yticks(0:0.2:1)
pbaspect([1 1 1])