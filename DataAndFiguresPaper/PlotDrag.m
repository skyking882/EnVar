%% Plot evolution of drag with time
A = pi/4;
L = 2*sqrt(A/pi);
Drag = importdata('Drag.dat'); % Introduce .dat file
Cd = (Drag(:,2)*L + Drag(:,3))*2/L;
Cd_v = Drag(:,2)*L*2/L;
Cd_p = Drag(:,3)*2/L;

% Get mean values
Cd_mean = mean(Cd(10000:end));
Cd_v_mean = mean(Cd_v(10000:end));
Cd_p_mean = mean(Cd_p(10000:end));

t = 0:0.01:(length(Cd)-1)*0.01;

% Plot
%subplot(1,2,2)
plot(t(10000:end),Cd(10000:end),'linewidth',1)
grid on
hold on
set(gca,'FontSize',18,'LineWidth',2)
xlabel('$t$','interpreter','latex','FontSize',38)
ylabel('$C_{D}$','interpreter','latex','FontSize',38)
%legend({'Circular shape','Optimal shape'},'location','northeast','FontSize',20)  