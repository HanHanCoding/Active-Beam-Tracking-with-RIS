clear
fs2=10;

sample_x = 25;
sample_y = 25;

user_coor_x = importdata('user_coor_x.mat');
user_coor_y = importdata('user_coor_y.mat');
range_coordinate_all_x = importdata('range_coordinate_x.mat');
range_coordinate_all_y = importdata('range_coordinate_y.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check the user path
N_w_using = 30;
total_w_number = 36;
total_frames = N_w_using * total_w_number;
z = 1:1:total_frames;

figure(1)
% plot3(user_coor_x,user_coor_y,z,'-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF')
% xlabel('User coordinate - x coordinate (m)','Interpreter','latex','FontSize',fs2+2)
% ylabel('User coordinate - y coordinate (m)','Interpreter','latex','FontSize',fs2+2)
% zlabel('time frames')
% grid on
plot(user_coor_x,user_coor_y,'-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF')
xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2+2)
ylabel('$y$-coordinate (m)','Interpreter','latex','FontSize',fs2+2)
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% change the dimension and prepare for the rate plot
rate_range_all_DNN_trainable = importdata('rate_range_all.mat');

% for benchmark DNN random v (trainable)
rate_range_DNN_trainable = zeros(sample_x, sample_y, total_frames);
% for x
for kk = 1:total_frames
    for ii = 1:sample_x
        % for y
        for jj = 1:sample_y
            rate_range_DNN_trainable(ii, jj, kk) = rate_range_all_DNN_trainable(sample_y*(ii-1)+jj, 1, kk);
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualize the Benchmark DNN - trainable
%plot the rate: frame[1, 91, 181, ..., 901, 901]
figure(2)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1coordinate (m)
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,1).') %%%%%%%%%%%%%%%%% 注意！这里对rate_range这个矩阵需要转置一下！！！
colorbar
shading interp
hold on
plot(user_coor_x(1,1), user_coor_y(1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
ylabel('$y$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$1$st transmission frame','Interpreter','latex','FontSize',fs2)
% 
% frame = 181
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,181).')
colorbar
shading interp
hold on
plot(user_coor_x(181,1), user_coor_y(181,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$7$th transmission frame','Interpreter','latex','FontSize',fs2)
% 
% frame = 271
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,271).')
colorbar
shading interp
hold on
plot(user_coor_x(271,1), user_coor_y(271,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$10$th transmission frame','Interpreter','latex','FontSize',fs2)
% frame = 451
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,451).')
colorbar
shading interp
hold on
plot(user_coor_x(451,1), user_coor_y(451,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$16$th transmission frame','Interpreter','latex','FontSize',fs2)
% 
% frame = 721
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,721).')
colorbar
shading interp
hold on
plot(user_coor_x(721,1), user_coor_y(721,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$25$th transmission frame','Interpreter','latex','FontSize',fs2)
% 
% frame = 811
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,811).')
colorbar
shading interp
hold on
plot(user_coor_x(811,1), user_coor_y(811,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$28$th transmission frame','Interpreter','latex','FontSize',fs2)
% 
% frame = 901
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, rate_range_DNN_trainable(:,:,901).')
colorbar
shading interp
hold on
plot(user_coor_x(901,1), user_coor_y(901,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')

xlim([15,40])
ylim([-20,8])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$31$th transmission frame','Interpreter','latex','FontSize',fs2)