clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.9290, 0.6940, 0.1250]; 
color7=[0.2, 0.3, 0.7]; 
color8=[0, 0, 0]; 
color9=[0.5, 0.5, 0.5];
fs2=14;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loading the data
optimal_rate_perframe = importdata('optimal_rate_perframe.mat');
rand_rate_perframe = importdata('rand_rate_perframe.mat');

trained_rate_perframe_DNN_random = importdata('trained_rate_perframe_DNN_random.mat');
trained_rate_perframe_DNN_random_trainable = importdata('trained_rate_perframe_DNN_random_trainable.mat');
trained_rate_perframe_LSTM_random = importdata('trained_rate_perframe_LSTM_random.mat');
trained_rate_perframe_LSTM_random_trainable = importdata('trained_rate_perframe_LSTM_random_trainable.mat');
trained_rate_perframe_LSTM_DNNdesign_v = importdata('trained_rate_perframe_LSTM_DNNdesign_v.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot

figure(1)
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
frames = 1:1:450;
plot(frames,optimal_rate_perframe(1:length(frames)),'-','color',color1,'lineWidth',1.7,'markersize',4);
hold on
p1 = plot(frames,trained_rate_perframe_LSTM_DNNdesign_v(1:length(frames)),'-o','color',color2,'lineWidth',1.7,'markersize',4);
p1.MarkerIndices = 1:30:length(frames);
p2 = plot(frames,trained_rate_perframe_LSTM_random_trainable(1:length(frames)),'-d','color',color3,'lineWidth',1.7,'markersize',4);
p2.MarkerIndices = 1:30:length(frames);
p3 = plot(frames,trained_rate_perframe_LSTM_random(1:length(frames)),'->','color',color4,'lineWidth',1.7,'markersize',4);
p3.MarkerIndices = 1:30:length(frames);
p4 = plot(frames,trained_rate_perframe_DNN_random_trainable(1:length(frames)),'-<','color',color5,'lineWidth',1.7,'markersize',4);
p4.MarkerIndices = 1:30:length(frames);
p5 = plot(frames,trained_rate_perframe_DNN_random(1:length(frames)),'-s','color',color6,'lineWidth',1.7,'markersize',4);
p5.MarkerIndices = 1:30:length(frames);

plot(frames,rand_rate_perframe(1:length(frames)),'-','color',color8,'lineWidth',1.7,'markersize',4);
grid on

lg=legend('Phase matching w/ perfect CSI',...
    'Proposed active sensing based approach',...
    'LSTM-based benchmark - fixed sensing vectors $\{ \mathbf{v}_l \}_{l = 1}^{L}$ learned from channel statistics',...
    'LSTM-based benchmark - fixed sensing vectors $\{ \mathbf{v}_l \}_{l = 1}^{L}$, randomly generated',...
    'DNN-based benchmark - fixed sensing vectors $\{ \mathbf{v}_l \}_{l = 1}^{L}$ learned from channel statistics',...
    'DNN-based benchmark - fixed sensing vectors $\{ \mathbf{v}_l \}_{l = 1}^{L}$, randomly generated',...
    'Random downlink RIS reflection coefficients $\mathbf{w}^{(k)}$',...
    'Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Interpreter','latex');
xlabel('Transmission frame','Interpreter','latex','FontSize',fs2);
ylabel('Downlink data rate (bps/Hz)','Interpreter','latex','FontSize',fs2);
xlim([1,450])

xticks([1,31,61,91,121,151,181,211,241,271,301,331,361,391,421])
xticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15'}); set(lg,'Interpreter','latex')
