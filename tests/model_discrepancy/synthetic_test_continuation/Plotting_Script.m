clear
close all
clc

surpress_figures = true; %true;

diff = [];
post_z_mean = load('posterior_update_mean.txt')';
% post_z_samples = zeros(51,100);
% for k = 1:100
%     post_z_samples(:,k) = load(['posterior_update_samples/Vector_',num2str(k),'.txt']);
% end
post_z_mean_sabl = load('Sabl_Output.mat','z_k').z_k;
% post_z_samples_sabl = load('Sabl_Output.mat','post_z_samples').post_z_samples;

local_diff = norm(post_z_mean - post_z_mean_sabl);
diff = [diff;local_diff];

% local_diff = norm(post_z_samples - post_z_samples_sabl);
% diff = [diff;local_diff];


if ~surpress_figures

    x = linspace(0,1,51)';

    figure,
    hold on
    plot(x, (1 + x) / (1.2^(1 / 3)), 'color', 'black', 'LineWidth', 3)
    plot(x, 1 + x, 'color', 'cyan', 'LineWidth', 3)
    plot(x,post_z_mean,'--','LineWidth',3,'Color','red')
    plot(x,post_z_samples,'LineWidth',3,'Color',.9*ones(3,1))
    plot(x, (1 + x) / (1.2^(1 / 3)), 'color', 'black', 'LineWidth', 3)
    plot(x, 1 + x, 'color', 'cyan', 'LineWidth', 3)
    plot(x,post_z_mean,'--','LineWidth',3,'Color','red')
    title('HdsaLib')
    figure,
    hold on
    plot(x, (1 + x) / (1.2^(1 / 3)), 'color', 'black', 'LineWidth', 3)
    plot(x, 1 + x, 'color', 'cyan', 'LineWidth', 3)
    plot(x,post_z_mean_sabl,'--','LineWidth',3,'Color','red')
    plot(x,post_z_samples_sabl,'LineWidth',3,'Color',.9*ones(3,1))
    plot(x, (1 + x) / (1.2^(1 / 3)), 'color', 'black', 'LineWidth', 3)
    plot(x, 1 + x, 'color', 'cyan', 'LineWidth', 3)
    plot(x,post_z_mean_sabl,'--','LineWidth',3,'Color','red')
    title('Sabl')

end
