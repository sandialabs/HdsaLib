clear
close all
clc

x = linspace(0,1,51)';
z_lofi = 1+x;
z_hifi = (1+x)/(1.2^(1/3));
z_update = load('z_update.txt');

figure,
hold on
plot(x, z_hifi, 'color', 'black', 'LineWidth', 3)
plot(x, z_lofi, 'color', 'cyan', 'LineWidth', 3)
plot(x,z_update,'--','LineWidth',3,'Color','red')


