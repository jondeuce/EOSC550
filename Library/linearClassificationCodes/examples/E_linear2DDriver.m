addpath ../softMax
addpath ../optimization

clear 
close all
testSoftMax;

fun = @(W,~) softMax(W,Y,C);
param.maxIter = 10;
x0 = randn(6,1);
xSD = steepestDescent(fun,x0,param);

fprintf('\n\n\n');
xNW = newtonOpt(fun,x0,param);

w = reshape(xNW,3,2);
figure(1)
t = linspace(-3,4,129);
q = -w(1,1)/w(2,1)*t -w(3,1)/w(2,1);
hold on
plot(t,q,'k','linewidth',3)
hold off