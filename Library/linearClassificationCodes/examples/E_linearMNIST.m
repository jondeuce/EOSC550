%% Linear classification of MNIST
function [W,param,train_error] = E_linearMNIST
%% setting all the path
% addpath ../Data
% addpath ../softMax
% addpath ../optimization
% addpath ../regularization

%% Upload the data and set it for training
Ytest = loadMNISTImages('t10k-images.idx3-ubyte')';
ctest = loadMNISTLabels('t10k-labels.idx1-ubyte');
nval  = length(ctest);
Ctest = full(sparse(1:nval,ctest+1,ones(nval,1),nval,10));

Y = loadMNISTImages('train-images.idx3-ubyte')';
c = loadMNISTLabels('train-labels.idx1-ubyte');
nex = length(c);
C = full(sparse(1:nex,c+1,ones(nex,1),nex,10));

%%  setup the function and regularization
param.nc = 10; % number of classes
param.n  = [28,28]; % image sizes
param.h  = 1./param.n;
param.L  = getLaplacian(param.n,param.h);

reg = @(W,param) genTikhonov(W,param);
fun = @(W,~)     softMax(W,Y,C);

%% Initialize weights
W = randn(size(Y,2)+1,10);
% W = zeros(size(Y,2)+1,10);
W  = W(:);

%% Steepest Descent Train
% param.alpha     = 1e-5;
% param.maxIter   = 10;
% param.lsMaxIter = 10;
% 
% [W,hist] = steepestDescent(fun,reg,W,param);


%% Newton parameter sweep
param.alpha     = 1e-5; 1e-6;
param.maxIter   = 5;    20;
param.lsMaxIter = 5;    10;
param.slvTol    = 1e-6; 1e-6;
param.slvIter   = 5;    10;

xlb = [-10.0, -10.0];
x0  = [ -5.0,  -6.0];
xub = [ -1.0,  -1.0];

objfun = @(x) newtonReg_driver(x,W,fun,reg,param,Y,Ytest,C,Ctest);
% x = patternsearch(objfun,x0,[],[],[],[],xlb,xub);
x = particleswarm(objfun,numel(x0),xlb,xub,...
    optimoptions(@particleswarm,'PlotFcn','pswplotbestf'));

[train_error,W] = objfun(x);

%% Visualize W
plot_W(W,param)

end

function [t_err, W] = newtonReg_driver(x,W,fun,reg,param,Y,Ytest,C,Ctest)

%% Regularized Newton Descent Train
param.alpha     = 10^x(1);
param.slvTol    = 10^x(2);

%% Calculate resulting accuracy
[W,hist] = newtonReg(fun,reg,W,param);
W      = reshape(W,[],10);
Strain = [Y, ones(size(Y,1),1)]*W;
S      = [Ytest, ones(size(Ytest,1),1)]*W;

% the probability function
htrain = bsxfun(@rdivide, exp(Strain), sum(exp(Strain),2));
h      = bsxfun(@rdivide, exp(S), sum(exp(S),2));

% Find the largesr entry at each row
[~,ind] = max(h,[],2);
Cv = zeros(size(Ctest));
Ind = sub2ind(size(Cv),[1:size(Cv,1)]',ind);
Cv(Ind) = 1;

[~,ind] = max(htrain,[],2);
Cpred = zeros(size(C));
Ind = sub2ind(size(Cpred),(1:size(Cpred,1)).',ind);
Cpred(Ind) = 1;

t_err = (nnz(abs(C-Cpred))/2)/nnz(C);
v_err = (nnz(abs(Cv-Ctest))/2)/nnz(Cv);

disp(param)
fprintf('Testing    Error    %.2e\n', t_err);
fprintf('Validation Error    %.2e\n', v_err);
fprintf('Testing    Accuracy %.2f%%\n', 100*(1-t_err));
fprintf('Validation Accuracy %.2f%%\n', 100*(1-v_err));


end

function plot_W(W,param)
W = reshape(W,[],10);

figure(1);
for i=1:9
    w = W(1:end-1,i);
    subplot(3,3,i)
    imagesc(reshape(w,param.n(1),param.n(2)));
    colorbar;
end

end

