% least squares 
clear
close all

vec = @(x) x(:);
addpath ../../../MNIST/
addpath ../../../cifar-10-batches-mat/


MNIST = 1;
if MNIST
    Y      = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');

    Y = Y';
else  % CIFAR10
    Y = []; lb = [];
    load data_batch_1.mat
    Y = [Y; data]; lb = [lb; labels]; 
    load data_batch_2.mat
    Y = [Y; data];  lb = [lb; labels];
    load data_batch_3.mat
    Y = [Y; data]; lb = [lb; labels];
    load data_batch_4.mat
    Y = [Y; data];  lb = [lb; labels];
    load data_batch_5.mat
    Y = [Y; data];  lb = [lb; labels];
    labels = lb;
end

Y = double(Y);
C = zeros(size(Y,1),10);
for i=1:size(Y,1)
    C(i,labels(i)+1) = 1;
end

alpha = 1e-3;
I     = speye(size(Y,2));
W = (Y'*Y + alpha*I)\(Y'*C);

fprintf('error = %3.2e\n',norm(vec(Y*W-C))/norm(C(:)))

%%
R = Y*W;
P = zeros(size(R));

for i=1:size(P,1)
    [~,j] = max(abs(R(i,:)));
    P(i,j) = 1;
end

fprintf('err %3.2e\n',1-nnz(P-C)/2/nnz(C))