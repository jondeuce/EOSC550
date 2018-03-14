function MNIST_linearmodel(imgs, labels)
if nargin < 2
    [imgs, labels] = readMNIST(	't10k-images.idx3-ubyte', ...
        't10k-labels.idx1-ubyte', 10000, 0);
end

vec = @(x) x(:);

Y = double(imgs);
Y = permute(Y,[3,1,2]);
Y = reshape(Y,size(Y,1),[]);
C = zeros(size(Y,1),10);
for i = 1:size(Y,1)
    C(i,labels(i)+1) = 1;
end

% Solution for the model C = Y*W -> W = (Y'*Y)\(Y'*C)
W = (Y'*Y)\(Y'*C);

% Solution for the model C = Y*W -> W = (Y'*Y)\(Y'*C)
a  = 1e-3;
Wa = (Y'*Y + a*eye(size(Y,2)))\(Y'*C);

fprintf('rel. frob. err. (LSQ)  = %3.2e\n', norm(vec(Y*W-C))/norm(vec(C)));
fprintf('rel. frob. err. (rLSQ) = %3.2e\n', norm(vec(Y*Wa-C))/norm(vec(C)));

figure, imagesc(C), title('MNIST Data')
figure, imagesc(Y*W), title('MNIST LSQ Linear Model')
figure, imagesc(Y*Wa), title('MNIST LSQ Linear Model')

end

function P = predictions(Y,W,C)

[~,preds] = max(abs(Y*W-C),[],2);
P = zeros(size(C));
P(:,preds) = 1;

end