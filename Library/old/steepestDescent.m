function [w,rho,eta,W] = steepestDescent(Y,c,maxIter,w0,alphaIn)
% [w,rho,eta] = steepestDescent(Y,c,k)
% 

if nargin == 3
    w = zeros(size(Y,2),1);
    r = c;
else
    w = w0;
    r = c - Y*w;
end

rho = zeros(maxIter,1); eta = zeros(maxIter,1);
W = w;
for i=1:maxIter
    
    % function value
    rho(i) = 0.5*(r'*r); eta(i) = w'*w;
    fprintf('%3d   %3.2e   %3.2e\n',i,rho(i),eta(i));
    % negative gradient
    s  = Y'*r;
    Ys = Y*s; 
    % compute learning rate (if needed)
    if exist('alphaIn')
        alpha = alphaIn;
    else
        alpha = (r'*Ys)/norm(Ys)^2;
    end
    
    % update
    w  = w + alpha*s;
    r  = r - alpha*Ys; %c - Y*w;
    W = [W, w];
end