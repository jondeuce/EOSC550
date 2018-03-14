function [w,rho,eta,W] = cgls(Y,c,k,w0)
%     w = cgls(Y,c,k)
%     
n = size(Y,2);
if nargin == 4
    w = w0;
else
    w = zeros(n,1);
end
W = w;
r = c-Y*w;
d = Y'*r;   
normr2 = d'*d;
rho = zeros(k,1); eta = zeros(k,1);
for j=1:k
    Ad = Y*d; 
    alpha = normr2/(Ad'*Ad);
    w  = w + alpha*d;
    r  = r - alpha*Ad;
    s  = Y'*r;
    normr2New = s'*s;
    beta = normr2New/normr2;
    normr2 = normr2New;
    d = s + beta*d;
    rho(j) = norm(r)/norm(c);
    eta(j)  = norm(w);
    W  = [W w];
    fprintf('%3d   %3.2e   %3.2e\n',j,norm(r)/norm(c),norm(w))
end