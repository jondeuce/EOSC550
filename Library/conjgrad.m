function [x,res,iter] = conjgrad(A, b, x0, tol, maxIter, dotfun)
%[x,res,iter] = conjgrad(A, b, x0, tol, maxIter, dotfun)

% [s,pcg_flag,pcg_relres,pcg_iter] = pcg(d2F,-dF,cgTol,cgIter);

if nargin < 6; dotfun = @(x,y) dot(x,y); end
if nargin < 5; maxIter = min(size(b,1),100); end
if nargin < 4; tol = 1e-6; end
if nargin < 3; x0 = []; end

if ~isa(A,'function_handle')
    % A should be a function handle
    A = @(x) A*x;
end

if isempty(x0)
    x = zeros(size(b));
    r = b;
else
    x = x0;
    r = b - A(x);
end
p = r;
rsold = dotfun(r,r);

for iter = 1:maxIter
    Ap = A(p);
    alpha = rsold / dotfun(p,Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = dotfun(r,r);
    res = sqrt(rsnew);
    if res < tol
        break;
    end
    p = r + (rsnew / rsold) * p;
    rsold = rsnew;
end

end