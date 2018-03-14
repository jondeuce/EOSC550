function[L] = getLaplacian(n,h)
%[L] = getLaplacian(n,h)
%

d2dx = 1/h(1)^2*spdiags(ones(n(1),1)*[1  -2  1],-1:1,n(1),n(1));
d2dy = 1/h(2)^2*spdiags(ones(n(2),1)*[1  -2  1],-1:1,n(2),n(2));

L = kron(speye(n(2)),d2dx) + kron(d2dy,speye(n(1))); 
