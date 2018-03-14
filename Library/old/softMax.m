function[E,dE,d2E] = softMax(W,Y,C)
%[E] = softMax(W,b,Y,C)
%

if nargin == 0
   runMinExample;
   return
end

Y = [Y, ones(size(Y,1),1)];
% the linear model
S = Y*W;

% make sure that the largest number in every row is 0
s = max(S,[],2);
S = S-s;


% The cross entropy
expS = exp(S);
sS   = sum(expS,2);

E = -C(:)'*S(:) +  sum(log(sS)); 
E = E/size(Y,1);

if nargout > 1
    dE  = -C + expS .* 1./sS;
    dE  = (Y'*dE)/size(Y,1);
    
    d2E1 = @(V) 1/size(Y,1) * (Y'*( (expS./sS) .* (Y*V)));
    d2E2 = @(V) -1/size(Y,1) *(Y'*  (expS.* ((1./sS.^2) .* sum(expS.*(Y*V),2))));
    d2E  = @(V) d2E1(V) + d2E2(V);
end

end

function runMinExample

vec = @(x) x(:);
nex = 100;
Y = hilb(500)*255;
Y = Y(1:nex,:);
C = ones(nex,3);
C = C./sum(C,2);
b = 0;
W = hilb(501);
W = W(:,1:3);
E = softMax(W,Y,C);
[E,dE,d2E] = softMax(W,Y,C);

h = 1;
rho = zeros(20,3);
dW = randn(size(W));
for i=1:20
    E1 = softMax(W+h*dW,Y,C);
    t  = abs(E1-E);
    t1 = abs(E1-E-h*dE(:)'*dW(:));
    t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW)));
    
    fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
    rho(i,1) = abs(E1-E);
    rho(i,2) = abs(E1-E-h*dE(:)'*dW(:));
    rho(i,3) = t2;
    h = h/2;
end

rho(2:end,:)./rho(1:end-1,:)

end