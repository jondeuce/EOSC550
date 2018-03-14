function [E,dE,d2E] = softMax(W,Y,C)
%[E,dE,d2E] = softMax(W,Y,C,isVecW)

if nargin == 0
   runMinExample;
   return
end

isVecW = (size(W,2) == 1);
if isVecW; W = reshape(W,[],size(C,2)); end
Y = [Y, ones(size(Y,1),1)];

% the linear model
S = Y*W;

% make sure that the largest number in every row is 0
s = max(S,[],2);
S = bsxfun(@minus, S, s);

% The cross entropy
expS = exp(S);
sS   = sum(expS,2);

E = -C(:)'*S(:) +  sum(log(sS)); 
E = E/size(Y,1);

if nargout > 1
    dE  = bsxfun(@rdivide, -C + expS, sS);
    dE  = (Y'*dE)/size(Y,1);
    if isVecW
        dE = dE(:);
    end
    
    mat = @(v) reshape(v,size(Y,2),[]);
    vec = @(V) V(:);
    
    if isVecW
        % d2E1 and d2E2 are matrices of size W, reshaped to column in d2E
        d2E1 = @(v) ( 1/size(Y,1)) * ...
            ( Y' * ( bsxfun(@rdivide, expS, sS) .* (Y*mat(v)) ) );
        d2E2 = @(v) (-1/size(Y,1)) * ...
            ( Y' * ( bsxfun(@times, expS, sum(expS.*(Y*mat(v)),2)./sS.^2 ) ) );
        d2E  = @(v)  vec( d2E1(v) + d2E2(v) );
    else
        % d2E1, d2E2, and d2E are all matrices of size W
        d2E1 = @(V) ( 1/size(Y,1)) * ...
            ( Y' * ( bsxfun(@rdivide, expS, sS) .* (Y*V) ) );
        d2E2 = @(V) (-1/size(Y,1)) * ...
            ( Y' * ( bsxfun(@times, expS, sum(expS.*(Y*V),2)./(sS.^2) ) ) );
        d2E  = @(V) d2E1(V) + d2E2(V);
    end
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