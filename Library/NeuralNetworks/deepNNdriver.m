clear
param.act = @smoothRelU;

% Width of each block (including initial conditions)
n = [2  5   5   5   5   10; ...
     5  5   5   5  10   3];

 for i=1:size(n,2)
    if n(1,i) ~= n(2,i)
        P{i} = opZero(n(2,i),n(1,i));
    else
        P{i} = opEye(n(1,i));
    end
    K{i} = randn(n(2,i),n(1,i));
    b{i} = randn(n(2,i),1);
end
N = length(P);
param.P = P;
Y0 = randn(2,100);
%% Run the NN forward
[Y,Yall,dA] = resNetForward(K,b,Y0,param);

dY0 = randn(size(Y0))*0;

for i=1:N 
    dK{i} = randn(size(K{i}))*1e-3;
    K1{i} = K{i} + dK{i};
    db{i} = randn(size(b{i})) * 1e-3;
    b1{i} = b{i} + db{i};
end
Y1 = resNetForward(K1,b1,Y0+dY0,param);

dY = dResNetMatVec(dK,db,dY0,K,Yall,dA,param);

fprintf('=== Derivative test ========================\n')
fprintf('%3.2e  %3.2e\n',norm(Y1(:)-Y(:)), norm(Y1(:)-Y(:)-dY(:)))

fprintf('=== Adjoint test ========================\n')
dZ = dResNetMatVec(dK,db,dY0,K,Yall,dA,param);

dW = randn(size(dZ));
t1 = dZ(:)'*dW(:);

[dK1,db1] = dResNetMatVecT(dW,K,Yall,dA,param);

dk  = cell2vec(dK);
dk1 = cell2vec(dK1);
d  = cell2vec(db);
d1 = cell2vec(db1);


t2  = dk'*dk1 + d'*d1;
fprintf('%3.2e  %3.2e\n',t1,t2)
