function[dY] = dResNetMatVec(dK,db,dY0,K,Yall,dA,param)
% [Z] = dResNetMatVec(dK,db,dY0,K,b,Y0,param)
% 

P  = param.P;
N  = length(P);
dY = dY0;
for i=1:N 
    dY = P{i}*dY + dA{i} .* (dK{i}*Yall{i} + K{i}*dY + db{i});
end

%    [  I   O   O   O  O][Y2]     [f(K1*Y1 + b1)]   [P1*Y1]
%    [ -P2  I   O   O  O][Y3]     [f(K2*Y2 + b2)]   [ O   ]
%    [  O  -P3  I   O  O][Y4]  =  [f(K3*Y3 + b3)] + [ O   ]
%    [  O   O  -P4  I  O][Y5]     [f(K4*Y4 + b4)]   [ O   ]
%    [  O   O   O  -P5 I][Y6]     [f(K5*Y5 + b5)]   [ O   ]
%
%  Differentiate to obrain
%
%    [  I   O   O   O  O][dY2]     [f'(K1*Y1 + b1)*(dK1*Y1 + K1*dY1 + db1)] +  [P1*dY1]
%    [ -P2  I   O   O  O][dY3]     [f'(K2*Y2 + b2)*(dK2*Y2 + K2*dY2 + db2)] 
%    [  O  -P3  I   O  O][dY4]  =  [f'(K3*Y3 + b3)*(dK3*Y3 + K3*dY3 + db3)]
%    [  O   O  -P4  I  O][dY5]     [f'(K4*Y4 + b4)*(dK4*Y4 + K4*dY4 + db4)]
%    [  O   O   O  -P5 I][dY6]     [f'(K5*Y5 + b5)*(dK5*Y5 + K5*dY5 + db5)]
