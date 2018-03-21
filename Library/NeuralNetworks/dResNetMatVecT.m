function[dK,db,dY0] = dResNetMatVecT(dY,K,Yall,dA,param)
% [dK,db,dY0] = dResNetMatVecT(dY,K,Yall,dA,param)
% 

P  = param.P;
N  = length(P);
for i=N:-1:1 
    dYdA  = dY .* dA{i};
    dK{i} = dYdA * Yall{i}';
    db{i} = sum(dYdA,2);
    
    dY = P{i}'*dY +  K{i}'*dYdA;
end
dY0 = dY;


% A1 = f'(K1(Y1+b1)
%    [  I   O   O   O  O][dY2]     [A1*K1*dY1] + A1*dK1*Y1 + A1*db1 + P1*dY1
%    [ -P2  I   O   O  O][dY3]     [A2*K2*dY2] + A2*dK2*Y2 + A2*db2
%    [  O  -P3  I   O  O][dY4]  =  [A3*K3*dY3] + A3*dK3*Y3 + A3*db3
%    [  O   O  -P4  I  O][dY5]     [A4*K4*dY4] + A4*dK4*Y4 + A4*db4
%    [  O   O   O  -P5 I][dY6]     [A5*K5*dY5] + A5*dK5*Y5 + A5*db5
%     
%    [  I        O   O   O  O][dY2]    A1*dK1*Y1 + A1*db1 + (P1 + A1*K1)*dY1
%    [-P2-A2*K2  I   O   O  O][dY3]    A2*dK2*Y2 + A2*db2
%    [O  -P3-A3*K3   I   O  O][dY4]  = A3*dK3*Y3 + A3*db3
%    [O   O  -P4-A4*K4   I  O][dY5]    A4*dK4*Y4 + A4*db4
%    [O   O   O  -P5-A5*K5  I][dY6]    A5*dK5*Y5 + A5*db5
%
%    The adjoint system
%    [I  -P2'-K2'*A2'           O           O          O][dY2]    [O  ]
%    [O          I   -P3'-K3'*A3'           O          O][dY3]    [O  ]
%    [O          O           I   -P4'-K4'*A4'          O][dY4]  = [O  ]
%    [O          O           O           I  -P5'-K5'*A5'][dY5]    [O  ]
%    [O          O           O           O             I][dY6]    [dYN]
