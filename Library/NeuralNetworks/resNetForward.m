function[Y,Yall,dA] = resNetForward(K,b,Y0,param)
%[Y,Yall] = resNetForward(K,b,P,Y0,param)
%

act = param.act;
P   = param.P;
N   = length(P);
Y   = Y0;

Yall{1} = Y0;
dA = {};
for i=1:N
    [Ai,dAi] = act(K{i}*Y + b{i});
    Y         = P{i}*Y + Ai;
    Yall{i+1} = Y;
    dA{i}     = dAi;
end
