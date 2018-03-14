function[R,dR,d2R] = genTikhonov(W,param)
%[R,dR,d2R] = genTikhonov(W,param)

nc = param.nc;
L  = param.L;
h  = param.h;
L  = blkdiag(L,1);

isVecW = (size(W,2) == 1);
if isVecW; W = reshape(W,[],nc); end

LW = L*W;

R   = (0.5 * prod(h)) * (LW(:)'*LW(:));
dR  = prod(h) * (L'*LW);
d2R = prod(h) * (L'*L);

mat = @(v) reshape(v,[],nc);
vec = @(V) V(:);

if isVecW
    dR = dR(:);
    d2R = @(v) vec(d2R * mat(v));
else
    d2R = @(V) d2R * V;
end

end