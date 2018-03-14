function [x,hist] = newtonReg(fun,reg,x0,param)
%[x,hist] = newtonReg(fun,reg,x0,param)

% initialize and unpack parameters
x = x0;
alpha = param.alpha;
maxIter = param.maxIter;
lsMaxIter = param.lsMaxIter;
slvTol = param.slvTol;
slvIter = param.slvIter;

[obj,dobj,H] = fun(x,param);
[R,dR,d2R] = reg(x,param);

F   = obj + alpha*R;
dF  = dobj + alpha*dR;
d2F = @(x) H(x) + alpha*d2R(x);

mu = 1;
norm0 = norm(dF);
rNorm = 1;
lsFailed = false;

% print header and init printer
fprintf('It.LS.sv   F          Obj        Reg        rNorm\n');
prnt = @(j,cnt,pcgIt,F,obj,R,rNorm) ...
    fprintf('%3d.%d.%02d   %3.2e   %3.2e   %3.2e   %3.2e\n', ...
    j, cnt, pcgIt, F, obj, R, rNorm);

% history
hidx = 1;
hist = zeros(5*maxIter, nargin(prnt)); %overestimate size and crop later

for j=1:maxIter
    
    % print status
    cnt = 0;
    prnt(j,cnt,0,F,obj,R,rNorm);
    hist(hidx,:) = [j,cnt,0,F,obj,R,rNorm];
    hidx = hidx + 1;
    
    % s = -dF;
    %[s,slv_flag,slv_relres,slv_iter] = pcg(d2F,-dF,cgTol,cgIter);
    %[s,slv_flag,slv_relres,slv_iter] = bicg(d2F,-dF,cgTol,cgIter);
    [s,slv_flag,slv_relres,slv_iter] = gmres(d2F,-dF,[],slvTol,slvIter);
    
    solveIter = slv_iter(2);
    if solveIter == 0 || s(:)'*dobj(:) > 0
        % zero solution provides sufficiently small residual, OR s is not a
        % descent direction; either way, default to negative gradient
        s = -dobj;
    end
    
    % Armijo line search
    cnt = 1;
    mu = 1;
    while true
        
        xtry = x + mu*s;
        
        [objtry, dobj, H  ] = fun(xtry, param);
        [Rtry,   dR,   d2R] = reg(xtry, param);
        
        Ftry = objtry + alpha*Rtry;
        dF   = dobj + alpha*dR;
        d2F  = @(x) H(x) + alpha*d2R(x);
        
        rNorm = norm(dF)/norm0;
        
        prnt(j,cnt,solveIter,Ftry,objtry,Rtry,rNorm);
        hist(hidx,:) = [j,cnt,solveIter,Ftry,objtry,Rtry,rNorm];
        hidx = hidx + 1;
        
        if Ftry < F
            break
        end
        
        mu = mu/2;
        cnt = cnt+1;
        if cnt > lsMaxIter
            lsFailed = true;
            warning('Armijo linesearch exceed max iters: %d', lsMaxIter);
            break
        end
        
    end
    
    if lsFailed
        break
    end
    
    if cnt == 1
        mu = min(2*mu,1);
    end
    
    x     = xtry;
    obj   = objtry;
    R     = Rtry;
    F     = Ftry;
    
end % for-loop

% crop hist to actual number of iterations
hist = hist(1:hidx-1,:);

end % steepestDescent
