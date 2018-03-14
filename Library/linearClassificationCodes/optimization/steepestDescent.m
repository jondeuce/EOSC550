function [x,hist] = steepestDescent(fun,reg,x0,param)
%[x,hist] = steepestDescent(fun,reg,x0,param)

% initialize and unpack parameters
x = x0;
alpha = param.alpha;
maxIter = param.maxIter;
lsMaxIter = param.lsMaxIter;

[obj,dobj] = fun(x,param);
[R,dR,d2R] = reg(x,param);

F  = obj + alpha*R;
dF = dobj + alpha*dR;

norm0 = norm(dF);
rNorm = 1;
cnt = 0;
mu = 1;
lsFailed = false;

% print header and init printer
fprintf('It.LS   F          Obj        Reg        rNorm\n');
prnt = @(j,cnt,F,obj,R,rNorm) ...
    fprintf('%3d.%d   %3.2e   %3.2e   %3.2e   %3.2e\n', ...
    j, cnt, F, obj, R, rNorm);

% history
hidx = 1;
hist = zeros(5*maxIter, nargin(prnt)); %overestimate size and crop later

for j=1:maxIter
    
    prnt(j,cnt,F,obj,R,rNorm);
    hist(hidx,:) = [j,cnt,F,obj,R,rNorm];
    hidx = hidx + 1;
    
    s = -dF;
    
    % Armijo line search
    cnt = 1;
    while true
        
        xtry = x + mu*s;
        
        [objtry,dobj] = fun(xtry,param);
        [Rtry,dR]     = reg(xtry,param);
        
        Ftry  = objtry + alpha*Rtry;
        dF    = dobj + alpha*dR;
        rNorm = norm(dF)/norm0;
        
        if Ftry < F
            prnt(j,cnt,Ftry,obj,R,rNorm);
            hist(hidx,:) = [j,cnt,F,obj,R,rNorm];
            hidx = hidx + 1;
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
        mu = mu*1.5;
    end
    
    x   = xtry;
    obj = objtry;
    R   = Rtry;
    F   = Ftry;
    
end % for-loop

% crop hist to actual number of iterations
hist = hist(1:hidx-1,:);

end % steepestDescent
