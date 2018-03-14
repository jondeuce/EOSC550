function[x] = newtonOpt(fun,x0,param)
%[x,hist] = newtonOpt(fun,x0,param)
%

x = x0;
[obj,dobj,H] = fun(x,param);
mu = 1; 

for j=1:param.maxIter
    
    fprintf('%3d.0   %3.2e   %3.2e\n',j,obj,norm(dobj))
    [s,~] = pcg(H,-dobj,1e-1,100);
    
    % test s is a descent direction
    if s(:)'*dobj(:) > 0
        s = -dobj;
    end
    % Armijo line search
    cnt = 1;
    while 1
        xtry = x + mu*s;
        [objtry,dobj,H] = fun(xtry,param);
        fprintf('%3d.%d   %3.2e   %3.2e\n',j,cnt,objtry,norm(dobj))

        if objtry < obj
            break
        end
        mu = mu/2;
        cnt = cnt+1;
        if cnt > 10
            error('Line search break');
        end
    end
    if cnt == 1
        mu = min(mu*1.5,1);
    end
    x = xtry;
    obj = objtry;
    
    plots=0;
    if plots
        w = reshape(x,3,2);
        figure(1)
        t = linspace(-3,4,129);
        q = -w(1,1)/w(2,1)*t -w(3,1)/w(2,1);
        hold on
        plot(t,q,'k','linewidth',3)
        hold off
        pause(0.3);
    end
end