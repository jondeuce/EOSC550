"""
This utility loads the chi2d dataset.

```
# Usage:
include("chi2d.jl")
xtrn, ytrn, xtst, ytst = chi2d()
# xtrn: M×N×1×Ntrn Array{Float32,4}
# ytrn: M×N×1×Ntrn Array{Float32,4}
# xtst: M×N×1×Ntst Array{Float32,4}
# ytst: M×N×1×Ntst Array{Float32,4}
```

"""
function chi2d(;T=Float32,m=50,n=60,Ntrn=5,Ntst=2)
    # Initialize model
    fpi = T(pi)
    xp, yp = linspace(-fpi,fpi,m), linspace(-fpi,fpi,n)'

    _chi2d_ytrn = zeros(T,m,n,1,Ntrn)
    _chi2d_ytst = zeros(T,m,n,1,Ntst)

    idx = 1
    W = 1:Ntrn
    for (ωx,ωy) in Iterators.product(W,W)
        _chi2d_ytrn[:,:,1,idx] = sin.(ωx*xp) .* cos.(ωy*yp);
        idx += 1
        idx > Ntrn && break
    end

    idx = 1
    W = Ntrn+1:Ntrn+Ntst
    for (ωx,ωy) in Iterators.product(W,W)
        _chi2d_ytst[:,:,1,idx] = sin.(ωx*xp) .* cos.(ωy*yp);
        idx += 1
        idx > Ntst && break
    end

    _chi2d_xtst = _chi2d_ytst + 0.3f0*randn(T,size(_chi2d_ytst));
    _chi2d_xtrn = _chi2d_ytrn + 0.3f0*randn(T,size(_chi2d_ytrn));

    return _chi2d_xtrn,_chi2d_ytrn,_chi2d_xtst,_chi2d_ytst
end

nothing
