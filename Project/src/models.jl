# ---------------------------------------------------------------------------- #
# Generic utilities
# ---------------------------------------------------------------------------- #
loss_L2mean(w,x,y) = mean(abs2,y.-predict(w,x))
accuracy_L2mean(w,data,predict) = 1.0 - sum(map(d->loss(w,d[1],d[2]),data))

# Cast elements of `w` to `atype`, but keep `w` itself an Array{any}
check_weights(atype,w) = Array{Any}(map(ww->convert(atype,ww), w))

# mean, etc. not implemented for 4D arrays with UnitRange's; do with reshaping
kernel_mat4(x) = reshape(x, prod(size(x)[1:2]), prod(size(x)[3:4]))
kernel_mean4(x) = (y = kernel_mat4(x); reshape(mean(y,1),1,1,size(x)[3:4]...))
kernel_norm4(x) = (y = kernel_mat4(x); reshape(sqrt.(sum(abs2,y,1)),1,1,size(x)[3:4]...))

function normalize_kernel4(x,ϵ=convert(eltype(x),5)*eps(eltype(x)))
    # Normalize 4D kernel x along first two dimensions, being careful not to
    # divide by zero
    x = x.-kernel_mean4(x)
    return x./max.(kernel_norm4(x),ϵ)
end

# ---------------------------------------------------------------------------- #
# Simple convolutional kernel with bias
# ---------------------------------------------------------------------------- #
params_simpleconv(;kwargs...) = merge(default_params_simpleconv(),Dict(kwargs))
default_params_simpleconv() = Dict{Symbol,Any}(
    :seed=>-1,:atype=>KnetArray,:sk=>5,:bsize=>(1,1))

function predict_simpleconv(w,x,p)
    pad = div(p[:sk],2)
    return conv4(w[1],x,padding=pad) .+ w[2]
end

function weights_simpleconv(p)
    p[:seed] > 0 && srand(p[:seed])
    sk, bsize = p[:sk], p[:bsize]

    w = Array{Any}(2)
    w[1] = xavier(Float32,sk,sk,1,1)
    w[2] = ones(Float32,bsize)
    return check_weights(p[:atype],w)
end

loss_simpleconv(w,x,y,p) = sum(abs2,predict_simpleconv(w,x,p).-y)/size(y,4)

# ---------------------------------------------------------------------------- #
# Convolutional neural network with fully connected layers (LeNet model)
# ---------------------------------------------------------------------------- #
params_lenet(;kwargs...) = merge(default_params_lenet(),Dict(kwargs))
default_params_lenet() = Dict{Symbol,Any}(
    :seed=>-1, :atype=>KnetArray, :imdims=>(50,60), :proj=>true,
    :kernsize=>5, :nconv=>2, :nc=>[5,5], :nfull=>2, :nf=>[500,500])

function predict_lenet(w,x,p) # LeNet model
    nconv, nfull = p[:nconv], p[:nfull]

    xsiz = size(x)
    for i in 1:2:2nconv
        x = pool(relu.(conv4(w[i],x,padding=2) .+ w[i+1]),window=1)
    end
    for i in 2nconv+(1:2:2nfull)
        x = relu.(w[i]*mat(x) .+ w[i+1])
    end
    x = w[end-1]*x .+ w[end]

    return reshape(x,xsiz)
end

function weights_lenet(p)
    ks, imdims, nconv, nfull, nc, nf =
    p[:kernsize], p[:imdims], p[:nconv], p[:nfull], p[:nc], p[:nf]

    p[:seed] > 0 && srand(p[:seed])
    N = prod(imdims)
    Nc = N*nc[end]

    w = Array{Any}(2nconv+2nfull+2)
    nc = vcat(1,nc)
    nf = vcat(Nc,nf)
    for (i,ix) in zip(1:2:2nconv,1:nconv)
        w[i]   = 0.1randn(ks,ks,nc[ix],nc[ix+1])
        w[i+1] = zeros(1,1,nc[ix+1],1)
    end
    for (i,ix) in zip(2nconv+(1:2:2nfull),1:nfull)
        w[i]   = 0.1randn(nf[ix+1],nf[ix])
        w[i+1] = zeros(nf[ix+1],1)
    end
    w[end-1] = 0.1randn(N,nf[end])
    w[end]   = zeros(N,1)

    # Equivalent and more clear construction for nconv = 2, nfull = 2
    # w = [ 0.1*randn(ks,ks,     1, nc[1]), zeros(1,1, nc[1], 1),
    #       0.1*randn(ks,ks, nc[1], nc[2]), zeros(1,1, nc[2], 1),
    #       0.1*randn(nf[1], Nc),    zeros(nf[1], 1),
    #       0.1*randn(nf[2], nf[1]), zeros(nf[2], 1),
    #       0.1*randn(    N, nf[2]), zeros(    N, 1) ]

    # Normalize weights
    p[:proj] && project_lenet!(w,p)

    return check_weights(p[:atype],w)
end

function project_lenet!(w,p)
    nconv = p[:nconv]
    for i in 1:2:2nconv
        w[i] = normalize_kernel4(w[i])
    end
end

loss_lenet(w,x,y,p) = sum(abs2,predict_lenet(w,x,p).-y)/size(y,4)

# ---------------------------------------------------------------------------- #
# Kobler's varational network
# ---------------------------------------------------------------------------- #
params_kobler(;kwargs...) = merge(default_params_kobler(),Dict(kwargs))
default_params_kobler() = Dict{Symbol,Any}(:seed=>-1,:atype=>KnetArray,
    :C=>1,:T=>1,:Nr=>3,:Nd=>3,:Nw=>7,:sr=>5,:sd=>5,:cycle=>"cyclic",:proj=>true)

function influence_function(x,p,μ,σ,wc,I)
    # The influence function ϕ′ is learned, and is given as a weighted sum of
    # radial basis functions of the form
    #   exp(-(x-μ)^2/(2σ^2)).
    # The corresponding potential function is given by
    #   sqrt(π*σ^2/4)*(1+erf((x-μ)/(σ*sqrt(2))))
    κ = 1./(2σ[1].^2)
    w(I,j) = ifelse(length(I)>1, reshape(wc[I,j],1,1,length(I),1), wc[I,j])

    # Unroll first loop so that ϕ′ is initialized properly (initializing with
    # zero, etc. are sketchy for KnetArray...)
    μj  = μ[1]
    wij = w(I,1)
    tmp = x.-μj # separate out squaring as literal_pow is broken for KnetArray
    ϕ′  = exp.(-κ.*tmp.*tmp).*wij
    for j in 2:p[:Nw]
        μj  = μ[j]
        wij = w(I,j)
        tmp = x.-μj
        ϕ′ += exp.(-κ.*tmp.*tmp).*wij
    end

    return ϕ′
end
# influence_function(x,p,μ,σ,w,I) = relu.(x)

# Vectorized ("tensorized") - and much faster - version of grad_kobler which
# works for KnetArray types on the GPU
function grad_kobler_vec(w,x,x0,p,c)
    C, T, Nr, Nd, sr, sd = p[:C], p[:T], p[:Nr], p[:Nd], p[:sr], p[:sd]
    μ, σ, wr, wd, Kr, Kd = w[1], w[2], w[2+c], w[2+C+c], w[2+2C+c], w[2+3C+c]

    pr = div(sr,2) # pad size
    pd = div(sd,2) # pad size
    Ir, Id = 1:Nr, 1:Nd

    df = deconv4(Kr, influence_function( conv4(Kr, x, padding=pr), p,μ,σ,wr,Ir), padding=pr) +
        deconv4(Kd, influence_function( conv4(Kd, x.-x0, padding=pr), p,μ,σ,wd,Id), padding=pd)

    return df
end

# Loop version for testing; only works for Array types, not KnetArrays
function grad_kobler(w,x,x0,p,c)
    C, T, Nr, Nd, sr, sd = p[:C], p[:T], p[:Nr], p[:Nd], p[:sr], p[:sd]
    μ, σ, wr, wd, Kr, Kd = w[1], w[2], w[2+c], w[2+C+c], w[2+2C+c], w[2+3C+c]

    df = zero(x)
    pr = div(sr,2) # pad size
    for i in 1:Nr
        kr = reshape(Kr[:,:,1,i],sr,sr,1,1) # single conv kernel
        df += deconv4( kr, influence_function( conv4(kr, x, padding=pr), p,μ,σ,wr,i ), padding=pr )
    end

    pd = div(sd,2) # pad size
    for i in 1:Nd
        kd = reshape(Kd[:,:,1,i],sd,sd,1,1) # single conv kernel
        df += deconv4( kd, influence_function( conv4(kd, x.-x0, padding=pd), p,μ,σ,wd,i ), padding=pd )
    end

    return df
end

# prox_kobler(w,y,y0,p,c) = (y -= grad_kobler(w,y,y0,p,c))
prox_kobler(w,y,y0,p,c) = (y -= grad_kobler_vec(w,y,y0,p,c))

function cycle_kobler(p)
    C, T = p[:C], p[:T]

    if p[:cycle] == "cyclic"
        c = t->mod1(t,C) # default to cyclic
    elseif p[:cycle] == "random"
        c = t->rand(1:C)
    else
        error("unknown cycle pattern $(p[:cycle]) for c(t)")
    end

    return c
end

function predict_kobler(w,x,p)
    y = copy(x)
    c = cycle_kobler(p)
    for t = 1:p[:T]
        y = prox_kobler(w,y,x,p,c(t))
    end
    return y
end

function project_kobler!(w,p)
    C = p[:C]
    for i in 2+2C+(1:2C)
        w[i] = normalize_kernel4(w[i])
    end
end

function weights_kobler(p=default_params_kobler())
    # w[1]  = μⱼ          [Nw×1]:  vector of potential function means
    # w[2]  = σ            [1×1]:  potential functions standard deviation
    # w[I1] = wrᶜij      [Nr×Nw]:  `C` potential function weights for prior term; I1 = 2+(1:C)
    # w[I2] = wdᶜij      [Nd×Nw]:  `C` potential function weights for data fidelity term; I2 = 2+C+(1:C)
    # w[I3] = Krᶜi  [sr×sr×1×Nr]:  `C` kernels for prior term; I3 = 2+2C+(1:C)
    # w[I4] = Kdᶜi  [sd×sd×1×Nd]:  `C` kernels for data fidelity term; I4 = 2+3C+(1:C)

    T, C, Nr, Nd, Nw, sr, sd = p[:T], p[:C], p[:Nr], p[:Nd], p[:Nw], p[:sr], p[:sd]
    p[:seed] > 0 && srand(p[:seed])

    phi_init = (c,args...) -> xavier(args...)
    if T < C && p[:cycle] == "cyclic"
        # Kernels with c > T will never be reached; initialize them to zero so
        # they do not affect the minimization
        phi_init = (c,args...) -> ifelse(c > T, zeros(args...), xavier(args...))
    end

    w = Array{Any}(2+4C)

    w[1] = xavier(Float32,Nw,1) # μⱼ
    w[2] = rand(Float32,1,1) # σ
    for c in 1:C
        # Normalize initial wrᶜij/wdᶜij by the number of total units T
        w[2+0C+c] = phi_init(c,Float32,Nr,Nw)/T # wrᶜij
        w[2+1C+c] = phi_init(c,Float32,Nd,Nw)/T # wdᶜij
        w[2+2C+c] = phi_init(c,Float32,sr,sr,1,Nr) # Krᶜi
        w[2+3C+c] = phi_init(c,Float32,sd,sd,1,Nd) # Kdᶜi
    end

    # Normalize weights
    p[:proj] && project_kobler!(w,p)

    return check_weights(p[:atype],w)
end

function loss_kobler(w,x,y,p)
    Ns = size(y,4)
    return sum(abs2,predict_kobler(w,x,p).-y)/Ns
end
