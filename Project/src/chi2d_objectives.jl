function params_kobler(;kwargs...)
    p = default_params_kobler()
    for (k,v) in kwargs
        p[k] = v
    end
    return p
end
default_params_kobler() = Dict(:C=>1,:T=>1,:Nr=>3,:Nd=>3,:Nw=>7,:sr=>5,:sd=>5,
    :atype=>KnetArray{Float32}, :seed=>-1, :cycle=>"cyclic")

# potential_function(x,μ,σ) = sqrt(π*σ^2/4)*(1+erf((x-μ)/(σ*sqrt(2))))
# radial_basis(x,μ,σ) = exp(-(x-μ)^2/(2σ^2))
function influence_function(x,μ,σ,wc,I)
    Nw = length(μ) # wᶜ is [N×Nw×C]
    κ = 1/(2σ[1]^2)
    ϕ′ = zero(eltype(x))*copy(x) # need to use this form so recording works...

    for j in 1:Nw
        μj = μ[j]
        wij = wc[I,j]
        length(I) > 1 && (wij = reshape(wc[I,j],1,1,length(I),1))

        # Equivalent to: ϕ′ += exp.(-κ*(x-μj).^2).*wij, but literal_pow is broken for KnetArray...
        tmp = x.-μj
        ϕ′ += exp.(-κ.*tmp.*tmp).*wij
    end
    return ϕ′
end
# influence_function(x,μ,σ,w,i) = relu.(x)

function grad_kobler_vec(w,x,x0,p,c)
    C, T, Nr, Nd, Nw, sr, sd = p[:C], p[:T], p[:Nr], p[:Nd], p[:Nw], p[:sr], p[:sd]
    μ, σ, wr, wd, Kr, Kd = w[1], w[2], w[2+c], w[2+C+c], w[2+2C+c], w[2+3C+c]

    pr = div(sr,2) # pad size
    pd = div(sd,2) # pad size
    Ir, Id = 1:Nr, 1:Nd

    df = deconv4(Kr, influence_function( conv4(Kr, x, padding=pr), μ,σ,wr,Ir), padding=pr) +
        deconv4(Kd, influence_function( conv4(Kd, x.-x0, padding=pr), μ,σ,wd,Id), padding=pd)

    return df
end
function grad_kobler(w,x,x0,p,c)
    C, T, Nr, Nd, Nw, sr, sd, atype = p[:C], p[:T], p[:Nr], p[:Nd], p[:Nw], p[:sr], p[:sd], p[:atype]
    μ, σ, wr, wd, Kr, Kd = w[1], w[2], w[2+c], w[2+C+c], w[2+2C+c], w[2+3C+c]

    df = zero(x)
    pr = div(sr,2) # pad size
    for i in 1:Nr
        kr = reshape(Kr[:,:,1,i],sr,sr,1,1) # single conv kernel
        df += deconv4( kr, influence_function( conv4(kr, x, padding=pr), μ,σ,wr,i ), padding=pr )
    end

    pd = div(sd,2) # pad size
    for i in 1:Nd
        kd = reshape(Kd[:,:,1,i],sd,sd,1,1) # single conv kernel
        df += deconv4( kd, influence_function( conv4(kd, x.-x0, padding=pd), μ,σ,wd,i ), padding=pd )
    end

    return df
end

# proj_kobler(w,y,y0,p,c) = (y -= grad_kobler(w,y,y0,p,c))
proj_kobler(w,y,y0,p,c) = (y -= grad_kobler_vec(w,y,y0,p,c))

function predict_kobler(w,x,p)
    C, T = p[:C], p[:T]
    y0 = y = copy(x)

    if p[:cycle] == "random"
        c(t) = rand(1:C)
    else
        # default to cyclic
        c(t) = mod1(t,C)
    end

    for t = 1:T
        y = proj_kobler(w,y,y0,p,c(t))
    end
    return y
end

function loss_kobler(w,x,y,p)
    Ns = size(y,4)
    return sum(abs2,predict_kobler(w,x,p).-y)/Ns
end

function weights_kobler(p=default_params_kobler())
    # w[1]  = μⱼ          [Nw×1]:  vector of potential function means
    # w[2]  = σ            [1×1]:  potential functions standard deviation
    # w[I1] = wrᶜij      [Nr×Nw]:  `C` potential function weights for prior term; I1 = 2+(1:C)
    # w[I2] = wdᶜij      [Nd×Nw]:  `C` potential function weights for data fidelity term; I2 = 2+C+(1:C)
    # w[I3] = Krᶜi  [sr×sr×1×Nr]:  `C` kernels for prior term; I3 = 2+2C+(1:C)
    # w[I4] = Kdᶜi  [sd×sd×1×Nd]:  `C` kernels for data fidelity term; I4 = 2+3C+(1:C)

    p[:seed] > 0 && srand(p[:seed])

    C, Nr, Nd, Nw, sr, sd = p[:C], p[:Nr], p[:Nd], p[:Nw], p[:sr], p[:sd]
    w = Array{Any}(2+4C)

    w[1] = xavier(Float32,Nw,1) # μⱼ
    w[2] = xavier(Float32,1,1) # σ
    for i in 2+(1:C)
        w[i] = xavier(Float32,Nr,Nw) # wrᶜij
    end
    for i in 2+C+(1:C)
        w[i] = xavier(Float32,Nd,Nw) # wdᶜij
    end
    for i in 2+2C+(1:C)
        w[i] = xavier(Float32,sr,sr,1,Nr) # Krᶜi
    end
    for i in 2+3C+(1:C)
        w[i] = xavier(Float32,sd,sd,1,Nd) # Kdᶜi
    end

    # Cast elements of `w` to `atype`, but keep `w` itself an Array{any}:
    w = Array{Any}(map(a->convert(p[:atype],a), w))

    return w
end

function predict_simpleconv(w,x)
    pad = div(size(w[1],1),2)
    return conv4(w[1],x,padding=pad) .+ w[2]
end

function weights_simpleconv(;atype=KnetArray{Float32})
    w = Array{Any}(2)
    w[1] = xavier(Float32,5,5,1,1)
    w[2] = ones(Float32,1,1)

    return map(a->convert(atype,a), w)
end

loss_L2mean(w,x,y) = mean(abs2,y-predict(w,x))
accuracy_L2mean(w,data,predict) = 1.0 - sum(map(d->loss(w,d[1],d[2]),data))
