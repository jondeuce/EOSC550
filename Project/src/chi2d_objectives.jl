
# potential_function(x,μ,σ) = sqrt(π*σ^2/4)*(1+erf((x-μ)/(σ*sqrt(2))))
# radial_basis(x,μ,σ) = exp(-(x-μ)^2/(2σ^2))
function influence_function(x,μ,σ,w,i=1,c=1)
    Nw = length(μ) # wᶜ is [N×Nw×C]
    κ = 1/(2σ[1]^2)
    # κ  = reshape(1./(2.*σ.^2),1,1,1,1)
    # κ  = 1./(2.*σ.^2)
    # ϕ′ = zeros(typeof(κ), size(x))
    # ϕ′ = zero(x)
    ϕ′ = zero(eltype(x))*copy(x)

    for j in 1:Nw
        wij = w[i,j,c]
        μj = μ[j]
        # assert(isa(κ,Number))
        # assert(isa(μj,Number))
        # assert(isa(wij,Number))
        # print("\n\n  wij: ",size(wij),"\n\n  μj: ",size(μj),"\n\n  κ: ",size(κ),"\n\n")
        # ϕ′ .+= exp.(-κ.*(x.-μj).^2).*wij
        ϕ′ += exp.(-κ*(x-μj).^2)*wij
        # for idx in 1:length(x)
        #     # tmp1 = x[idx].-μj
        #     # tmp2 = tmp1.*tmp1
        #     # tmp3 = κ.*tmp2
        #     # tmp4 = exp.(-tmp3)
        #     # tmp5 = wij.*tmp4
        #     # if typeof(tmp5) ≠ Float32 || typeof(ϕ′) ≠ Float32
        #     #     print("\n\n    ϕ′: ", typeof(ϕ′))
        #     #     print("\n\n  tmp5: ", typeof(tmp5), "\n\n")
        #     # end
        #     # ϕ′[idx] += tmp5
        #     ϕ′[idx] += (wij * exp(-κ*(x[idx]-μj)^2))
        # end
    end
    return ϕ′
end
# influence_function(x,μ,σ,w,i=1,c=1) = relu.(x)

function grad_kobler(x,x0,w,c=1)
    μ, σ, wr, wd, Kr, Kd = w[1], w[2], w[3], w[4], w[5], w[6]
    sr, Nr, sd, Nd = size(Kr,1), size(Kr,3), size(Kd,1), size(Kd,3)

    df = zero(x)
    for i in 1:Nr
        # print("i = ",i," c = ",c,"\n")
        # print(Kr,"\n")
        # kr = Kr[:,:,i,c]
        # print(kr,"\n")
        # kr = reshape(kr,sr,sr,1,1) # single conv kernel
        kr = reshape(Kr[:,:,i,c],sr,sr,1,1) # single conv kernel
        pd = div(sr,2) # pad size
        df += deconv4( kr, influence_function( conv4(kr, x, padding=pd),
                μ,σ,wr,i,c ), padding=pd )
    end

    for i in 1:Nd
        kd = reshape(Kd[:,:,i,c],sd,sd,1,1) # single conv kernel
        pd = div(sd,2) # pad size
        df += deconv4( kd, influence_function( conv4(kd, x.-x0, padding=pd),
                μ,σ,wd,i,c ), padding=pd )
    end
    return df
end

proj_kobler(y,y0,w,c=1) = (y -= grad_kobler(y,y0,w,c))

function predict_kobler(w,x,T=1,C=1)
    y0 = y = copy(x)
    c(t) = mod1(t,C) #cyclic
    for t = 1:T
        y = proj_kobler(y,y0,w,c(t))
    end
    return y
end

function loss_kobler(w,x,y,T=1,C=1)
    Ns = size(y,4)
    return sum(abs2,predict_kobler(w,x,T,C).-y)/Ns
end

function weights_kobler(;C=1,Nr=3,Nd=3,Nw=7,sr=5,sd=5,atype=Array{Float32})
    # w[1] = μⱼ         [Nw×1]:  vector of potential function means
    # w[2] = σ           [1×1]:  potential functions standard deviation
    # w[3] = wrᶜij   [Nr×Nw×C]:  potential function weights for prior term
    # w[4] = wdᶜij   [Nd×Nw×C]:  potential function weights for data fidelity term
    # w[5] = Krᶜi [sr×sr×Nr×C]:  kernels for prior term
    # w[6] = Kdᶜi [sd×sd×Nd×C]:  kernels for data fidelity term

    w = Array{Any}(6)
    # w[1] = collect(linspace(-1f0,1f0,Nw)) # μⱼ
    # w[2] = ones(Float32,1,1) # σ
    # w[3] = ones(Float32,Nr,Nw,C) # wrᶜij
    # w[4] = ones(Float32,Nd,Nw,C) # wdᶜij
    # w[5] = xavier(Float32,sr,sr,Nr,C) # Krᶜi
    # w[6] = xavier(Float32,sd,sd,Nd,C) # Kdᶜi

    w[1] = xavier(Float32,Nw,1) # μⱼ
    w[2] = xavier(Float32,1,1) # σ
    w[3] = xavier(Float32,Nr,Nw,C) # wrᶜij
    w[4] = xavier(Float32,Nd,Nw,C) # wdᶜij
    w[5] = xavier(Float32,sr,sr,Nr,C) # Krᶜi
    w[6] = xavier(Float32,sd,sd,Nd,C) # Kdᶜi

    return map(a->convert(atype,a), w)
    # return w
    # return Array{Any}(map(a->convert(atype,a), w))
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
