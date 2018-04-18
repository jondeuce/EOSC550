for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

This example demonstrates the usage of stochastic gradient descent(sgd) based
optimization methods. We train LeNet model on MNIST dataset similar to `lenet.jl`.

You can run the demo using `julia optimizers.jl`.  Use `julia optimizers.jl
--help` for a list of options. By default the [LeNet](http://yann.lecun.com/exdb/lenet)
convolutional neural network model will be trained using sgd for 10 epochs.
At the end of the training accuracy for the training and test sets for each epoch will be printed
and optimized parameters will be returned.

"""
module Optimizers
using Knet,ArgParse

include("data/chi2d.jl")
include("chi2d_objectives.jl")

import Base.zero
zero{T,N}(a::KnetArray{T,N}) = fill!(similar(a),zero(T))

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="optimizers.jl (c) Ozan Arkan Can and Deniz Yuret, 2016. Demonstration of different sgd based optimization methods using LeNet."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1; help="minibatch size")
        ("--lr"; arg_type=Float64; default=1e-7; help="learning rate")
        ("--eps"; arg_type=Float64; default=1e-6; help="epsilon parameter used in adam, adagrad, adadelta")
        ("--gamma"; arg_type=Float64; default=0.95; help="gamma parameter used in momentum and nesterov")
        ("--rho"; arg_type=Float64; default=0.9; help="rho parameter used in adadelta and rmsprop")
        ("--beta1"; arg_type=Float64; default=0.9; help="beta1 parameter used in adam")
        ("--beta2"; arg_type=Float64; default=0.95; help="beta2 parameter used in adam")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--iters"; arg_type=Int; default=1000; help="number of updates for training")
        ("--optim"; default="Sgd"; help="optimization method (Sgd, Momentum, Nesterov, Adagrad, Adadelta, Rmsprop, Adam)")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    if atype <: Array; warn("CPU conv4 support is experimental and very slow."); end

    # xtrn,ytrn,xtst,ytst = Main.mnist()
    xtrn,ytrn,xtst,ytst = chi2d()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize], xtype=atype, ytype = atype)
    dtst = minibatch(xtst, ytst, o[:batchsize], xtype=atype, ytype = atype)
    p = params(atype=atype, seed=o[:seed])
    w = weights(p)
    opts = optimopts(w, o)

    # log = Any[]
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict,p=p),:tst,accuracy(w,dtst,predict,p=p)))
    report(0)
    iters = o[:iters]
    @time for epoch=1:o[:epochs]
        train(w, dtrn, p, opts; epochs=1, iters=iters)
        report(epoch)
        (iters -= length(dtrn)) <= 0 && break
    end

    # for t in log; println(t); end
    return w
end

function __mock_main__()
    srand(42)
    atype = KnetArray{Float32}
    # atype = Array{Float32}
    batchsize = 1
    iters = 1000
    epochs = 3

    xtrn,ytrn,xtst,ytst = chi2d()
    dtrn = minibatch(xtrn, ytrn, batchsize, xtype=atype, ytype = atype)
    dtst = minibatch(xtst, ytst, batchsize, xtype=atype, ytype = atype)

    p = params(atype=atype)
    w = weights(p)

    o = Dict(:lr => 1e-6, :optim => "Sgd")
    opts = optimopts(w, o)

    for (x,y) in dtrn
        Σ = loss_kobler(w,x,y,p)
    end

    # log = Any[]
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict,p=p),:tst,accuracy(w,dtst,predict,p=p)))
    report(0)
    @time for epoch in 1:epochs
        for e=1:epoch
            for (x,y) in dtrn
                g = lossgradient(w, x, y; p=p)
                update!(w, g, opts)
                if (iters -= 1) <= 0
                    return w
                end
            end
        end
        report(epoch)
        (iters -= length(dtrn)) <= 0 && break
    end
end

function train(w, data, p, opts; epochs=10, iters=6000)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y; p=p)
            update!(w, g, opts)
            if (iters -= 1) <= 0
                return w
            end
        end
    end
    return w
end

# Choose loss function, etc. for: kobler
const TT = 10
const CC = 10
params(;kwargs...) = params_kobler(;C=CC,T=TT,kwargs...)
weights(p) = weights_kobler(p)
predict(w,x;p=default_params_kobler()) = predict_kobler(w,x,p)
loss(w,x,y;p=default_params_kobler()) = loss_kobler(w,x,y,p)
lossgradient = grad(loss)
accuracy(w,data,predict;p=default_params_kobler()) = sum(map(d->loss(w,d[1],d[2],p=p),data))

# Choose loss function, etc. for: simple convolution
# predict = predict_simpleconv
# weights = weights_simpleconv
# lossgradient = grad(loss)
# accuracy = accuracy_L2mean

#Creates necessary parameters for each weight to use in the optimization
function optimopts(ws, o)
	opts = Any[]

	for i=1:length(ws)
		w = ws[i]
		if o[:optim] == "Sgd"
			prm = Sgd(;lr=o[:lr])
		elseif o[:optim] == "Momentum"
			prm = Momentum(lr=o[:lr], gamma=o[:gamma])
		elseif o[:optim] == "Nesterov"
			prm = Nesterov(lr=o[:lr], gamma=o[:gamma])
		elseif o[:optim] == "Adagrad"
			prm = Adagrad(lr=o[:lr], eps=o[:eps])
		elseif o[:optim] == "Adadelta"
			prm = Adadelta(lr=o[:lr], rho=o[:rho], eps=o[:eps])
		elseif o[:optim] == "Rmsprop"
			prm = Rmsprop(lr=o[:lr], rho=o[:rho], eps=o[:eps])
		elseif o[:optim] == "Adam"
			prm = Adam(lr=o[:lr], beta1=o[:beta1], beta2=o[:beta2], eps=o[:eps])
		else
			error("Unknown optimization method!")
		end
		push!(opts, prm)
	end

	return opts
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia optimizers.jl --epochs 10
# julia> Optim.main("--epochs 10")
PROGRAM_FILE == "chi2d_optimizers.jl" && main(ARGS)

end # module
