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

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="optimizers.jl (c) Ozan Arkan Can and Deniz Yuret, 2016. Demonstration of different sgd based optimization methods using LeNet."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
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
    dtrn = minibatch(xtrn, ytrn, o[:batchsize], xtype=atype)
    dtst = minibatch(xtst, ytst, o[:batchsize], xtype=atype)
    w = weights(atype=atype)
    prms = params(w, o)

    # log = Any[]
    # report(epoch)=push!(log, (:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    # report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    report(0)
    iters = o[:iters]
    @time for epoch=1:o[:epochs]
        train(w, prms, dtrn; epochs=1, iters=iters)
        report(epoch)
        (iters -= length(dtrn)) <= 0 && break
    end

    # for t in log; println(t); end
    return w
end

function train(w, prms, data; epochs=10, iters=6000)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            update!(w, g, prms)
            if (iters -= 1) <= 0
                return w
            end
        end
    end
    return w
end

# Choose loss function, etc. for: kobler
const TT = 3
const CC = 3
weights(;kwargs...) = weights_kobler(;C=CC,kwargs...)
predict(w,x) = predict_kobler(w,x,TT,CC)
loss(w,x,y) = loss_kobler(w,x,y,TT,CC)
lossgradient = grad(loss)
accuracy(w,data,predict) = sum(map(d->loss(w,d[1],d[2]),data))

# Choose loss function, etc. for: simple convolution
# predict = predict_simpleconv
# weights = weights_simpleconv
# lossgradient = grad(loss)
# accuracy = accuracy_L2mean

#Creates necessary parameters for each weight to use in the optimization
function params(ws, o)
	prms = Any[]

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
		push!(prms, prm)
	end

	return prms
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia optimizers.jl --epochs 10
# julia> Optim.main("--epochs 10")
PROGRAM_FILE == "chi2d_optimizers.jl" && main(ARGS)

end # module
