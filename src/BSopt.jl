using Quiqbox
using Optim

## L-BFGS as the gradient-based parameter optimizer
lbfgs = function (f, gf, x0)
    om = LBFGS() # linesearch: Hager-Zhang
    d = Optim.OnceDifferentiable(f, x->gf(x)[begin], x0, inplace=false)
    options = Optim.Options(;allow_f_increases=false, Optim.default_options(om)...)
    state = Optim.initial_state(om, options, d, x0)
    function (x, _, _)
        state.x .= x
        Optim.update_state!(d, state, om)
        Optim.update_g!(d, state, om)
        Optim.update_h!(d, state, om)
        x .= state.x
    end
end

#= Geometry parameter of BeH2 =#
#=             Be             =#
#=             |              =#
#=          H-----H           =#
#= | => zBeLen (Bohr R)       =#
#= ----- => xH2Len (Bohr R)   =#
function optAObs(xH2Len, zBeLen, bsName; 
                 ΔEThreshold=1e-6, ΔgThreshold = 1e-4, maxStep = 800, 
                 method=:HFenergy)
    # e.g. of input arguments
    # zBeLen = 0.2
    # xH2Len = 0.5
    # bsName = "STO-3G" (suppot options: "STO-2G", "STO-3G", "STO-6G", 
                                       # "3-21G", "6-31G", 
                                       # "cc-pVDZ", "cc-pVTZ", "cc-pVQZ")

    nuc = ["Be", "H", "H"]
    pointsH2 = GridBox((1,0,0), xH2Len).point
    coordBe = [0., 0., zBeLen]
    nucCoords = [coordBe, coordOf.(pointsH2)...]

    bsHl = genBasisFunc(pointsH2[1], bsName, "H")
    bsHr = genBasisFunc.(bsHl, Ref(pointsH2[2]))
    bsBe = genBasisFunc(coordBe, bsName, "Be")
    bs = [bsBe..., bsHl..., bsHr...]

    parsAll = markParams!(bs, true)
    parsToOpt = vcat(getParams.(Ref(parsAll), (:α, :d))...)
    cfg = POconfig(;maxStep, method, threshold=(ΔEThreshold, ΔgThreshold), optimizer=lbfgs)
    optimizeParams!(parsToOpt, bs, nuc, nucCoords, cfg, infoLevel=2)

    bs
end