function diag_ABt(A, B)
    vec(sum(A .* B; dims=2))
end

struct IncreasingRate
    α::Float64 # Maximum learning rate
    γ::Float64 # Convergence rate to the maximum
    state
end

IncreasingRate(α=1.0, γ=1e-8) = IncreasingRate(α, γ, IdDict())

function Optimise.apply!(opt::IncreasingRate, x, g)
    t = get!(()->0, opt.state, x)
    opt.state[x] += 1
    return g .* opt.α * (1 - exp(-opt.γ * t))
end

struct LogLinearIncreasingRate
    γmax::Float64 # Maximum learning rate
    γmin::Float64 # Convergence rate to the maximum
    K::Int
    state
end

LogLinearIncreasingRate(γmax=1.0, γmin=1e-6, K=100) = LogLinearIncreasingRate(γmax, γmin, K, IdDict())

function Optimise.apply!(opt::LogLinearIncreasingRate, x, g)
    t = get!(()->1, opt.state, x)
    γ = 10^(((opt.K - min(t, opt.K)) * log10(opt.γmin) + min(t, opt.K) * log10(opt.γmax))/opt.K)
    opt.state[x] += 1
    return g .* γ
end

struct InverseDecay
    τ::Float64 # Maximum learning rate
    κ::Float64 # Convergence rate to the maximum
    state
end

InverseDecay(τ=1, κ=0.51) = InverseDecay(τ, κ, IdDict())

function Optimise.apply!(opt::InverseDecay, x, g)
    t = get!(()->1, opt.state, x)
    γ = (opt.τ + t)^-opt.κ
    opt.state[x] += 1
    return g .* γ
end

mutable struct DimWiseADADelta
  rho::Float64
  ϵ::Float64
  state::IdDict
end

DimWiseADADelta(ρ = 0.9, ϵ=1e-9) = DimWiseADADelta(ρ, ϵ, IdDict())

function Optimise.apply!(o::DimWiseADADelta, x, Δ)
  ρ = o.rho
  acc = get!(o.state, x) do 
      return zeros(size(x, 1)), zeros(size(x, 1))
  end
  acc .= ρ * acc + (1 - ρ) * vec(mean(Δ; dims=1)).^2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  Δ = Diagonal(sqrt.(Δacc + o.ϵ) ./ sqrt.(acc + o.ϵ)) * Δ
  Δacc .= ρ * Δacc + (1 - ρ) * vec(mean(Δ; dims=1)).^2
  return Δ
end

mutable struct DimWiseADAGrad
  eta::Float64
  ϵ::Float64
  state::IdDict
end

DimWiseADAGrad(η = 0.9, ϵ=1e-9) = DimWiseADAGrad(η, ϵ, IdDict())

function Optimise.apply!(o::DimWiseADAGrad, x, Δ)
  η = o.eta
  acc = get!(o.acc, x) do 
      fill!(zeros(size(x, 1)), o.ϵ)
  end
  acc .+= vec(mean(Δ; dims=1)).^2
  return Δ = Diagonal(η ./ (sqrt.(acc) + ϵ)) * Δ
end

mutable struct DimWiseRMSProp
  eta::Float64
  gamma::Float64
  ϵ::Float64
  state::IdDict
end

DimWiseRMSProp(η = 0.9, γ = 0.9, ϵ=1e-9) = DimWiseRMSProp(η, γ, ϵ, IdDict())

function Optimise.apply!(o::DimWiseRMSProp, x, Δ)
  η = o.eta
  γ = o.gamma
  acc = get!(o.state, x) do
    if x isa CuArray
      gpu(fill!(zeros(size(x, 1)), o.ϵ))
    else
      fill!(zeros(size(x, 1)), o.ϵ)
    end
  end
  acc .= γ * acc + (1 - γ) * diag_ABt(Δ, Δ)
  return Diagonal(η ./ (sqrt.(acc .+ o.ϵ))) * Δ
end

mutable struct DimWiseADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

DimWiseADAM(η = 0.001, β = (0.9, 0.999)) = DimWiseADAM(η, β, IdDict())
const ϵ = 1e-10
function Optimise.apply!(o::DimWiseADAM, x, Δ)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, x) do
      (zeros(size(x, 1)), zeros(size(x, 1)), Float64[β[1], β[2]])
  end

  mt .= β[1] * mt + (1 - β[1]) * vec(mean(Δ; dims=2))
  vt .= β[2] * vt + (1 - β[2]) * vec(mean(Δ; dims=2)).^2
  A =  Diagonal(mt) / (1 - βp[1]) * Diagonal(inv.(sqrt.(vt / (1 - βp[2])) .+ ϵ)) * η
  βp .= βp .* β
  return A * Δ
end
