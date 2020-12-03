

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

mutable struct ScalarADADelta
  rho::Float64
  ϵ::Float64
  state::IdDict
end

ScalarADADelta(ρ = 0.9, ϵ=1e-9) = ScalarADADelta(ρ, ϵ, IdDict())

function Optimise.apply!(o::ScalarADADelta, x, Δ)
  ρ = o.rho
  acc, Δacc = get!(() -> (0.0, 0.0), o.state, x)
  acc = ρ * acc + (1 - ρ) * mean(Δ)^2
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  Δ *= √(Δacc + o.ϵ) / √(acc + o.ϵ)
  Δacc = ρ * Δacc + (1 - ρ) * mean(Δ)^2
  return Δ
end