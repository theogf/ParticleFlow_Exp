using LinearAlgebra

D = 100
N = 10

psi = rand(D, N)
x = rand(D, N)


true_f(psi, x) = mean(p * x_' for (p, x_) in zip(eachcol(psi), eachcol(x))) * x

true_f(psi, x)

permute_f(psi, x) = 1 / N * (psi * x') * x

permute_f(psi, x)

true_f(psi, x) ≈ permute_f(psi, x)

psi * x' * x
x * psi' * x
