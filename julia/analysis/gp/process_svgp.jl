g = @gif for (i, mu_g, mu_a, mu_s) in zip(iters, mus_g, mus_a, mus_s)
    plot(x, f, label = "Truth",title = "i = $i")
    plot!(x, P*mu_g, label = "Gauss", color = colors[1])
    scatter!(Z, mu_g, label = "Gauss", color = colors[1])
    plot!(x, P*mu_a, label = "ADVI", color = colors[2])
    scatter!(Z, mu_a, label = "ADVI", color = colors[2])
    plot!(x, P*mu_s, label = "Stein", color = colors[3])
    scatter!(Z, mu_s, label = "Stein", color = colors[3])
end

display(g)

labels = ["Gauss" "ADVI" "Stein"]
plot(get.([g_h, a_h, s_h], :l_kernel), label = labels)
plot(get.([g_h, a_h, s_h], :σ_kernel), label = labels)
plot(get.([g_h, a_h, s_h], :σ_gaussian), label = labels)
