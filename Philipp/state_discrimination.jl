using JuMP
import LinearAlgebra    
import SCS
using Hypatia


"""
    state_discrimination(p[, primal=true])

    Return a positive operator-valued measure (POVM) ``{E_i}_{i=1}^N`` and probability P such that if we observe ``E_i`` with average probability P we are in state ``ρ_i``
    
    # Examples
    ```
    julia> Ψ = LinearAlgebra.Hermitian([1 0; 0 0])  # |0><0|
    Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|

    E, P = state_discrimination([Ψ,Φ])
    P
    0.8535533863084384
    ```

"""
function state_discrimination(ρ, primal = true)
    model = Model(() -> Hypatia.Optimizer(verbose = true))
    N = length(ρ)
    d = maximum(size(ρ[1]))
    set_silent(model)
    if primal
        E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
        @constraint(model, sum(E) == LinearAlgebra.I)
        @objective(
            model,
            Max,
            sum(real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N) / N,
        )
        optimize!(model)
    else # solve the dual
        E =@variable(model, [1:d, 1:d] in HermitianPSDCone())
        for i in 1:N
            @constraint(model, E >= (ρ[i]/N), HermitianPSDCone() )
        end
        @objective(
            model,
            Min,
            real(LinearAlgebra.tr(E))
        )
    end
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return E, objective_value(model)
end


function random_state(d)
    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

N = 2
d= 2
ρ = [random_state(d) for i in 1:N]
#ρ = 0.5 * [LinearAlgebra.Hermitian([[1,0] [0,1]])]
#ρ = 0.5 * [[[1,1] [1,1]],[[1,-1] [-1,1]]] # discriminate between plus and minus

Ψ = LinearAlgebra.Hermitian([1 0; 0 0])         # |0><0|
Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|

E, P = state_discrimination([Ψ,Φ])
println(P)


