module ContrProject
using JuMP
import LinearAlgebra    
import SCS
using Hypatia
using Ket

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
function state_discrimination(ρ::Vector{<:LinearAlgebra.Hermitian},q::Vector{Float64} = Float64[], primal = true::Bool)
    model = Model(() -> Hypatia.Optimizer(verbose = true))
    N = length(ρ)
    d = maximum(size(ρ[1]))
    set_silent(model)

    if  ~isempty(q)
        @assert length(q) == length(ρ) 
        @assert isapprox(sum(q), 1.0; atol=eps(Float32))
    end

    if primal
        E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
        @constraint(model, sum(E) == LinearAlgebra.I)
        if isempty(q)
            @objective(
                model,
                Max,
                sum(real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N) / N,
            )
        else
            @objective(
                model,
                Max,
                sum(q[i] * real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N),
            )
        end
        optimize!(model)
    else # solve the dual
        E =@variable(model, [1:d, 1:d] in HermitianPSDCone())
        for i in 1:N
            if isempty(q)
                @constraint(model, E >= (ρ[i]/N), HermitianPSDCone() )
            else
                @constraint(model, E >= (ρ[i]/q[i]), HermitianPSDCone() )
            end
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

N = 2
d= 2
ρ = [random_state(d) for i in 1:N]
#ρ = 0.5 * [LinearAlgebra.Hermitian([[1,0] [0,1]])]
#ρ = 0.5 * [[[1,1] [1,1]],[[1,-1] [-1,1]]] # discriminate between plus and minus

Ψ = LinearAlgebra.Hermitian([1 0; 0 0])         # |0><0|
Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|

E, P = state_discrimination([Ψ,Φ],[0.3,0.7])
println(P)

println(state_discrimination([LinearAlgebra.Hermitian([1 0; 0 0]),LinearAlgebra.Hermitian([0 0; 0 1])])[2])

function a(b::Integer) 
    println(b)
end

export state_discrimination
export a


end