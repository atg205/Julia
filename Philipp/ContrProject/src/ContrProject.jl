module ContrProject
using JuMP
import LinearAlgebra    
import SCS
using Hypatia
using Ket


```@meta
DocTestSetup = quote
    using LinearAlgebra
end
```

"""
    state_discrimination(p[,q=[], primal=true])

Return a positive operator-valued measure (POVM) ``{E_i}_{i=1}^N`` and probability P such that if we observe ``E_i`` with average probability P we are in state ``ρ_i``

# Arguments
- ρ: list of state to be discriminated
- q: (optional) list of probabilities associated with the state ρ, if not provided, uniform probability is assumed
- primal: (optional) if false, compute the dual of the optimization problem, default: True

# Examples
```
Ψ = LinearAlgebra.Hermitian([1 0; 0 0])  # |0><0|
Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|

E, P = state_discrimination([Ψ,Φ])
P
0.8535533863084384
```

"""
function state_discrimination(ρ::Vector{<:LinearAlgebra.Hermitian},q::Vector{<:Any} = Float64[], primal = true::Bool)
    model = Model(() -> Hypatia.Optimizer(verbose = false))
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
    else # solve the dual
        E =@variable(model, [1:d, 1:d] in HermitianPSDCone())
        for i in 1:N
            if isempty(q)
                @constraint(model, E >= (ρ[i]/N), HermitianPSDCone() )
            else
                @constraint(model, E >= (ρ[i]*q[i]), HermitianPSDCone() )
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

"""
    pretty_good_povm(ρ...[,q=[]])

Return a positive operator-valued measure (POVM) ``{E_i}_{i=1}^N`` using the pretty good measurements method to discriminate between the states ρ

# Examples
```
jldoctest
julia> using LinearAlgebra
julia> plus = Hermitian(1/2 * [1 1; 1 1])
julia> zero = Hermitian([1 0; 0 0])
julia> E = pretty_good_povm(plus,zero)
julia> tr(E[1]*plus)
0.8535533905932735
```
"""
function pretty_good_povm(ρ::LinearAlgebra.Hermitian...;q::Vector{<:Any} = Float64[])
    M = sum(ρ)
    M_inv_sqrt = sqrt(inv(M))
    if q == []
        M_i = [M_inv_sqrt * ρi * M_inv_sqrt for ρi in ρ]
    else
        @assert sum(q) ≈ 1
        M_i = [M_inv_sqrt * qi * ρi * M_inv_sqrt for (qi, ρi) in zip(q,ρ)]
    end
    println(sum(M_i))
    @assert sum(M_i) ≈ LinearAlgebra.I
    return M_i
end


export state_discrimination
export pretty_good_povm

end