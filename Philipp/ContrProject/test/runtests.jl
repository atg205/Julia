using ContrProject
using Test
using LinearAlgebra
using Ket
using Documenter

doctest(ContrProject, manual=false)

@testset "Basic tests" begin
    Ψ = LinearAlgebra.Hermitian([1 0; 0 0])         # |0><0|
    Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|
    @test state_discrimination([Ψ,Φ])[2] ≈ (2+sqrt(2))/4 atol=1e-6 

    @test state_discrimination([LinearAlgebra.Hermitian([1 0; 0 0]), LinearAlgebra.Hermitian([0 0; 0 1])])[2] ≈ 1.0 atol=1e-6
    @test state_discrimination([LinearAlgebra.Hermitian([1 0; 0 0]), LinearAlgebra.Hermitian([1 0; 0 0])])[2] ≈ 0.5 atol=1e-6
    @test state_discrimination([LinearAlgebra.Hermitian([1 0 0; 0 0 0; 0 0 0]), LinearAlgebra.Hermitian([1 0 0; 0 1 0; 0 0 1]),
    LinearAlgebra.Hermitian([0 0 0; 0 0 0; 0 0 1])])[2] ≈ 1.0 atol=1e-6
end

@testset "Dual Tests" begin
    for i = 1:5
        Ψ = random_state(i)
        Φ = random_state(i)
        @test state_discrimination([Ψ, Φ])[2] ≈ state_discrimination([Ψ, Φ],[],false)[2] atol=1e-6
    end 
    for i = 1:5
        q = rand(2)
        q = q / sum(q)
        Ψ = random_state(i)
        Φ = random_state(i)
        @test state_discrimination([Ψ, Φ], q)[2] ≈ state_discrimination([Ψ, Φ],q,false)[2] atol=1e-6
    end

    for i = 2:5
        q = rand(i)
        q = q / sum(q)
        ρ = [random_state(3) for _ in 1:i]
        @test state_discrimination(ρ, q)[2] ≈ state_discrimination(ρ,q,false)[2] atol=1e-6
    end
end


