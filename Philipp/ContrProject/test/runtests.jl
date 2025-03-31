using ContrProject
using Test
using LinearAlgebra
using Ket

@testset "Basic tests" begin
    Ψ = LinearAlgebra.Hermitian([1 0; 0 0])         # |0><0|
    Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|
    @test state_discrimination([Ψ,Φ])[2] ≈ (2+sqrt(2))/4

    @test state_discrimination([LinearAlgebra.Hermitian([1 0; 0 0]), LinearAlgebra.Hermitian([0 0; 0 1])])[2] ≈ 1.0
    @test state_discrimination([LinearAlgebra.Hermitian([1 0; 0 0]), LinearAlgebra.Hermitian([1 0; 0 0])])[2] ≈ 0.5
    @test state_discrimination([LinearAlgebra.Hermitian([1 0 0; 0 0 0; 0 0 0]), LinearAlgebra.Hermitian([1 0 0; 0 1 0; 0 0 1]),
    LinearAlgebra.Hermitian([0 0 0; 0 0 0; 0 0 1])])[2] ≈ 1.0
end

@testset "Dual Tests" begin
    Ψ = random_state(2)
    Φ = random_state(2)
    @test state_discrimination([Ψ, Φ])[2] ≈ state_discrimination([Ψ, Φ],[],false)[2] 
end


