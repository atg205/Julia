using ContrProject
using LinearAlgebra
plus = LinearAlgebra.Hermitian(1/2 * [1 1; 1 1])
zero = LinearAlgebra.Hermitian([1 0; 0 0])
M = pretty_good_povm(plus,zero)
println(tr(M[1]*zero))
println(tr(M[2]*zero))
println(tr(M[1]*plus))
println(tr(M[2]*plus))
tr(M[1]*plus)