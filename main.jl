include("nn.jl")
using .nn

using Random
using CairoMakie

x1 = collect(1:100)
x2 = x1 .^ 2
y = (3 .* x1) + (7 .* x2) + (randn(100) .* 5)

x1 = min_max_normalize(x1)
x2 = min_max_normalize(x2)
X = Matrix(hcat(x1, x2)')
y = min_max_normalize(y)

Random.seed!(69)
epoch = 100
lr = 1e-3

model = NeuralNetwork()

add_layer!(model, 2, 2, relu)
add_layer!(model, 2, 1, relu)

print_model(model)
output = forward(model, X)


