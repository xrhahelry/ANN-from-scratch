using Random
using CairoMakie

mutable struct Layer
    weights::Matrix{Float64}
    biases::Vector{Float64}
    activation::Function

    function Layer(input_size::Int, output_size::Int, activation::Function)
        weights = randn(output_size, input_size) * sqrt(2/input_size)
        biases = zeros(output_size)
        new(weights, biases, activation)
    end
end

mutable struct NeuralNetwork
    layers::Vector{Layer}

    function NeuralNetwork()
        new(Layer[])
    end
end

function add_layer!(model::NeuralNetwork, input_size::Int, output_size::Int, activation::Function)
    push!(model.layers, Layer(input_size, output_size, activation))
end

function forward(model::NeuralNetwork, inputs::Matrix{Float64})
    outputs = inputs

    for layer in model.layers
        z = layer.weights * outputs .+ layer.biases
        outputs = layer.activation.(z)
    end

    return outputs
end

function forward(model::NeuralNetwork, inputs::Vector{Float64})
    outputs = inputs

    for layer in model.layers
        z = layer.weights * outputs .+ layer.biases
        outputs = layer.activation.(z)
    end

    return outputs
end

function min_max_normalize(arr)
    min_val = minimum(arr)
    max_val = maximum(arr)
    return (arr .- min_val) ./ (max_val - min_val)
end

relu(x) = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))

x1 = collect(1:100)
x2 = x1 .^ 2
y = (3 .* x1) + (7 .* x2) + (randn(100) .* 5)

x1 = min_max_normalize(x1)
x2 = min_max_normalize(x2)
X = Matrix(hcat(x1, x2)')
y = min_max_normalize(y)

fig = Figure()
ax = Axis(fig[1, 1], title="Linear Regression Data", xlabel="X", ylabel="Y")
scatter!(ax, x1, y, label="Data", markersize=4)

Random.seed!(69)
epoch = 100
lr = 1e-3
l = length(x1)
buf = ones(l)

dLw(Y, x, y) = transpose(x) * (-2(Y-y))
dLb(Y, y) = transpose(-2(Y-y)) * buf

model = NeuralNetwork()

add_layer!(model, 2, 2, relu)
add_layer!(model, 2, 1, relu)

output = forward(model, X)

print(output)

