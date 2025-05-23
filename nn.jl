module nn
export Layer, NeuralNetwork, add_layer!, forward, min_max_normalize, relu, sigmoid, dLw, dLb, print_model

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

function print_model(model::NeuralNetwork)
    for layer in model.layers
        println(layer.weights)
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
dLw(Y, x, y) = transpose(x) * (-2(Y-y))
dLb(Y, y) = transpose(-2(Y-y)) * ones(length(y))
end
