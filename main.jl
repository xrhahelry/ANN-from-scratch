using Random
using CairoMakie

function min_max_normalize(arr)
    min_val = minimum(arr)
    max_val = maximum(arr)
    return (arr .- min_val) ./ (max_val - min_val)
end

relu(x) = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))

x1 = collect(1:100)
x2 = x1 .^ 2
Y = (3 .* x1) + (7 .* x2) + (randn(100) .* 5)

x1 = min_max_normalize(x1)
x2 = min_max_normalize(x2)
x = hcat(x1, x2)
Y = min_max_normalize(Y)

fig = Figure()
ax = Axis(fig[1, 1], title="Linear Regression Data", xlabel="X", ylabel="Y")
scatter!(ax, x1, Y, label="Data", markersize=4)

Random.seed!(69)
W1 = reshape(randn(4), 2, 2)
W2 = rand(2)
b1 = randn(2)
b2 = randn()
epoch = 100
lr = 1e-3
l = length(x1)
buf = ones(l)

dLw(Y, x, y) = transpose(x) * (-2(Y-y))
dLb(Y, y) = transpose(-2(Y-y)) * buf

for i in 1:epoch
    loss = 0
    z = x*W1 .+ b1
    Z = relu.(z)
    y = Z*W2 .+ b2
    y = relu.(y)
    loss = transpose((Y-y).^2)*buf
    loss /= l
    # global W -= lr.*dLw(Y, x, Z)
    # global b -= lr*dLb(Y, y)
end
# println("W1 = $(W[1]), W2 = $(W[2]), b = $(b)")

# outputs = relu.(W[1] .* x2 + W[2] .* x1 .+ b)
# scatter!(ax, x1, outputs, label="Fitted Line", markersize=4)
# Legend(fig[1,2], ax, "Legend")
# display(fig)
