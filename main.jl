using Random
using CairoMakie

function min_max_normalize(arr)
    min_val = minimum(arr)
    max_val = maximum(arr)
    return (arr .- min_val) ./ (max_val - min_val)
end

relu(x) = max(0, x)

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
W = randn(2)
b = randn()
epoch = 100
lr = 1e-3
l = length(x1)
buf = ones(l)

dLw(Y, x, y) = transpose(x) * (-2(Y-y))
dLb(Y, y) = transpose(-2(Y-y)) * buf
accum = []

for i in 1:epoch
    loss = 0
    y = x*W .+ b
    yy = relu.(y)
    loss = transpose((Y-yy).^2)*buf
    loss /= l
    global W -= lr.*dLw(Y, x, yy)
    global b -= lr*dLb(Y, y)
    append!(accum, loss)
end
println("W1 = $(W[1]), W2 = $(W[2]), b = $(b)")

outputs = W[1] .* x2 + W[2] .* x1 .+ b

lines!(ax, x1, outputs, label="Fitted Line", linewidth=2, color=:red)
Legend(fig[1,2], ax, "Legend")
# lines!(fig[1,3], ax, collect(1:epoch), accum, label="Fitted Line", linewidth=2, color=:red)
display(fig)
