using Random: seed!

data = [[0, 1], [1,3], [2, 5], [3, 7], [4, 9]]
x = [row[1] for row in data]
Y = [row[2] for row in data]

seed!(69)
W = randn()
b = randn()
epoch = 100
lr = 1e-3
l = length(data)
buf = ones(l)

dLw(Y, x, W, b) = transpose(-2(Y-(W.*x .+ b)))*x
dLb(Y, x, W, b) = transpose(-2(Y-(W.*x .+ b)))*buf

for i in 1:epoch
    loss = 0
    y = W .* x .+ b
    loss = transpose((Y-y).^2)*buf
    loss /= l
    global W -= lr*dLw(Y, x, W, b)
    global b -= lr*dLb(Y, x, W, b)
    println("Epoch[$(i)/$(epoch)]: Loss = $(loss), W = $(W), b = $(b)")
end

# jfkalsdjfkld
