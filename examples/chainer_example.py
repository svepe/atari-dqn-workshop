import numpy as np

from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer.optimizers import Adam

# Generate sample data
N = 100
xs = np.linspace(0, 1, N)
noise = np.random.rand(N)
ys = 2 * xs + 1
ys += 1e-2 * noise

xs = xs.reshape(N, 1).astype(np.float32)
ys = ys.reshape(N, 1).astype(np.float32)

# Create simple linear regression model
class LinearRegression(Chain):
    def __init__(self):
        super().__init__(
            linear=L.Linear(1),
        )

    def __call__(self, x):
        return self.linear(x)


# Train model on data
model = LinearRegression()
optimizer = Adam(alpha=1e-3)
optimizer.setup(model)

print("Optimising...")
for iter in range(10000):
    loss = F.mean_squared_error(ys, model(xs))

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if iter % 1000 == 0:
        print("{}: {}".format(iter, loss.array))

w = model.linear.W.array[0, 0]
b = model.linear.b.array[0]
print("The line is y = {}x + {}".format(w, b))
