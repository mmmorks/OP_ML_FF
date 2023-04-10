using Flux

# Input size
n_inputs = 4

# Train the model with bias
train_model = Chain(
    Dense(n_inputs, 1)
)

# ... training process ...

# Create a new model for evaluation with bias set to zero
eval_model = Chain(
    Dense(n_inputs, 1, bias=zeros(1))
)

# Copy the weights from the trained model
Flux.params(eval_model)[1] .= Flux.params(train_model)[1]

# Evaluate the model
prediction = eval_model(x)
