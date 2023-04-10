using Flux

# Input size and hidden layer size
n_inputs = 4
n_hidden = 64

# Train the model with bias
train_model = Chain(
    Dense(n_inputs, n_hidden, relu),
    Dense(n_hidden, 1)
)

# ... training process ...

# Create a copy of the trained model for evaluation
eval_model = deepcopy(train_model)

# Set the biases of all dense layers in the evaluation model to zero
for layer in eval_model.layers
    if layer isa Dense
        Flux.params(layer)[2] .= zeros(size(Flux.params(layer)[2]))
    end
end

# Evaluate the model
prediction = eval_model([1,1,1,1])

