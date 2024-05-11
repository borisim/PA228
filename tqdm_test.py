import torch

# Create a tensor with requires_grad=True
tensor_with_grad = torch.randn(3, 3, requires_grad=True)

# Perform some computation that involves the tensor
result = tensor_with_grad.sum()

# Backpropagate to compute gradients
result.backward()

# Check if gradients have been computed for the tensor
if tensor_with_grad.grad is not None:
    print("Gradients have been computed for the tensor.")
else:
    print("Gradients have not been computed for the tensor.")

# Output the gradients
print("Gradients:", result.grad)
