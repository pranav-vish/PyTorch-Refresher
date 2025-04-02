# Import necessary PyTorch libraries
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms

# Define a simple neural network class that inherits from nn.Module
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Call parent class constructor
        super(SimpleNN, self).__init__()
        
        # Create first fully connected layer
        # Transforms data from input_size dimensions to hidden_size dimensions
        self.layer1 = nn.Linear(input_size, hidden_size)
        
        # ReLU activation function - introduces non-linearity
        # f(x) = max(0,x)
        self.relu = nn.ReLU()  # ReLU activation function
        
        # Second fully connected layer
        # Transforms data from hidden_size dimensions to output_size dimensions
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define the forward pass of the network
        x = self.layer1(x)  # First linear transformation
        x = self.relu(x)    # Apply ReLU activation
        x = self.layer2(x)  # Second linear transformation
        return x

# Example usage with detailed explanations
def main():
    # Define network architecture parameters
    input_size = 10    # Number of input features
    hidden_size = 20   # Number of neurons in hidden layer
    output_size = 2    # Number of output features
    learning_rate = 0.01   # Step size for gradient descent
    num_epochs = 100   # Number of times to process entire dataset
    
    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)  # Create instance of our network
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    # SGD optimizer with specified learning rate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Create dummy training data
        # torch.randn creates random numbers from normal distribution
        X = torch.randn(32, input_size)  # 32 samples, each with input_size features
        y = torch.randn(32, output_size)  # Corresponding target values
        
        # Forward pass: compute model predictions
        outputs = model(X)
        # Calculate loss between predictions and targets
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update model parameters
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Standard Python idiom to ensure main() only runs if this file is run directly
if __name__ == "__main__":
    main()

