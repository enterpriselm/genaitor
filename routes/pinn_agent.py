import torch
import torch.nn as nn
import ast
import requests
import matplotlib.pyplot as plt

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

class Wave1DPINN(nn.Module):
    def __init__(self, layers):
        super(Wave1DPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i],
                                         layers[i+1]))
            self.activation = torch.tanh

    def forward(self, x, t):
        u = torch.cat([x,t], dim=1)
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        u = self.layers[-1](u)
        return u

    def loss(self, x, t):
        u = self.forward(x, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        c = 1.0
        wave_eq = u_t - c**2 * u_xx
        loss = torch.mean(wave_eq**2)
        return loss
    
def train_pinn(pinn, optimizer, epochs, x, t):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn.loss(x, t)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return pinn

def query(history, lr_list, layers_list):
    input_ = {'losses': history, 'lr\_history': lr_list, 'layers\_history': layers_list}
    prompt = f"""given the following input:
    {input_}
    
    return new value for lr and layers, following this example:

    lr: {lr_list[-1]}
    layers: {layers_list[-1]}
    """
    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {
                "role": "system",
                "content": """You are the best AI agent specialized in providing the best layer and learning rate based on the history of loss, layers used, and learning rates used. You know all the techniques and how to provide the best solution."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]}

    response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)
    return response.json()['choices'][0]['message']['content']

def parse_suggestion(suggestion):
    try:
        lr = suggestion.partition('lr: ')[2].partition('layers')[0]
        if ' ' in lr:
            lr = float(lr.partition(' ')[0])
        else:
            lr = float(lr)
    except:
        print(lr)
    layers = '['+suggestion.partition('layers: [')[2].partition(']')[0]+']'
    layers = ast.literal_eval(layers)
    return lr, layers

x_val = torch.linspace(0, 1, 50, requires_grad=True).reshape(-1, 1)
t_val = torch.linspace(0, 1, 50, requires_grad=True).reshape(-1, 1)

history = []
lr = 1e-3
layers = [2,20,20,1]
loss_without_agent = []
loss_with_agent    = []
lr_list = []
layers_list = []

for iteration in range(10):
    x = torch.linspace(0,1,100, requires_grad=True).reshape(-1,1)
    t = torch.linspace(0,1,100, requires_grad=True).reshape(-1,1)

    print(f"\nIteration {iteration +1 }")
    pinn = Wave1DPINN(layers)
    optimizer = torch.optim.Adam(pinn.parameters(), lr = lr)

    pinn = train_pinn(pinn, optimizer, epochs=1000, x=x, t=t)

    final_loss = pinn.loss(x, t).item()
    loss_without_agent.append(final_loss)

predictions_without_agent = pinn.forward(x_val, t_val).detach().numpy()

history = []

for iteration in range(10):
    x = torch.linspace(0,1,100, requires_grad=True).reshape(-1,1)
    t = torch.linspace(0,1,100, requires_grad=True).reshape(-1,1)

    print(f"\nIteration {iteration +1 }")
    pinn = Wave1DPINN(layers)
    optimizer = torch.optim.Adam(pinn.parameters(), lr = lr)
    print(lr, layers)
    pinn = train_pinn(pinn, optimizer, epochs=1000, x=x, t=t)

    final_loss = pinn.loss(x, t).item()
    history.append(final_loss)
    lr_list.append(lr)
    layers_list.append(layers)
    
    print("Asking Suggestion")
    
    suggestion = query(history, lr_list, layers_list)
    lr, layers = parse_suggestion(suggestion)
    print(lr, layers)
    loss_with_agent.append(final_loss)

predictions_with_agent = pinn.forward(x_val, t_val).detach().numpy()

plt.plot(loss_without_agent, label=f'Iteration {iteration + 1} (Without Agent)', alpha=0.5)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Behavior during Training')
plt.legend()
plt.show()

plt.plot(loss_with_agent, label=f'Iteration {iteration + 1} (With Agent)', alpha=0.5)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Behavior during Training')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(x_val.numpy(), predictions_without_agent, label='Predictions Without Agent', color='blue')
plt.plot(x_val.numpy(), predictions_with_agent, label='Predictions With Agent', color='orange')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Model Predictions Comparison')
plt.legend()
plt.show()
