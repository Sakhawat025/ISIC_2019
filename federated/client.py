import torch
import flwr as fl


class ISICClient(fl.client.NumPyClient):

    def __init__(self, model, train_loader, device, combined_loss, get_binary_label):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.combined_loss = combined_loss
        self.get_binary_label = get_binary_label

    # Send weights to server
    def get_parameters(self, config):
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    # Receive global weights
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # Local training
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5)

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)

            binary_labels = torch.tensor(
                [self.get_binary_label(x.item()) for x in labels],
                device=self.device
            )

            loss = self.combined_loss(outputs, binary_labels, labels)
            loss.backward()
            optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    # Client evaluation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {}
