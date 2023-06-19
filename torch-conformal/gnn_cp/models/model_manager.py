import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, APPNP, SAGEConv

from sklearn.metrics import classification_report
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")


class GraphModelManager(object):
    def __init__(
        self, model,
        optimizer_lambda=None, optimizer_keyargs=None,
        checkpoint_address=None, model_name=None):

        self.model = model.to(device)
        
        if optimizer_keyargs is None:
            optimizer_keyargs = {"weight_decay": 1e-2}

        if optimizer_lambda is None:
            optimizer_lambda = lambda model_params: torch.optim.Adam(model_params, **optimizer_keyargs)
            
        self.optimizer = optimizer_lambda(self.model.parameters())
        
        self.checkpoint_address = checkpoint_address or "./model_cache"
        self.model_name = model_name or self.random_model_name()

    def load_model(self):
        if f"{self.model_name}.model" in os.listdir(self.checkpoint_address):
            if torch.cuda.is_available() is False:
                self.model = torch.load(os.path.join(self.checkpoint_address, f"{self.model_name}.model"), map_location=torch.device('cpu'))
            else:
                self.model = torch.load(os.path.join(self.checkpoint_address, f"{self.model_name}.model"))
            return True
        return False

    def train_iteration(
        self, X,
        training_idx=None,
        validation_idx=None,
        loss_fn=F.cross_entropy):

        self.model.train()
        self.optimizer.zero_grad()
        
        prediction = self.model(X.to(device))
        training_loss = loss_fn(prediction[training_idx], X.y[training_idx])
        if not validation_idx is None:
            validation_loss = loss_fn(prediction[validation_idx], X.y[validation_idx])

        training_loss.backward()
        self.optimizer.step()

        if not validation_idx is None:
            return training_loss.detach().item(), validation_loss.detach().item()

        return training_loss.detach().item(), None

    def fit(
        self,
        X,
        training_idx=None,
        validation_idx=None,
        n_epochs=100,
        loss_fn=F.cross_entropy,
        warmup_steps=50,
        early_stopping_warnings=10,
        verbose_on_quantile=0.1):

        verbose_on = int(n_epochs * verbose_on_quantile)
        saved_flag = False

        if verbose_on == 0:
            verbose_on = 1

        training_losses = []
        validation_losses = []

        for warmup_epoch in range(warmup_steps):
            training_loss, validation_loss = self.train_iteration(
                X, training_idx, validation_idx, loss_fn)

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            if warmup_epoch % verbose_on == 0:
                print(f"WarmUP:: training: epoch = {warmup_epoch} --> "
                      f"training loss = {training_loss} "
                      f"validation loss = {validation_loss}")


        min_epochs = n_epochs // 10
        warning_steps = early_stopping_warnings
        
        best_validation_loss = None
        for epoch in range(n_epochs - warmup_steps):
            training_loss, validation_loss = self.train_iteration(X, training_idx,
                                                                  validation_idx, loss_fn=loss_fn)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            if epoch % verbose_on == 0:
                print(f"training: epoch = {epoch} --> training loss = {training_loss} "
                      f"validation loss = {validation_loss}")
            if validation_loss <= min(validation_losses):
                torch.save(self.model, os.path.join(self.checkpoint_address, f"{self.model_name}.model"))
                saved_flag = True
            
            else:
                warning_steps -= 1
                if warning_steps <= 0:
                    print(f"Early stopping on epoch {epoch + warning_steps}")
                    break

        if saved_flag is False:
            torch.save(self.model, os.path.join(self.checkpoint_address, f"{self.model_name}.model"))

        return training_losses, validation_losses

    def predict(self, X, test_idx=None, training_idx=None, return_embeddings=True):
        self.model.eval()
        prediction_result = self.model(X.to(device))
        if not test_idx is None:
            prediction_result = prediction_result[test_idx]

        if return_embeddings:
            return prediction_result.detach()

        prediction_classes = prediction_result.argmax(dim=1).detach()
        return prediction_classes

    def prediction_evaluation(self, X, test_idx):
        prediction_result = self.predict(X, test_idx=test_idx, return_embeddings=False).cpu().numpy()

        print(classification_report(y_true=X.y[test_idx].cpu(), y_pred=prediction_result))

    @staticmethod
    def random_model_name():
        import random
        import string
        def randStr(chars = string.ascii_uppercase + string.digits, N=10):
            return ''.join(random.choice(chars) for _ in range(N))
        return randStr(N=10)