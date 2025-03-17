"""
Model trainer class for audio classification models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score

from data_module import kfold


class ModelTrainer:
    def __init__(self, model, optimizer, loss_function, epochs, batch_size, model_name):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer to use
            loss_function: Loss function to optimize
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            model_name: Name of the model for logging
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = model_name

    def train(self, custom_data_module):
        """
        Train the model using k-fold cross-validation.
        
        Args:
            custom_data_module: Data module containing dataset configuration
            
        Returns:
            Lists of training and validation metrics for plotting
        """
        wandb.init(project='Audio_Classification', name=self.model_name)

        train_loss_list = []
        train_accuracy_list = []
        validation_loss_list = []
        validation_accuracy_list = []

        for epoch in range(self.epochs):
            train_loss_fold = []
            train_accuracy_fold = []
            validation_loss_fold = []
            validation_accuracy_fold = []

            # Perform k-fold cross-validation (folds 2-5)
            for k in range(2, 6):
                # Get data loaders for this fold
                data_module = kfold(
                    k, 
                    custom_data_module.data_module_kwargs["data_directory"],
                    custom_data_module.data_module_kwargs["data_frame"]
                )
                train_loader = data_module.train_dataloader()
                val_loader = data_module.val_dataloader()

                batch_train_loss = 0
                batch_train_accuracy = 0

                # Training loop
                self.model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = self.loss_function(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    batch_train_loss += loss.item()
                    accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y_batch).item()
                    batch_train_accuracy += accuracy

                train_loss_fold.append(batch_train_loss)
                train_accuracy_fold.append((batch_train_accuracy / 240))

                # Validation loop
                batch_validation_loss = 0
                batch_validation_accuracy = 0
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                        y_pred = self.model(X_batch)
                        loss = self.loss_function(y_pred, y_batch)
                        accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y_batch).item()
                        batch_validation_loss += loss.item()
                        batch_validation_accuracy += accuracy

                validation_loss_fold.append(batch_validation_loss)
                validation_accuracy_fold.append((batch_validation_accuracy / 80))

            # Average metrics across folds for this epoch
            avg_train_loss = np.mean(train_loss_fold)
            avg_train_accuracy = np.mean(train_accuracy_fold)
            avg_validation_loss = np.mean(validation_loss_fold)
            avg_validation_accuracy = np.mean(validation_accuracy_fold)

            # Store metrics for plotting
            train_loss_list.append(avg_train_loss)
            train_accuracy_list.append(avg_train_accuracy)
            validation_loss_list.append(avg_validation_loss)
            validation_accuracy_list.append(avg_validation_accuracy)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_accuracy * 100,
                "validation_loss": avg_validation_loss,
                "validation_accuracy": avg_validation_accuracy * 100
            })

            # Print epoch summary
            print(f'Epoch:{epoch + 1}\n'
                  f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy * 100:.2f}%\n'
                  f'Validation Loss: {avg_validation_loss:.4f}, Validation Accuracy: {avg_validation_accuracy * 100:.2f}%')

        return train_loss_list, train_accuracy_list, validation_loss_list, validation_accuracy_list

    def plot_loss_accuracy(self, epochs, metric_values_list, labels, ylabel, title, colors):
        """
        Plot metrics over epochs.
        
        Args:
            epochs: Number of epochs
            metric_values_list: List of metric values to plot
            labels: Labels for the plot legend
            ylabel: Y-axis label
            title: Plot title
            colors: Colors for each line
        """
        for metric_values, label, color in zip(metric_values_list, labels, colors):
            plt.plot(range(1, epochs + 1), metric_values, label=label, color=color)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(f"{self.model_name}_{ylabel.lower()}.png")
        plt.show()

    def evaluation_and_metrics(self, custom_data_module):
        """
        Evaluate the model on test data and compute metrics.
        
        Args:
            custom_data_module: Data module containing test dataset
        """
        self.model.eval()
        with torch.no_grad():
            test_loader = custom_data_module.test_dataloader()

            avg_test_loss = 0
            avg_test_accuracy = 0
            Y_pred = torch.tensor([])
            y_test = torch.tensor([])
            
            # Collect predictions
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y_batch).item()
                avg_test_loss += loss.item()
                avg_test_accuracy += accuracy
                
                # Move to the same device before concatenating
                Y_pred = Y_pred.to(self.device)
                Y_pred = torch.cat((Y_pred, y_pred), dim=0)
                y_test = y_test.to(self.device)
                y_test = torch.cat((y_test, y_batch), dim=0)

            avg_test_accuracy = avg_test_accuracy / 80

            # Log test metrics
            wandb.log({"Test Loss": f"{avg_test_loss:.4f}"})
            wandb.log({"Test Accuracy": f"{(avg_test_accuracy * 100):.2f}"})
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy * 100:.2f}%')

            # Calculate F1 score
            f1 = f1_score(y_test.cpu(), torch.argmax(Y_pred.cpu(), dim=1), average='weighted')
            print(f"F1 SCORE: {f1:.4f}")
            wandb.log({"Overall F1 Score": f"{f1}"})

            # Plot confusion matrix
            conf_matrix = confusion_matrix(y_test.cpu(), torch.argmax(Y_pred.cpu(), dim=1))
            plt.figure(figsize=(10, 8))
            class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(f"{self.model_name}_confusion_matrix.png")
            plt.show()

            # Plot ROC curves
            y_test_onehot = torch.zeros(len(y_test), len(class_names), device=Y_pred.device)
            y_test_indices = y_test.unsqueeze(1).to(torch.long).to(Y_pred.device)
            y_test_onehot.scatter_(1, y_test_indices, 1)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(len(class_names)):
                fpr[i], tpr[i], _ = roc_curve(y_test_onehot.cpu().numpy()[:, i], torch.argmax(Y_pred.cpu(), dim=1) == i)
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(12, 8))
            for i in range(len(class_names)):
                plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')

            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(f"{self.model_name}_roc_curve.png")
            plt.show()
            
    def count_parameters(self):
        """
        Count the number of parameters in the model.
        
        Returns:
            Tuple of (total parameters, trainable parameters, non-trainable parameters)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print(f'Total parameters: {total_params}')
        print(f'Trainable parameters: {trainable_params}')
        print(f'Non-trainable parameters: {non_trainable_params}')
        
        return total_params, trainable_params, non_trainable_params
