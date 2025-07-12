import torch
import time


class NetWrapper:
    def __init__(self, model, loss_fcn, optimizer, optim_args, reshape_fcn=None):
        """
        A model wrapper that contains a model and adds fit/evaluation functions

            Parameters:
                model: The model to fit/evaluate
                loss_fcn: The nn loss function module to be used
                optimizer: The optimizer to be used
                optim_args: The arguments to be used with the given optimizer. Must be a list with args in order
                            as inputted by the optimizer init function.
                reshape_fcn: a function to reshape batch data (default=None)
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), *optim_args)
        self.loss_fcn = loss_fcn
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }  # metrics recording trainset accuracy
        self.reshape_fcn = reshape_fcn

    def _record_metrics(self, verbose, epoch, train_loss, val_loss, train_acc, val_acc):
        """
        Records epoch's loss and accuracy statistics

            Parameters:
                verbose: whether or not model statistics are printed
                epoch: the current epoch
                train_loss: the training loss for the current epoch
                val_loss: the validation loss for the current epoch
                train_acc: the train accuracy for the current epoch
                val_acc: the validation accuracy for the current epoch
        """
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_accuracy"].append(train_acc)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["val_accuracy"].append(val_acc)
        if verbose:
            print(f"Training loss in epoch {epoch} :::: {train_loss}")
            print(f"Training Accuracy in epoch {epoch} :::: {train_acc * 100:.2f}")
            print(f"Validation loss in epoch {epoch} :::: {val_loss}")
            print(f"Validation Accuracy in epoch {epoch} :::: {val_acc * 100:.2f}")

    def _run_validation(self, input_data, num_classes=10):
        """
        Runs the validation loop

            Parameters:
                input_data: the input data to run the model on
        """
        self.model.eval()
        nb_classes = num_classes
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            loss_ep = 0
            for _, (data, target) in enumerate(input_data):
                if self.reshape_fcn is not None:
                    data = self.reshape_fcn(data)
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                output = self.model.forward(data)
                loss = self.loss_fcn(output, target)
                _, predictions = output.max(1)
                loss_ep += loss.item()
                num_correct += (predictions == target).sum()
                num_samples += predictions.size(0)
                for t, p in zip(target.view(-1), predictions.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        accuracy = float(num_correct) / float(num_samples)
        return (
            accuracy,
            loss_ep / len(input_data),
            confusion_matrix.numpy(),
            (confusion_matrix.diag() / confusion_matrix.sum(1)).numpy(),
            (confusion_matrix.diag() / confusion_matrix.sum(0)).numpy()
        )

    def fit(self, train_input, val_input, num_epochs, verbose, num_classes=10, patience=5):
        """
        Train the model with early stopping

            Parameters:
                train_input: a dataloader to train the model on
                val_input: a dataloader to run model validation
                num_epochs: number of epochs to run
                verbose: whether or not model statistics are printed
                patience: how many epochs to wait after last improvement before stopping
        """
        best_val_acc = 0.0
        counter = 0

        for i in range(num_epochs):
            start = time.time()
            train_loss = 0
            self.model.train()
            for _, (data, target) in enumerate(train_input):
                if self.reshape_fcn is not None:
                    data = self.reshape_fcn(data)
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                loss = self.loss_fcn(output, target)
                loss.backward()
                self.optimizer.step()
                # Record for metrics
                train_loss += loss.item()

            train_acc, train_loss, _, _, _ = self._run_validation(train_input, num_classes)
            val_acc, val_loss, _, _, _ = self._run_validation(val_input, num_classes)

            self._record_metrics(
                verbose, i + 1, train_loss, val_loss, train_acc, val_acc
            )
            end = time.time()
            if verbose:
                print(f"Time Elapsed: {end - start:.2f}s")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0  # Reset counter if improvement
            else:
                counter += 1
                if verbose:
                    print(f"EarlyStopping counter: {counter} out of {patience}")
                if counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break


    def evaluate(self, input_data, num_classes=10):
        """
        Evaluates the model's performance on a input set of data

            Parameters:
                input_data: the data to run the model on

            Returns:
                accuracy: The model accuracy on the input data
                loss: The model loss
                conf_matrix: The confusion matrix on the different classes. type: np.array
                class_recall: The model recall on each class
        """
        return self._run_validation(input_data, num_classes)

    def get_metrics(self):
        return self.metrics
