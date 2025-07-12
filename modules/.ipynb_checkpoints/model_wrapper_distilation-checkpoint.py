class NetWrapper:
    def __init__(self, model, loss_fcn, optimizer, optim_args, reshape_fcn=None, log_dir=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), *optim_args)
        self.loss_fcn = loss_fcn
        self.reshape_fcn = reshape_fcn
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _run_validation(self, dataloader, num_classes=10):
        """
        Runs the validation loop and computes additional metrics: recall and precision.
        """
        self.model.eval()
        confusion_matrix = torch.zeros(num_classes, num_classes, device=self.device)
        total_loss = 0.0
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for inputs, labels in dataloader:
                if self.reshape_fcn:
                    inputs = self.reshape_fcn(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                outputs = self.model(inputs)
                loss = self.loss_fcn(outputs, labels)
                total_loss += loss.item()
    
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(dataloader)
    
        # Safeguard against divide-by-zero issues
        class_recall = (confusion_matrix.diag() / confusion_matrix.sum(1)).cpu().numpy()
        class_precision = (confusion_matrix.diag() / confusion_matrix.sum(0)).cpu().numpy()
        class_recall = np.nan_to_num(class_recall)  # Replace NaN with 0
        class_precision = np.nan_to_num(class_precision)  # Replace NaN with 0
    
        return accuracy, avg_loss, confusion_matrix.cpu().numpy(), class_recall, class_precision


    def fit(self, trainloader, valloader, num_epochs, verbose=True):
        """
        Train the model.
        """
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in trainloader:
                if self.reshape_fcn:
                    inputs = self.reshape_fcn(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fcn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_accuracy = correct / total
            train_loss = running_loss / len(trainloader)

            # Validation
            val_loss, val_accuracy, val_f1 = self._run_validation(valloader)[:3]

            # Log metrics
            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_accuracy"].append(train_accuracy)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_accuracy"].append(val_accuracy)
            self.metrics["val_f1"].append(val_f1)

            if verbose:
                print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, "
                      f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, "
                      f"Time: {time.time() - start:.2f}s")

    def evaluate(self, dataloader, num_classes=10):
        """
        Evaluates the model's performance on a dataset and returns additional metrics.
        """
        return self._run_validation(dataloader, num_classes)
