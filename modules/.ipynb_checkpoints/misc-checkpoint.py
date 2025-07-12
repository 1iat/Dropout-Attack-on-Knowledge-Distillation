import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from os.path import exists


def summarize_diagnostics(history, show):
    """
    Plots loss and accuracy of a model

        Parameters:
            history: history of loss and accuracy values
            show: boolean on whether or not to show the figure at the end (if False, it is up to the user
                  to manage handling showing the figure themselves)
    """
    plt.suptitle("Model Train History", fontsize=14)
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history["train_loss"], color="blue", label="train")
    plt.plot(history["val_loss"], color="yellow", label="validation")
    plt.legend()
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history["train_accuracy"], color="blue", label="train")
    plt.plot(history["val_accuracy"], color="yellow", label="validation")
    plt.legend()
    plt.tight_layout(3)
    plt.show(block=show)


def write_to_json(filename, name, model, acc, conf_matrix, class_rec, class_prec, classes):
    """
    Write model performance information into a json file. Files are written to ../output/filename.

        Parameters:
            filename: the filename of the output json file
            name: the name of the model
            model: the model being evaluated
            acc: the model accuracy on the test set
            conf_matrix: the model confusion matrix on the test set (numpy array)
            class_rec: the model recall on the test set by class (numpy array)
            classes: the dataset label names
    """
    output_dict = {
        "name": name,
        "metrics": model.get_metrics(),
        "accuracy": acc,
        "confusion": conf_matrix.tolist(),
        "classRecall": class_rec.tolist(),
        "classPrecision": class_prec.tolist(),
        "classes": classes,
    }
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, f"../output/{filename}.json")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)


def read_json(file):
    """
    Reads in a json file with a model's saved information metrics

        Parameters:
            filename: The file path of the json file (assumes it is found in the output folder)

        Returns:
            model_info: The saved model information
    """
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, f"../output/{file}")
    with open(file, "r", encoding="utf-8") as f:
        model_info = json.load(f)
        return model_info

def display_model_info(model_info):
    """
    Displays model information in the sys.out channel

        Parameters:
            model_info: the model information to display

    """
    print(f'Model Name: {model_info["name"]}')
    summarize_diagnostics(model_info["metrics"], False)
    print(f'Test set accuracy: {model_info["accuracy"]}')
    conf_matrix = pd.DataFrame(
        model_info["confusion"],
        index=model_info["classes"],
        columns=model_info["classes"],
    )
    class_rec = pd.DataFrame(
        model_info["classRecall"], index=model_info["classes"], columns=["Recall"]
    )
    print("Confusion matrix:")
    print(conf_matrix.to_string())
    print("Test set recall - per class:")
    print(class_rec.to_string())
    if "classPrecision" in model_info:
      class_prec = pd.DataFrame(
        model_info["classPrecision"], index=model_info["classes"], columns=["Precision"]
      )
      print("Test set precision - per class:")
      print(class_prec.to_string())
    plt.show()

def write_to_csv(pd: pd.DataFrame, file: str):
  if not exists(file):
    pd.to_csv(file, index=False)
  else:
    pd.to_csv(file, mode='a', index=False, header=False)