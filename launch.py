import os
import sys


dataset = sys.argv[1]

if dataset.lower() == "kdd":
    task_type = "multiclass"
elif dataset.lower() == "forest_cover":
    task_type = "multiclass"
elif dataset.lower() == "adult_income":
    task_type = "binclass"
elif dataset.lower() == "dont_get_kicked":
    task_type = "binclass"
elif dataset.lower() == "used_cars":
    task_type = "regression"





models = ["resnet","mlp"]
batch_sizes = [128, 256]
epochs = 20
reproduction = 5



for model in models:
    for batch_size in batch_sizes:
        command = "python aTOz.py %s %s %s %s %s %s" % (dataset, task_type, model, str(epochs),  str(batch_size), str(reproduction))
        print("Launched model:%s - batch:%s" % (model, str(batch_sizes)))
        os.system(command)
        print("Terminated model:%s - batch:%s" % (model, str(batch_sizes)))
