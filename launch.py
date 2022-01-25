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





models = ["transformer","resnet","mlp"]
batch_sizes = [128, 256]
epochs = 50
reproduction = 5



for model in models:
    for batch_size in batch_sizes:
        command = "python aTOz.py %s %s %s %s %s %s" % (dataset, task_type, model, str(epochs),  str(batch_size), str(reproduction))
        os.system(command)
        print("model:%s - batch:%s" % (model, str(batch_sizes)))
