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




models = ["mlp", "resnet"]
batch_sizes = [128, 256]

for model in models:
    for batch_size in batch_sizes:
        command = "python aTOz.py %s %s %s 10 %s 10" % (dataset, task_type, model, str(batch_size))
        os.system(command)
        print("model:%s - batch:%s" % (model, str(batch_sizes)))
