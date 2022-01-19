import os


models = ["mlp", "resnet"]
batch_sizes = [64, 128]

for model in models:
    for batch_size in batch_sizes:
        command = "python aTOz.py dont_get_kicked binclass %s 10 %s 10" % (model, str(batch_size))
        os.system(command)
