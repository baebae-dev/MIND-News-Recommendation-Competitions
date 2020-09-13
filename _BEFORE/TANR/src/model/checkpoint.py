import os

model_name = 'TANR'
batch_size = 128

checkpoint_dir = os.path.join('../checkpoint', model_name, str(batch_size))

all_checkpoints = {
        int(x.split('.')[-2].split('-')[1]): x
        for x in os.listdir(checkpoint_dir)
    }

print(all_checkpoints)
print('--------')
print(max(all_checkpoints.keys()))
print('--------')
print(all_checkpoints[max(all_checkpoints.keys())])