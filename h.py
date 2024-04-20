from tqdm import tqdm
import time

# Create an iterable (for example, a range of numbers)
iterable = range(30)

# Wrap the iterable with tqdm to create a progress bar
for item in tqdm(iterable):
    for item in tqdm(iterable):
    # Simulate some computation
        time.sleep(0.1)
    print()
    print('ahoj')
    print()
