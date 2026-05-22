import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim

from model import BabyGPT, BabyGPTConfig
from tqdm import tqdm

from utils import prep_shakespeare, sample

### AUTO VS MANUAL ###
AUTO = False

### GPU VS CPU ###
DEVICE = "cuda"

### TRAINING PARAMETERS ###
TRAINING_ITERS = 3000 # Total training steps
EVAL_ITERS = 250 # After how many steps do you want to eval?
LR = 0.0001
BATCH_SIZE = 8
GEN_ITER = 1000
GEN_START = "K"

### LOAD DATASET ###
trainset, testset, meta = prep_shakespeare()
char2idx = meta["char2idx"]
idx2char = meta["idx2char"]

config = BabyGPTConfig(vocab_size=meta["vocab_size"], use_full_auto=AUTO)

### Prebatch all training samples up front for faster sampling ###
# Use NumPy arrays directly for better memory and speed, plus avoid double-call of sample()
train_samples = np.array([sample(trainset, config.max_seq_len) for _ in range(len(trainset) // config.max_seq_len)], dtype=object)
test_samples = np.array([sample(testset, config.max_seq_len) for _ in range(len(testset) // config.max_seq_len)], dtype=object)

### LOAD MODEL ###
config = BabyGPTConfig(vocab_size=meta["vocab_size"], use_full_auto=AUTO)
model = BabyGPT(config).to(DEVICE)

### LOAD OPTIMIZER ###
optimizer = optim.AdamW(model.parameters(), lr=LR)

### LOSS FUNCTION ###
loss_fn = nn.CrossEntropyLoss()

### TRAIN MODEL ###
completed_steps = 0
train = True
pbar = tqdm(range(TRAINING_ITERS))

### GENERATION METHOD ###
def generate():
    start_idx = [char2idx[i] for i in GEN_START]
    start_tensor = mytorch.Tensor(start_idx, dtype=mytorch.uint32).unsqueeze(0).to(DEVICE)

    ### Inference ###
    for _ in range(config.max_seq_len):

        ### Sample next token ###
        with mytorch.no_grad():
            last_logits = model(start_tensor)[:, -1, :]
            last_logits = last_logits.astype(mytorch.float32)
            exp_logits = mytorch.exp(last_logits - mytorch.max(last_logits))
            probs = exp_logits / mytorch.sum(exp_logits)
            next_id = mytorch.multinomial(probs, num_samples=1)[0].numpy().item()

        start_idx.append(next_id)
        start_tensor = mytorch.Tensor(start_idx, dtype=mytorch.uint32).unsqueeze(0).to(DEVICE)

    return start_idx

# Precompute number of batches for efficient random sampling
num_batches = len(train_samples) // BATCH_SIZE

# Cache tensor cast objects to minimize allocations per batch loop
def batch_to_tensor(batch_samples):
    x = np.stack([item[0] for item in batch_samples])
    y = np.stack([item[1] for item in batch_samples])
    return (
        mytorch.Tensor(x, dtype=mytorch.uint32).to(DEVICE),
        mytorch.Tensor(y, dtype=mytorch.uint32).to(DEVICE)
    )

# Optionally enable model compilation / JIT if mytorch supports for further speed:
# model = mytorch.compile(model)

# Enable mixed precision if supported by mytorch (simulated with model.half() here, remove if not valid for your framework)
# model = model.half()

while train:
    # Efficiently sample a batch by random indexing from prebatched train_inputs
    batch_idxs = np.random.choice(len(train_samples), BATCH_SIZE, replace=False)
    batch_samples = train_samples[batch_idxs]
    x, y = batch_to_tensor(batch_samples)

    ### Get Logits ###
    output = model(x)

    ### Compute Loss ###
    loss = loss_fn(output, y)

    ### Update Model ###
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ### Iter ##
    completed_steps += 1
    pbar.update(1)

    if completed_steps % EVAL_ITERS == 0:
        model.eval()
        # Prebatch 100 evaluation samples to avoid eval-time sampling overhead
        eval_idxs = np.random.choice(len(train_samples), 100, replace=False)
        eval_batch = train_samples[eval_idxs]
        eval_x, eval_y = batch_to_tensor(eval_batch)

        with mytorch.no_grad():
            output = model(eval_x)
            eval_loss = loss_fn(output, eval_y)
        
        print(f"Training Loss: {loss.item()}")
        print(f"Eval Loss: {eval_loss.item()}")
        model.train()

    if completed_steps % GEN_ITER == 0 or completed_steps == 1:
        model.eval()
        gen_tokens = generate()
        model.train()

        print("Generation:")
        gen = "".join([idx2char[i] for i in gen_tokens])
        print(gen)
        print("-"*50)

    if completed_steps >= TRAINING_ITERS:
        print("Completed Training!!")
        train = False
        break

model.eval()
gen_tokens = generate()

print("Generation:")
gen = "".join([idx2char[i] for i in gen_tokens])
print(gen)
print("-"*50)