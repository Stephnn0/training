import os
import torch
import torch.multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from llama.model import Transformer, ModelArgs  # Assuming Llama3 is your model class
import torch.nn as nn
import torch.nn.functional as F

from llama import Tokenizer
from llama.tokenizer import ChatFormat



def preprocess_data_with_tokenizer(train_data, tokenizer, format):
        tokenized_data = []
        for question, answer in train_data:

            question_encoded = format.encode_message({"role": "user", "content": question})

            answer_encoded = format.encode_message({"role": "assistant", "content": answer})

            tokenized_data.append((question_encoded, answer_encoded))

        return tokenized_data


def setup_model_parallel(rank, world_size, master_addr="localhost", master_port=12357, backend="nccl"):
    """
    Initialize the model parallelism and distributed process group
    """
    # Initialize the process group for distributed training
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    print(f"Initializing with rank {rank}/{world_size}...")

    torch.distributed.init_process_group(backend=backend)

    fs_init.initialize_model_parallel(world_size)

    torch.cuda.set_device(rank)

    torch.manual_seed(1)



def run(rank, world_size):
    setup_model_parallel(rank, world_size)  # Setup model parallelism and distributed environment

    params = {
      "dim": 2048,
      "ffn_dim_multiplier": 1.5,
      "multiple_of": 256,
      "n_heads": 32,
      "n_kv_heads": 8,
      "n_layers": 16,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0,
      "use_scaled_rope": True,
      "vocab_size": 128256
    }


    model_args = ModelArgs(
      dim=params["dim"],
      n_layers=params["n_layers"],
      n_heads=params["n_heads"],
      n_kv_heads=params["n_kv_heads"],
      ffn_dim_multiplier=params["ffn_dim_multiplier"],
      multiple_of=params["multiple_of"],
      norm_eps=params["norm_eps"],
      rope_theta=params["rope_theta"],
      use_scaled_rope=params["use_scaled_rope"],
      vocab_size=params["vocab_size"],
      max_batch_size=32,
      max_seq_len=2048,
     )



    model = Transformer(model_args).cuda(rank)
    print(f"Model on GPU {rank}")


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    path_tokenizer = os.path.expanduser("/workspace/work/llama3/Llama3.2-1B/tokenizer.model")
    tokenizer = Tokenizer(path_tokenizer)
    format = ChatFormat(tokenizer)


    train_data = [("Where do you work?", "for Nuflorist")]
    tokenized_data = preprocess_data_with_tokenizer(train_data, tokenizer, format)
    print(tokenized_data)
    criterion = torch.nn.CrossEntropyLoss()
    print(criterion)



    for epoch in range(3):
        print("--------------epoch------------")
        model.train()  # Set the model to training mode
        epoch_loss = 0

        torch.autograd.set_detect_anomaly(True)


        for question_encoded, answer_encoded in tokenized_data:
            start_pos = 0
            question_input = torch.tensor(question_encoded).unsqueeze(0).cuda(rank)
            answer_input = torch.tensor(answer_encoded).unsqueeze(0).cuda(rank)

            #answer_input = answer_input[:, :10]  # Slice answer_input to 10 tokens

            output = model(question_input, start_pos)  # Assuming model.forward
            print(f"Output shape: {output.shape}")
            print(f"Answer input shape: {answer_input.shape}")
            output_len = output.size(1)  # The model's output sequence length (e.g., 10)

            target_len = answer_input.size(1)
            print(output_len, "output_len")
            print(target_len, "target")

            #output = output.contiguous().view(-1, model.vocab_size)  # Flattening for compatibility

            # Target tensor shape should be [batch_size * seq_len] and should not be one-hot encoded
            #loss = criterion(output, answer_input.view(-1))


            #loss = F.cross_entropy(output.view(-1, model.vocab_size), answer_input.view(-1))
            loss = criterion(output.view(-1, model.vocab_size), answer_input.view(-1))
            #optimizer.zero_grad()
            #loss.backward(retain_graph=True)
            optimizer.step()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    world_size = 1  # Total number of GPUs
    mp.spawn(run, args=(world_size,), nprocs=world_size)

