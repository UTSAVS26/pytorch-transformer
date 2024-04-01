from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb

import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Define start-of-sequence and end-of-sequence token indices
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every decoding step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the start-of-sequence token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ],
            dim = 1
        )

        # Break if the end-of-sequence token is predicted
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    # Set the model to evaluation mode
    model.eval()
    count = 0

    # Lists to store source texts, expected target texts, and predicted target texts
    source_texts = []
    expected = []
    predicted = []

    try:
        # Get the console window width for formatting output
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If console width retrieval fails, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # Check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate model output using greedy decoding
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Extract source, target, and model output texts
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Append texts to respective lists
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print source, target, and model output texts
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Break after printing specified number of examples
            if count == num_examples:
                print_msg('-' * console_width)
                break
    
    # Evaluate metrics: Character Error Rate (CER), Word Error Rate (WER), BLEU Score
    # Compute Character Error Rate (CER)
    cer_metric = torchmetrics.CharErrorRate()
    cer = cer_metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute Word Error Rate (WER)
    wer_metric = torchmetrics.WordErrorRate()
    wer = wer_metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute BLEU Score
    bleu_metric = torchmetrics.BLEUScore()
    bleu = bleu_metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds, lang):
    # Iterate over the dataset and yield sentences in the specified language
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # Define the path to store the tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # Check if tokenizer file exists
    if not Path.exists(tokenizer_path):
        # If tokenizer file does not exist, build a new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))  # Initialize tokenizer with WordLevel model
        tokenizer.pre_tokenizer = Whitespace()  # Set pre-tokenizer to split on whitespaces
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)  # Define trainer with special tokens and minimum frequency
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)  # Train tokenizer from sentences in the dataset
        tokenizer.save(str(tokenizer_path))  # Save tokenizer to file
    else:
        # If tokenizer file exists, load tokenizer from file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    # Load dataset
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split = 'train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset into training and validation sets (90% training, 10% validation)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Wrap datasets with BilingualDataset class
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find maximum length of source and target sentences
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # Print maximum lengths of source and target sentences
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Build and return the model
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model = config['d_model'])
    return model

def train_model(config):
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    # Get data loaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Get the model and move it to the appropriate device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    # Load pretrained weights if specified
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    # Define custom metrics for logging
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric = "global_step")
    wandb.define_metric("train/*", step_metric = "global_step")

    # Iterate over epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # Move batch to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            },
            model_filename
        )

if __name__ == '__main__':
    # Ignore warnings during execution
    warnings.filterwarnings("ignore")
    
    # Load configuration settings
    config = get_config()
    
    # Update number of epochs and preload option in the configuration
    config['num_epochs'] = 30
    config['preload'] = None

    # Initialize wandb run to log metrics and hyperparameters
    wandb.init(
        project = "pytorch-transformer",  # Set the wandb project where this run will be logged
        config = config  # Track hyperparameters and run metadata
    )
    
    # Train the model with the specified configuration
    train_model(config)