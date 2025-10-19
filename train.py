import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from Transformer_from_scratch import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
import os
from pathlib import Path
import warnings
import torchmetrics
import json
import psutil
import time



import psutil
import torch
import csv
import time
from pathlib import Path
import threading
import pandas as pd



class ResourceLogger:
    def __init__(self, interval=1.0):
        cnfg = get_config()
        self.log_dir = Path(cnfg["resource_logging"])
        self.interval = interval
        self.running = False
        self.thread = None
        self.current_epoch = -1
        self.lock = threading.Lock()
        self.start_time = time.time()

        self.file_sec = self.log_dir / "usage_seconds.csv"
        self.file_epoch = self.log_dir / "usage_epochs.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.file_sec.exists():
            with open(self.file_sec, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "cpu_percent", "ram_gb", "gpu_gb", "gpu_total_gb", "epoch_marker"])
                
        if not self.file_epoch.exists():
            with open(self.file_epoch, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "time_abs", "cpu_percent", "ram_gb", "gpu_gb", "gpu_total_gb"])

    def _get_gpu_stats(self):
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return gpu_mem, gpu_total
            except Exception:
                return 0.0, 0.0
        return 0.0, 0.0


    def mark_epoch(self, epoch):
        with self.lock:
            self.current_epoch = int(epoch)
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().used / 1024**3
        gpu, gpu_total = self._get_gpu_stats()

        # --- Append row to per-epoch log file ---
        try:
            with open(self.file_epoch, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, time.time(), cpu, ram, gpu, gpu_total])
        except Exception:
            pass

    def _run(self):
        while self.running:
            # --- Append new data point ---
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().used / 1024**3
            gpu, gpu_total = self._get_gpu_stats()
            current_time = time.time() 
            
            with self.lock:
                marker = f"epoch_{self.current_epoch}" if self.current_epoch >= 0 else ""

            try:
                with open(self.file_sec, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([current_time, cpu, ram, gpu, gpu_total, marker])
            except Exception:
                pass
            
            time.sleep(self.interval)

    def start(self):
        # --- Clear and initialize files ---
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try: 
            with open(self.file_sec, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "cpu_percent", "ram_gb", "gpu_gb", "gpu_total_gb", "epoch_marker"])
        except Exception:
            pass

        try:
            with open(self.file_epoch, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "time_abs", "cpu_percent", "ram_gb", "gpu_gb", "gpu_total_gb"])
        except Exception:
            pass


        # --- reset counter ---
        with self.lock:
            self.current_epoch = -1

        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None


EVAL_RESULTS_DIR = "eval_results"

def save_eval_metrics(epoch, bleu, cer, wer, samples, config):
    results_file = Path(config.get("eval_results_file", "eval_results/eval_metrics.json"))
    Path(EVAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "epoch": epoch,
        "bleu": float(bleu),
        "cer": float(cer),
        "wer": float(wer),
        "samples": samples
    }
   
    # At the beginning of a run (epoch 0) clear the file and write only this entry.
    if int(epoch) == 0:
        all_results = [entry]
    else:
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    all_results = json.load(f)
            except Exception:
                all_results = []
        else:
            all_results = []

    # replace any existing entry for this epoch, then append and sort
    all_results = [r for r in all_results if int(r.get("epoch", -1)) != int(epoch)]
    all_results.append(entry)
    all_results.sort(key=lambda x: int(x["epoch"]))

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calcu output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model,
                   validation_ds,
                   tokenizer_src, tokenizer_tgt,
                   max_len, print_msg, global_step,
                   writer, device, epoch, config, num_examples=2):

    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    samples = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, 
                                      tokenizer_src, tokenizer_tgt,
                                      max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Save these sample sentences as performance data
            samples.append({"source": source_text, "target": target_text, "predicted": model_out_text})

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # --- Evaluate the character error rate ---
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # --- Compute the word error rate ---
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # --- Tokenize input for BLEU ---
        def tokenize_bleu(s):
            s = s.replace('[SOS]', '').replace('[EOS]', '').replace('[PAD]', '')
            s = s.strip()
            return s.split() if s != "" else []

        predicted_tokens = [tokenize_bleu(p) for p in predicted]
        expected_tokens = [[tokenize_bleu(t)] for t in expected]

        # torchmetrics.BLEUScore expects strings by default (it will tokenize them),
        # so join token lists back into space-separated strings.
        predicted_strs = [" ".join(toks) for toks in predicted_tokens]
        expected_strs = [[" ".join(toks) for toks in refs] for refs in expected_tokens]

        # --- Compute the BLEU metric ---
        metric = torchmetrics.BLEUScore(n_gram=4)
        bleu = metric(predicted_strs, predicted_strs)
        writer.add_scalar('validation BLEU', float(bleu), global_step)
        writer.flush()
        
        # --- Save eval metrics as performance data ---
        save_eval_metrics(epoch, bleu, cer, wer, samples, config)


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config, max_samples=5000):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Filter out sentences that are too long
    ds_raw = ds_raw.filter(lambda x: len(tokenizer_src.encode(x['translation'][config['lang_src']]).ids) <= config['seq_len'] - 2 and \
                                     len(tokenizer_tgt.encode(x['translation'][config['lang_tgt']]).ids) <= config['seq_len'] - 1)

    if max_samples is not None:
      ds_raw = ds_raw.select(range(min(max_samples, len(ds_raw))))


    # Keep 90% of data for training and rest for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validate_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt =  0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validate_dataloader = DataLoader(validate_ds, batch_size = 1, shuffle=True)

    return train_dataloader, validate_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    torch.cuda.empty_cache()
    logger = ResourceLogger(interval=1.0)
    logger.start()

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        logger.mark_epoch(epoch)
     
        # Validation and metric saving
        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}')
        run_validation(model, val_dataloader,
                        tokenizer_src, tokenizer_tgt,
                        config['seq_len'], lambda msg: batch_iterator.write(msg), global_step, writer, device, epoch, config)

        model.train()


        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # Log the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)

            # Backpropagate the loss
            loss.backward()

            #Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        #save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    
    logger.stop()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
