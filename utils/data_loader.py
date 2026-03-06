# utils/data_loader.py

"""
Data loading and preprocessing for CIFAR-10 and Penn Treebank.
Implements exact preprocessing described in the paper.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import urllib.request


# ============================================================
# CIFAR-10
# ============================================================

def get_cifar10_loaders(data_dir="./data/cifar10",
                         batch_size=128,
                         val_size=5000,
                         num_workers=4):
    """
    Load CIFAR-10 with exact preprocessing from the paper:
        1. ZCA whitening of all images
        2. Upsample to 40x40, random crop to 32x32
        3. Random horizontal flip

    Train split: 45,000 samples
    Val split:   5,000 samples (randomly sampled from train)
    Test split:  10,000 samples

    Returns:
        train_loader, val_loader, test_loader
    """

    # Step 1: Compute ZCA whitening parameters on raw training data
    # (download first without transforms)
    raw_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    raw_loader = DataLoader(raw_train, batch_size=len(raw_train), shuffle=False)
    all_data, _ = next(iter(raw_loader))

    # Flatten: [N, C*H*W]
    N, C, H, W = all_data.shape
    flat = all_data.view(N, -1).numpy()

    # ZCA whitening
    zca_matrix, zca_mean = compute_zca_whitening(flat)

    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        ZCAWhitening(zca_matrix, zca_mean, (C, H, W)),
        transforms.ToPILImage(),
        transforms.Resize(40),                         # Upsample to 40x40
        transforms.RandomCrop(32),                     # Random crop to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Test transforms (only whitening, no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        ZCAWhitening(zca_matrix, zca_mean, (C, H, W)),
    ])

    # Load with transforms
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False,
        transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=test_transform
    )

    # Split train into train (45k) and val (5k)
    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"CIFAR-10: {len(train_subset)} train | "
          f"{len(val_subset)} val | {len(test_dataset)} test")
    return train_loader, val_loader, test_loader


def compute_zca_whitening(X, epsilon=1e-5):
    """
    Compute ZCA whitening matrix and mean.

    Args:
        X: numpy array [N, D]
        epsilon: regularization

    Returns:
        zca_matrix: [D, D]
        mean: [D]
    """
    mean = X.mean(axis=0)
    X_centered = X - mean

    cov = np.dot(X_centered.T, X_centered) / X_centered.shape[0]

    U, S, _ = np.linalg.svd(cov)
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    return zca_matrix.astype(np.float32), mean.astype(np.float32)


class ZCAWhitening:
    """
    Torchvision transform that applies ZCA whitening to an image tensor.
    """

    def __init__(self, zca_matrix, mean, image_shape):
        self.zca_matrix = torch.tensor(zca_matrix, dtype=torch.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.image_shape = image_shape  # (C, H, W)

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: [C, H, W] float tensor

        Returns:
            whitened: [C, H, W] float tensor
        """
        C, H, W = self.image_shape
        flat = img_tensor.view(-1) - self.mean
        whitened = torch.mv(self.zca_matrix, flat)
        return whitened.view(C, H, W)


# ============================================================
# Penn Treebank
# ============================================================

PTB_URLS = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "test":  "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
}


def download_ptb(data_dir="./data/ptb"):
    """Download Penn Treebank dataset files if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filenames = {
        "train": "ptb.train.txt",
        "valid": "ptb.valid.txt",
        "test":  "ptb.test.txt"
    }
    for split, filename in filenames.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            url = PTB_URLS[split]
            print(f"Downloading PTB {split} from {url}...")
            urllib.request.urlretrieve(url, path)
            print(f"Saved to {path}")
        else:
            print(f"PTB {split} already exists at {path}")


def build_vocab(train_path):
    """
    Build vocabulary from PTB training file.
    Standard PTB preprocessing: 10,000 word vocabulary.

    Returns:
        word2idx: dict {word: int}
        idx2word: dict {int: word}
    """
    words = []
    with open(train_path, "r") as f:
        for line in f:
            words += line.strip().split() + ["<eos>"]

    vocab = sorted(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"Vocabulary size: {len(vocab)}")
    return word2idx, idx2word


def tokenize_ptb(file_path, word2idx):
    """
    Tokenize PTB text file to integer indices.

    Args:
        file_path: path to ptb.*.txt
        word2idx: vocabulary dict

    Returns:
        tokens: list of int
    """
    tokens = []
    with open(file_path, "r") as f:
        for line in f:
            words = line.strip().split() + ["<eos>"]
            for w in words:
                tokens.append(word2idx.get(w, word2idx.get("<unk>", 0)))
    return tokens


def batchify_ptb(tokens, batch_size):
    """
    Reshape token sequence into [batch_size, seq_len] tensor.

    Args:
        tokens: list of int
        batch_size: number of parallel sequences

    Returns:
        data: LongTensor [batch_size, seq_len]
    """
    data = torch.tensor(tokens, dtype=torch.long)
    num_steps = data.size(0) // batch_size
    data = data[:num_steps * batch_size]
    data = data.view(batch_size, -1)
    return data


def get_ptb_data(data_dir="./data/ptb", batch_size=20, eval_batch_size=20):
    """
    Full PTB data pipeline: download, tokenize, batchify.

    Returns:
        train_data: [batch_size, seq_len] LongTensor
        val_data:   [eval_batch_size, seq_len] LongTensor
        test_data:  [eval_batch_size, seq_len] LongTensor
        vocab_size: int (10,000)
        word2idx: dict
    """
    download_ptb(data_dir)

    train_path = os.path.join(data_dir, "ptb.train.txt")
    val_path   = os.path.join(data_dir, "ptb.valid.txt")
    test_path  = os.path.join(data_dir, "ptb.test.txt")

    word2idx, idx2word = build_vocab(train_path)

    train_tokens = tokenize_ptb(train_path, word2idx)
    val_tokens   = tokenize_ptb(val_path, word2idx)
    test_tokens  = tokenize_ptb(test_path, word2idx)

    train_data = batchify_ptb(train_tokens, batch_size)
    val_data   = batchify_ptb(val_tokens, eval_batch_size)
    test_data  = batchify_ptb(test_tokens, eval_batch_size)

    print(f"PTB Train: {train_data.shape} | "
          f"Val: {val_data.shape} | Test: {test_data.shape}")

    return train_data, val_data, test_data, len(word2idx), word2idx


# ============================================================
# Character-level PTB
# ============================================================

def get_ptb_char_data(data_dir="./data/ptb", batch_size=32):
    """
    Character-level PTB data pipeline.

    Returns:
        train_data, val_data, test_data: [batch_size, seq_len] LongTensor
        vocab_size: number of unique characters
        char2idx: dict
    """
    download_ptb(data_dir)

    def read_chars(path):
        with open(path, "r") as f:
            return list(f.read())

    train_chars = read_chars(os.path.join(data_dir, "ptb.train.txt"))
    val_chars   = read_chars(os.path.join(data_dir, "ptb.valid.txt"))
    test_chars  = read_chars(os.path.join(data_dir, "ptb.test.txt"))

    all_chars = sorted(set(train_chars))
    char2idx = {c: i for i, c in enumerate(all_chars)}

    def encode(chars):
        return [char2idx[c] for c in chars if c in char2idx]

    train_data = batchify_ptb(encode(train_chars), batch_size)
    val_data   = batchify_ptb(encode(val_chars), batch_size)
    test_data  = batchify_ptb(encode(test_chars), batch_size)

    print(f"PTB Char vocab size: {len(char2idx)}")
    print(f"Train: {train_data.shape} | Val: {val_data.shape} | Test: {test_data.shape}")

    return train_data, val_data, test_data, len(char2idx), char2idx