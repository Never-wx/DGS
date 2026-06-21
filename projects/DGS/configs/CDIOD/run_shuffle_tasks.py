#!/usr/bin/env python3
"""
Shuffle datasets with reproducible random seed.
Usage: python shuffle_datasets.py <seed> <dataset1> <dataset2> ...
Output: Space-separated shuffled dataset list
"""
import sys
import random
import argparse

def shuffle_datasets(datasets, seed, verbose=False):
    """
    Shuffle datasets with given seed.
    
    Args:
        datasets: List of dataset names
        seed: Random seed for reproducibility
        verbose: If True, print detailed information
    
    Returns:
        Shuffled list of datasets
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create a copy to avoid modifying the original list
    shuffled = datasets.copy()
    random.shuffle(shuffled)
    
    if verbose:
        print("==========================================", file=sys.stderr)
        print(f"Shuffled dataset order (SEED={seed}):", file=sys.stderr)
        for idx, dataset in enumerate(shuffled):
            print(f"  Phase {idx}: {dataset}", file=sys.stderr)
        print("==========================================", file=sys.stderr)
    
    return shuffled


def main():
    parser = argparse.ArgumentParser(
        description='Shuffle datasets with reproducible random seed'
    )
    parser.add_argument(
        'seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        'datasets',
        nargs='+',
        help='List of dataset names to shuffle'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed shuffle information to stderr'
    )
    
    args = parser.parse_args()
    
    # Shuffle datasets
    shuffled = shuffle_datasets(args.datasets, args.seed, args.verbose)
    
    # Output space-separated list to stdout (for bash to capture)
    print(' '.join(shuffled))


if __name__ == '__main__':
    main()