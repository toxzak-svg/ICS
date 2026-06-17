# Isomorphic Channel Sorting (ICS) Prototype

This repository contains the prototype for the Isomorphic Channel Sorting (ICS) framework, which reorders neural network weights offline to eliminate memory divergence bottlenecks on mobile NPUs.

## Overview
- **Goal**: Replace runtime weight patching with compile-time topological sorting.
- **Key Idea**: Permute weight matrix rows based on Fisher Information or magnitude, making the model a single contiguous memory block.
- **Benefits**: Single memory read, variable-bit packing, drop-in robustness on CoreML, QNN, NNAPI.

## Directory Structure
```
ics_prototype.ipynb      # Jupyter notebook with demo
README.md                # This file
.gitignore               # Files to ignore
requirements.txt         # Python dependencies
setup.sh                 # Optional script to set up environment
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook ics_prototype.ipynb`
3. Follow the steps to apply the permutation and evaluate.

## License
MIT License

## Contributing
Feel free to open issues or submit pull requests.
```
