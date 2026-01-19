Collective Noise Filtering in Complex Networks
==============================================

This repository contains the code and data associated with the manuscript:

**Tingyu Zhao, István A. Kovács**, *Collective Noise Filtering in Complex Networks.*

---

## Overview

We introduce the **Network Wiener Filter (NetWF)**, a principled method for collective edge noise filtering in complex networks that jointly utilizes network topology and noise characteristics. The framework is applicable to binary, weighted, signed, and directed networks, and is designed to scale to large empirical systems.

This repository provides:
- Core algorithmic implementations of NetWF
- Example applications to two real-world network datasets

---

## Repository Structure

### Core algorithms
- **`utils_WF.py`**  
  Core implementation of the NetWF, including:
  - Profile similarity computation
  - Iterative NetWF algorithm
  - Direct NetWF algorithm

### Data folders
- **`GI-data/`**  
  Contains data for the genetic interaction (GI) network of the yeast *Saccharomyces cerevisiae* as well as benchmark data for evaluation.
- **`Enron-data/`**  
  Contains data for the Enron Corpus email network.

### Notebooks
- **`GI.ipynb`**  
  Demonstration notebook applying NetWF to the GI network.
- **`Enron.ipynb`**  
  Demonstration notebook applying NetWF to the Enron network.  

### Supporting utilities
- **`utils_GI.py`**  
  Helper functions for the GI network analysis and visualization.
- **`utils_Enron.py`**  
  Helper functions for the Enron network analysis and visualization.

---

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
