# Ar5 - Tail Free DLC Analysis Using MDAnalysis

By Alfredo Roman Jordan  
Built on foundation of Sn5 by Dr. Mitch Powers

---

## Overview

**Ar5** is a Python package for analyzing *Tail-Free Discotic Liquid Crystals* (DLC) from molecular dynamics trajectories using **MDAnalysis**.  

It provides tools to:

Detect column hopping events  
Analyze fragment trajectories  
Compute orientational order parameters  
Combine XYZ frames for visualization  
Convert `.dyn` files to `.xyz` format  
Standardize atom labels  

---

## Design Philosophy

- The user is expected to load files and universes in their own analysis environment (scripts / notebooks).
- **File input is intentionally flexible** — functions operate on **lists of directories** or **lists of file paths**, not hardcoded internal traversal.
- Functions consistently support a `verbose` argument to control logging and visualization.
- The code is designed to be as **backward compatible** and **broad** as possible for future expansion.

---

## Installation

```bash
git clone https://github.com/Powers-Lab-2025/Ar5.git
cd Ar5
pip install -e .
