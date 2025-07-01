# 💪🧠 Reasoning Gym

**Reasoning Gym** is a community-created Python library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL). The goal is to generate virtually infinite training data with adjustable complexity.

It currently provides **more than 100** tasks over many domains, including but not limited to _algebra_, _arithmetic_, _computation_, _cognition_, _geometry_, _graph theory_, _logic_, and many common _games_.

Some tasks have a single correct answer, while others, such as [Rubik‘s Cube](https://en.wikipedia.org/wiki/Rubik%27s_Cube) and [Countdown](<https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_Round>), have many correct solutions. To support this, we provide a standard interface for procedurally verifying solutions.

## 🖼️ Dataset Gallery

In [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md), you can find example outputs of all datasets available in `reasoning-gym`.

## ⬇️ Installation

The `reasoning-gym` package requires Python >= 3.11.

Install the latest published [package from PyPI](https://pypi.org/project/reasoning-gym/) via `pip`:

```
pip install reasoning-gym
```

_Note that this project is currently under active development, and the version published on PyPI may be a few days behind `main`._

## 🛠️ Development

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md#development-setup).

## 🛠️ Modifications in This Fork

This fork introduces custom evaluation utilities, updated reasoning tasks, and tools for retrieval-augmented workflows and visualization.

### 🔍 `reasoning_gym/`
- **Curriculum Customization**: Modified reasoning curriculum levels to explore and benchmark new task difficulties or capabilities.

### 📊 `eval/`
- **`eval_vllm.py`**: Custom evaluation script using vllm for fast inference.
- **`sbatch.sh`**: SLURM batch submission script to run `eval_vllm.py` in distributed or cluster environments.

### 📈 `notebooks/`
- **`visualization.ipynb`**: Jupyter notebook to visualize:
  - Reasoning Gym performance metrics.
  - Token-to-Token Ratio (TTR) plots for analysis.

### 🔎 `retrieval_websearch/`
- Tools to run web-based retrieval and evaluate model performance using retrieved documents.
- Useful for experiments involving RAG (Retrieval-Augmented Generation) workflows.


