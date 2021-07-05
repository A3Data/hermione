
# Installing


## Dependencies

- Python (>= 3.6)
- **docker**

Hermione does not depend on conda to build and manage virtual environments anymore. It uses `venv` instead.


## Install

```python

pip install -U hermione-ml

```

## Enabling autocompletion (unix users):

For bash:

```bash
echo 'eval "$(_HERMIONE_COMPLETE=source_bash hermione)"' >> ~/.bashrc
```

For Zsh:

```bash
echo 'eval "$(_HERMIONE_COMPLETE=source_zsh hermione)"' >> ~/.zshrc
```