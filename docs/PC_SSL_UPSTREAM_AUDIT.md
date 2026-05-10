# PC-SSL Upstream Audit

Upstream repository checked:

```text
https://github.com/Niki-sh/PC-SSL
commit f59f2f5c3bd13396fe838dd6d0e7985642183642
```

The README describes a `src/models/` package:

```text
src/models/__init__.py
src/models/attention.py
src/models/predictive_coding.py
src/models/base.py
```

The actual upstream tree at the checked commit does not contain `src/models/`.
This is the exact reason the local replication needed reconstructed model
files before `scripts/train_model.py` could run.

Verification commands:

```bash
cd PC-SSL
git ls-tree -r --name-only origin/main | sort
git show origin/main:src/models/predictive_coding.py
```

The second command fails with:

```text
fatal: path 'src/models/predictive_coding.py' exists on disk, but not in 'origin/main'
```

Interpretation:

- The README is useful as an architecture description.
- The public repository tree does not ship the model implementation referenced
  by the README and training script.
- Any PC-SSL reproduction from this repo must either obtain the missing author
  model files or reconstruct them from the paper/README.

