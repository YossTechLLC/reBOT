# Contributing Guide

## Repository Structure (Post-Restructure 2026-01-16)

This repository contains multiple components:

- `GOOGLE/timesfm/` - TimesFM submodule (Google Research fork)
- `commodity-forecasting-system/` - Independent forecasting system
- `docs/` - Project documentation
- Other project directories

## Git Workflow

### TimesFM Submodule

The timesfm submodule uses a **fork-based workflow** to sync with upstream while preserving custom work:

**Branch Strategy:**
- `master` branch - Syncs with google-research/timesfm upstream (read-only for upstream changes)
- Custom work - Create feature branches (e.g., `feature/progress-tracking`)
- Remote setup:
  - `origin` → https://github.com/google-research/timesfm.git (upstream source)
  - `fork` → https://github.com/YossTechLLC/timesfm.git (your fork)

**Creating a New Feature Branch:**

```bash
cd GOOGLE/timesfm
git checkout master
git pull origin master                    # Sync with upstream
git checkout -b feature/your-feature-name # Create new branch from synced master
# Make your changes, commit
git push fork feature/your-feature-name   # Push to your fork
```

**Syncing with Upstream:**

```bash
cd GOOGLE/timesfm
git fetch origin                          # Fetch from google-research/timesfm
git checkout master
git merge origin/master                   # Or use rebase: git rebase origin/master
git push fork master                      # Update your fork's master
```

**Working on Existing Feature Branches:**

```bash
cd GOOGLE/timesfm
git checkout feature/your-feature-name
# Make changes, commit
git push fork feature/your-feature-name
```

**Updating Feature Branch with Latest Upstream:**

```bash
cd GOOGLE/timesfm
git checkout master
git pull origin master                    # Get latest upstream
git checkout feature/your-feature-name
git rebase master                         # Rebase your feature on latest master
# Or use merge if you prefer: git merge master
git push fork feature/your-feature-name --force-with-lease  # Only if rebased
```

### Commodity Forecasting System

The `commodity-forecasting-system/` directory is an independent git repository with its own history:

```bash
cd commodity-forecasting-system
git status                                # Check status
git log                                   # View history
# Normal git workflow applies here
```

This repository tracks the commodity system as a **gitlink** (commit pointer), not as a full submodule. Changes inside `commodity-forecasting-system/` are committed within that repository independently.

### Parent Repository (reBOT)

Standard git workflow for the parent repository:

```bash
# Working on parent repo changes
git status
git add <files>
git commit -m "Your message"
git push origin main                      # Or your default branch
```

**Updating Submodule Pointer:**

When you want the parent repo to point to a different timesfm commit:

```bash
cd GOOGLE/timesfm
git checkout <commit-or-branch>           # Switch to desired commit/branch
cd ../..                                  # Return to parent repo
git add GOOGLE/timesfm                    # Stage the submodule pointer update
git commit -m "Update timesfm submodule to <commit>"
git push origin main
```

## Remote Configuration

### TimesFM Submodule Remotes

Current setup in `GOOGLE/timesfm/`:

```
origin  → https://github.com/google-research/timesfm.git (fetch/push)
fork    → https://github.com/YossTechLLC/timesfm.git (fetch/push)
```

To verify:
```bash
cd GOOGLE/timesfm
git remote -v
```

### Alternative: Rename Remotes for Clarity

If you prefer more semantic naming:

```bash
cd GOOGLE/timesfm
git remote rename origin upstream
git remote rename fork origin
git fetch --all
```

After this change:
- `upstream` → google-research/timesfm (pull from here to sync)
- `origin` → YossTechLLC/timesfm (push here for your work)

Update your workflow commands accordingly (replace `origin` with `upstream` and `fork` with `origin`).

## Key Branches

### TimesFM Submodule

- `master` - Synchronized with google-research/timesfm upstream
- `feature/progress-tracking` - Preserved custom progress tracking commits (ff09b8f, 60b2e00, 822be05)
- `feature/*` - Your feature branches

### Commodity System

- `main` - Main development branch (5 phase commits)

## Best Practices

1. **Never commit directly to timesfm master** - Always create feature branches
2. **Sync regularly** - Pull upstream changes frequently to avoid conflicts
3. **Use descriptive branch names** - e.g., `feature/add-xreg-support`, `fix/inference-memory-leak`
4. **Keep feature branches focused** - One feature or fix per branch
5. **Rebase or merge from master** - Keep your feature branches up to date with upstream

## Backup and Recovery

Backup files created during restructure (2026-01-16):
- `~/timesfm_local_commits.patch` - Patch file of 3 original commits
- `~/timesfm_commit_log.txt` - Commit metadata

To restore from backup (if needed):
```bash
cd GOOGLE/timesfm
git am < ~/timesfm_local_commits.patch
```

## Useful Commands

**Check submodule status:**
```bash
git submodule status
```

**Update all submodules to latest commit:**
```bash
git submodule update --remote
```

**Clone this repository with submodules:**
```bash
git clone --recursive <repo-url>
# Or if already cloned:
git submodule update --init --recursive
```

**View current submodule configuration:**
```bash
cat .gitmodules
```

## Questions?

- TimesFM-specific development: See `CLAUDE.md` in repository root
- Parent repository structure: See `docs/MAP.md`
- Feature implementation progress: See `CHECKLIST.md`

## Changelog

- **2026-01-16**: Repository restructured
  - Created YossTechLLC/timesfm fork
  - Moved commodity-forecasting-system to parent repo
  - Established fork-based workflow for timesfm submodule
  - Preserved custom work in feature/progress-tracking branch
