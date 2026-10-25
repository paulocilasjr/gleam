# Repository backup — 2026-10-25

This branch preserves the local hot_fix checkout plus all 25 manuscript documents,
figures, and scripts present when the backup was made. OS metadata and Python/test
caches are omitted because they are regenerated locally. The shared upstream repo
is not changed by this backup.

## Clone on another computer

```sh
git clone --branch codex/backup-2026-10-25 git@github.com:paulocilasjr/gleam.git
cd gleam
```

The manuscript_tables directory is tracked on this branch. Its ignore rule has
been removed here so future manuscript edits can be committed normally.

## Preserved local branches

Each original local branch is backed up under the prefix
codex/backup-2026-10-25-branches/ in the personal fork:

```text
feature/computer_resource c938e9f81bf57b8fac398450345144a263da1d7b
hot_fix fdeb6adf7f83c70e9a05c4ab31a1fa9b258475a2
main 1dca5bb73cb8c56cefe30602cce401b0cd12157d
```

To work on one, for example:

```sh
git switch -c hot_fix origin/codex/backup-2026-10-25-branches/hot_fix
```

## Saved model-selection work

The original Git stash is preserved intact at origin/codex/backup-2026-10-25-stash
(commit 47908378ab4f20548735a59ad8c135ec0f1c31ac), including its original base and staged state.
Cloning does not populate the local stash list automatically. Restore it with:

```sh
git switch -c recovered-model-selection origin/codex/backup-2026-10-25-stash^1
git stash apply --index origin/codex/backup-2026-10-25-stash
```

The original local checkout and stash were left untouched.
