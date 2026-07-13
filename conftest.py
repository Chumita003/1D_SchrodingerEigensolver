# Empty on purpose. Its only job is to mark this directory as pytest's root, so pytest
# adds it to sys.path before collecting tests. Without this, tests/test_eigensolver.py
# (which does `from Eigensolver_1Dimension import ...`) can't find that module, because
# pytest only auto-adds a test file's own directory to sys.path, not the repo root.
