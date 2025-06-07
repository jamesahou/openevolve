#!/bin/bash

set -e

cd /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy

git clean -fd
git reset --hard 96cc7fbe

COORDS_CORE="/Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py"

# Insert 'import funsearch' at the top if not present
if ! grep -q "^import funsearch" "$COORDS_CORE"; then
  sed -i '' '10i\
import funsearch
' "$COORDS_CORE"
fi

# Insert '    @funsearch.evolve' at line 397 (since import line was added at the top)
sed -i '' '398i\
    @funsearch.evolve
' "$COORDS_CORE"

pip install -e .