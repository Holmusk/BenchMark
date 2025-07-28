#!/bin/sh
set -e

# Clone the repo if not already present
if [ ! -d Bench ]; then
  git clone https://github.com/Holmusk/BenchMark.git Bench
fi
cd Bench

# Set up virtual environment
python3 -m venv venv
. venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make CLI wrapper executable
chmod +x bench

# Optionally symlink to ~/.local/bin for global access (if ~/.local/bin is in PATH)
mkdir -p ~/.local/bin
ln -sf "$PWD/bench" ~/.local/bin/bench

echo "Bench installed. Running benchmark using config.yml in current directory..."
./bench 