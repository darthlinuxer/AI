source venv/bin/activate
pip install uv

# Install PyTorch packages separately with the correct find-links
pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu torchvision==0.18.0+cpu -f https://download.pytorch.org/whl/torch/ -f https://download.pytorch.org/whl/torchaudio/ -f https://download.pytorch.org/whl/torchvision/

# Install the rest of the requirements
uv pip install -r requirements.txt