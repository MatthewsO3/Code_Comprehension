

# Setup script for GraphCodeBERT training on C++

echo "Setting up environment for GraphCodeBERT C++ MLM training..."

# Install Python dependencies
pip install torch torchvision torchaudio
pip install transformers==4.30.0
pip install datasets
pip install tree-sitter==0.20.1
pip install tree-sitter-cpp
pip install pandas
pip install tqdm
pip install accelerate

# Clone CodeBERT repository for reference
if [ ! -d "CodeBERT" ]; then
    echo "Cloning CodeBERT repository..."
    git clone https://github.com/microsoft/CodeBERT.git
fi

# Build tree-sitter C++ language
python -c "
from tree_sitter import Language
Language.build_library(
    'build/my-languages.so',
    ['tree-sitter-cpp']
)
"

echo "Setup complete!"
echo "Next steps:"
echo "1. Run: python 2_extract_dfg.py"
echo "2. Run: python 3_train_mlm.py"