# Distibuted Graph Learning Library
Welcome to the Distributed Graph Learning Library.

# VERY IMPORTANT NOTES

## Backend
- Do not use `import torch` directly.
- Instead, use `from dgll import backend as F`.
- In your code, refer to PyTorch functions using `F.` (e.g., `F.tensor`).
- PyTorch is integrated as a backend in the dgll library to ensure consistency and support for multiple backends.

## Documentation
- Pay close attention to source code formatting and adhere to coding standards.
- Avoid spaghetti code; ensure your code is clean and maintainable.
- Use Object-Oriented Design principles.
- Ensure that all User Api functions and classes are properly documented.
- Final documentation will be generated from the source code.
- For sample documentation, refer to dgll/data/dgraph.py.

## Modularity
- This library is designed to be modular. Maintain proper modularity in your code.
- Ensure any new source files are placed in their appropriate locations, are accessible to other modules, and are properly named, formatted, and documented.

## Open-Source Guidelines
- The code will undergo a plagiarism check.
- Strictly follow Open Source Guidelines.
- Do not use DGL or any other third-party libraries.
- You may use PyTorch for data loading, neural network operations, etc.
- If you must borrow code, make sure it is marked, properly cited, and attributed to the original author.