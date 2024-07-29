# Distibuted Graph Learning Library
Welcome to the Distributed Graph Learning Library.


# VERY IMPORTANT NOTES
Please strictly follow the following guidelines:

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
- Strictly follow Open Source Guidelines.
- The code will undergo a plagiarism check.
- Do not use DGL or any other third-party libraries.
- You may use PyTorch for data loading, neural network operations, etc.
- If you must borrow code, make sure it is marked, properly cited, and attributed to the original author.

## How to use the repository
- Create a virtual environment.
- Fork the repository.
- Exclude virtual environment files/directories using `.gitignore` to avoid them being pushed to GitHub.
- Add the names and versions of any new modules installed using pip to the `requirements.txt` file.
- Please do not commit your changes directly to the main code base. Instead, issue a Pull Request to merge your changes. This allows us to review and discuss the changes before integrating them into the main codebase.
- Every merge request will undergo a thorough review, ensuring code quality, proper documentation, formatting, and functionality before merging. Please consider this process carefully before issuing a merge request, as timely review is crucial.

## Identification and Attribution
- Add your name to the top of each source file you own.
- Also, include your name in the comments or documentation for any changes you make to code owned by others for identification and attribution purposes.