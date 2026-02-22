# Project Mandates

## Engineering Standards
- **Code Preservation:** Never erase or overwrite existing code, functions, or logic unless specifically instructed to do so by the user. When modifying files, always prioritize appending or surgically updating rather than replacing large blocks of established code.
- **Context Awareness:** Always refer to `constants.py` for column names and tags to ensure consistency across the pipeline.
- **Testing:** When adding features to `load_data.py`, ensure the `preprocessing` function remains compatible with the existing LightGBM training loop.
