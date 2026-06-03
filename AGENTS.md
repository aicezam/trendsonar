1.  **Code Organization**:
    -   **Tools/Utils**: All utility functions must be placed in the `app/utils` directory.
    -   **Services**: Business logic should be encapsulated in services under `app/services`.
2.  **Language & Comments**:
    -   **Comments**: All code comments must be written in **Chinese (Simplified)**. 代码中的注释语言，使用中文。
    -   **Responses**: Always interact with the user in Chinese.
3.  **Environment**:
    -   The system is Windows and Debain 13. Ensure file paths are handled correctly (use `os.path.join` or `pathlib`).
4.  **Best Practices**:
    -   Use Type Hints for all function arguments and return values.
    -   Prefer `async/await` for I/O bound operations.
    -   Implement all functional modules using modular programming principles and avoid excessive coupling.
    -   Use dependency injection to reduce coupling between components.
    -   Follow the Single Responsibility Principle (SRP) for classes and functions.
    -   Use Chinese comments at the beginning of classes and functions to explain inputs, outputs, purpose and basic logic.
    -   Each file should start with a comment explaining the purpose of the file.
