## Coding Standards
- If responding with code that includes a class, function, or method definition, add docstring comments. Docstrings should follow the Google Python Style Guide (including `Args:`, `Returns:`, and `Raises:` sections where applicable).
- Review for possible exceptions and add specific exception handling (e.g., `try...except FileNotFoundError:`), but only for code that is expected to raise exceptions. Avoid catching generic exceptions unless absolutely necessary.
- Strive for descriptive variable and function names. Avoid unclear abbreviations. Common, well-understood abbreviations within the project's domain (e.g., `img` for image, `dir` for directory, `rso` for Resident Space Object, `bg` for background) are acceptable if they enhance readability without sacrificing clarity.
- Constants follow `ALL_CAPS_SNAKE_CASE` (otherwise known as Screaming Snake Case).
- Avoid using magic strings and numbers. Parameterize configurable values or define them as constants. This applies particularly to string literals used as dictionary keys, file naming components, or numerical values for thresholds and kernel sizes unless their meaning is obvious from context.
- Utilize the `pathlib` module for all file system path manipulations to ensure cross-platform compatibility and improve code readability.
- All function and method signatures, as well as variable declarations where appropriate, must include type hints. Use standard Python types and types from the `typing` module.
- For console output, logging, and progress bars, prefer using the `rich` library to ensure consistent and user-friendly command-line interfaces.

###  Code Nesting
- Avoid deeply nested code. Break down logic into smaller functions.
- Use 4 spaces for indentation

### Docstring Comments
- Use triple double quotes for docstring comments.
- Use `"""` for docstring comments. Start the summary at the same line as the opening quotes.
- Ensure you include the `Args:`, `Returns:`, and `Raises:` sections in docstrings as applicable, or all functions and methods.

### Libraries to use
- Use `pathlib` for file and directory paths.
- Use `numpy` for numerical operations and array manipulations.
- Use `rich` for console output, logging, and progress bars.
- Use `typer` for command-line interface (CLI) applications, if applicable.

### Example Code
```python
"""CLI utilities for mock image processing.

This module provides a small command-line interface built with Typer
for (1) applying a mock transformation to an image and (2) retrieving
mock image dimensions.  All I/O is logged with *rich* for a pleasant UX.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
DEFAULT_INTENSITY: float = 0.75
OUTPUT_SUBDIR: str = "processed_images"

# --------------------------------------------------------------------------- #
# Rich console (shared)
# --------------------------------------------------------------------------- #
console: Console = Console()

# --------------------------------------------------------------------------- #
# Image utilities
# --------------------------------------------------------------------------- #
class ImageFileProcessor:
    """Load, (mock-)transform, and save images."""

    def __init__(self, base_input_dir: Path, base_output_dir: Path) -> None:
        """Create processor and make sure directories exist.

        Args:
            base_input_dir: Directory containing source images.
            base_output_dir: Directory where processed images will be stored.

        Raises:
            FileNotFoundError: If *base_input_dir* does not exist.
            OSError: If the output directory cannot be created.
        """
        self.base_input_dir: Path = base_input_dir
        self.output_dir: Path = base_output_dir / OUTPUT_SUBDIR

        if not self.base_input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.base_input_dir}")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print_exception()
            raise OSError(f"Failed to create output directory: {self.output_dir}") from exc

    # ------------------------- internal helpers ----------------------------- #
    def _load(self, path: Path) -> np.ndarray:
        """Mock image loader.

        Args:
            path: Path to the image file.

        Returns:
            A dummy 2 × 2 NumPy array.

        Raises:
            FileNotFoundError: If *path* is missing.
            ValueError: If file suffix is unsupported.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        console.print(f"[green]Loaded[/green] {path.name}")
        return np.array([[10, 20], [30, 40]], dtype=np.uint8)

    def _save(self, data: np.ndarray, path: Path) -> None:
        """Mock image saver (writes plain text).

        Args:
            data: Image array to save.
            path: Target path.

        Raises:
            IOError: If writing fails.
        """
        try:
            with path.open("w") as file:
                file.write(str(data.tolist()))
            console.print(f"[cyan]Saved[/cyan] {path}")
        except IOError as exc:
            console.print_exception()
            raise

    # ---------------------------- public API -------------------------------- #
    def transform(
        self,
        filename: str,
        intensity: float = DEFAULT_INTENSITY,
    ) -> Path:
        """Apply a mock transformation and save result.

        Args:
            filename: Image filename located in *base_input_dir*.
            intensity: Transformation intensity (0–1).

        Returns:
            Path to the saved file.

        Raises:
            FileNotFoundError, ValueError: Propagated from loaders.
            RuntimeError: For unexpected failures.
        """
        input_path: Path = self.base_input_dir / filename
        try:
            img: np.ndarray = self._load(input_path)
            transformed: np.ndarray = np.clip(
                img + int(intensity * 100), 0, 255
            ).astype(np.uint8)

            output_path = self.output_dir / f"transformed_{filename}"
            self._save(transformed, output_path)
            return output_path
        except (FileNotFoundError, ValueError):
            raise  # Re-raise for caller / CLI
        except Exception as exc:  # Any other unforeseen error
            raise RuntimeError(f"Processing failed for {filename}") from exc


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return mock image dimensions (100 × 100).

    Args:
        path: Path to an image file.

    Returns:
        tuple ``(width, height)``.

    Raises:
        FileNotFoundError, ValueError: When *path* is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    console.print(f"Mock size for {path.name}: 100×100")
    return 100, 100


# --------------------------------------------------------------------------- #
# Typer CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=True, help="Mock image utilities CLI")


@app.command()
def process(
    filename: str = typer.Argument(..., help="Image filename inside INPUT_DIR."),
    input_dir: Path = typer.Option(
        ".", "--input-dir", "-i", exists=True, file_okay=False, help="Source directory."
    ),
    output_dir: Path = typer.Option(
        "./out", "--output-dir", "-o", file_okay=False, help="Root directory for results."
    ),
    intensity: float = typer.Option(
        DEFAULT_INTENSITY,
        min=0.0,
        max=1.0,
        help="Mock transformation intensity (0–1).",
    ),
) -> None:
    """Apply a mock transformation to *filename*."""
    try:
        processor = ImageFileProcessor(input_dir, output_dir)
        result_path = processor.transform(filename, intensity)
        console.print(f"[bold green]Done:[/bold green] {result_path}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")


@app.command()
def dimensions(
    image_path: Path = typer.Argument(..., exists=True, help="Path to an image file.")
) -> None:
    """Print mock dimensions of *image_path*."""
    try:
        width, height = get_image_dimensions(image_path)
        console.print(f"{image_path.name}: {width}×{height}")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    app()

```

---

### Error Handling
- Always catch a specific error instead of a generic one.
- Log the error message and stack trace.

## Response Instructions
- If I tell you that you are wrong, think about whether or not you think that's true and respond with facts.
- Avoid apologizing or making conciliatory statements.
- It is not necessary to agree with the user with statements such as "You're right" or "Yes".
- Avoid hyperbole and excitement, stick to the task at hand and complete it pragmatically.
- Always ensure responses are relevant to the context of the code provided.
- Avoid unnecessary detail and keep responses concise.
- Revalidate before responding. Think step by step.

## Tool Instructions
- @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.
- @python Rule - Use Python Best Practices: When generating Python code, follow Ruff rules outlined in pyproject.toml.
- @docker Rule - Use Docker Best Practices: When generating Dockerfiles or Docker Compose files, follow Docker best practices and security guidelines.

## MCP Server Instructions
If the respective server exists, follow these instructions:
- GitHub MCP Server: for any query about repositories, issues, actions, and the like, execute against the GitHub repository: https://github.com/raffertyuy/github-copilot-prompts
