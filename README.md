

# Glyph OCR - Extract unfamiliar characters using Optical Character Recognition

A Python based OCR tool designed for documents containing unfamiliar characters, glyphs, or fonts that conventional OCR libraries and AI powered text extraction tools may struggle to recognise accurately.

Instead of relying on a pre-trained OCR model, **Glyph OCR allows users to build their own character recognition library by manually teaching it glyphs directly from the source document.** It then uses image-based similarity matching to identify characters throughout the document.

The goal is to provide a practical approach to extracting text from difficult or unfamiliar character sets where existing OCR solutions may produce unreliable results.

## Display

<img width="1552" height="965" alt="snapshot" src="https://github.com/user-attachments/assets/f3644314-32db-484b-b386-7fb19948e045" />


## Features

- **Train-by-example OCR:** Teach the program characters by selecting glyphs directly from source documents.
- **Custom glyph library:** Save trained character samples for reuse across documents.
- **PDF and image support:** Process PDF documents and common image formats.
- **Interactive training GUI:** Select glyphs using a mouse and assign their corresponding characters.
- **Image-based recognition:** Compare normalised glyph images against previously trained samples.
- **Character similarity matching:** Use image similarity and aspect ratio to identify likely matches.
- **Unknown character detection:** Characters that do not meet the recognition threshold are marked as `?`.
- **Batch processing:** Process multiple documents and selected pages.
- **Recognition overlays:** Visually inspect recognised and unknown glyphs.
- **Adjustable recognition parameters:** Control matching thresholds, glyph separation, and character segmentation.
- **Export recognised text:** Save extracted text to a UTF-8 encoded text file.

## Why I Built This

Traditional OCR tools and AI based text extraction systems can struggle with unfamiliar writing systems, unusual fonts, obscure symbols, damaged text, or characters that are not well represented in their training data.

This project explores an alternative approach: rather than expecting an OCR engine to recognise everything automatically, the user provides examples of the characters that need to be recognised.

By building a reference library directly from the source material, the program can perform character recognition using visual similarity without requiring a pre-trained OCR model for the target character set.

This is particularly useful when working with material where the characters are visually available but difficult to extract accurately using conventional tools.

## How It Works

### 1. Load a document

Load one or more PDF files or images containing the text you want to recognise.

Supported input formats include:

- PDF
- PNG
- JPG / JPEG
- BMP
- TIFF

PDF pages are rendered into images for processing.

### 2. Train the glyph library

Using the graphical interface, select a character by dragging a box around it.

Enter the corresponding character or symbol when prompted.

The selected glyph is:

1. Extracted from the source image.
2. Cropped to its visible ink.
3. Normalised into a square image.
4. Resized to a standard resolution.
5. Stored alongside its assigned label and aspect ratio.

The process can be repeated to build a library of reference characters.

### 3. Recognise characters

The OCR process analyses the document image and:

- Detects connected components.
- Groups glyphs into text lines.
- Identifies potential character regions.
- Normalises detected glyphs.
- Compares them against the trained reference library.
- Assigns the closest matching character when it falls within the configured threshold.

Characters that cannot be confidently matched are marked as `?`.

### 4. Review the output

The recognised text can be displayed directly or exported to a text file.

Optional debug images can be generated with recognition bounding boxes to help inspect the results and identify incorrectly recognised or unknown characters.

## Technologies Used

- **Python**
- **OpenCV** – Image processing, thresholding, segmentation, glyph normalisation and visual comparison.
- **NumPy** – Numerical operations, array manipulation and similarity calculations.
- **PyMuPDF** – Rendering PDF pages into images.
- **Tkinter** – Interactive glyph-training GUI.
- **Pillow** – Image display within the GUI.
- **argparse** – Command-line interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/William2716057/ocr_unknown.git
cd ocr_unknown
```

Install the required dependencies:

```bash
pip install opencv-python numpy pymupdf pillow
```

Tkinter is also required for the graphical training interface. It is included with many standard Python installations, although some Linux distributions require it to be installed separately.

## Usage

The application supports both interactive operation and command-line usage.

### Interactive Mode

Run the script without command-line arguments:

```bash
python glyph_ocr.py
```

The program will prompt you to:

1. Choose between training a glyph library or performing OCR.
2. Select PDF or image files.
3. Specify the glyph library.
4. Configure recognition settings.
5. Select additional OCR options where applicable.

A file-selection dialog is provided when Tkinter is available.

### Training Mode

Start the training interface with one or more source files:

```bash
python glyph_ocr.py train document.pdf
```

Multiple documents can be provided:

```bash
python glyph_ocr.py train document1.pdf document2.png
```

Specify a custom glyph library:

```bash
python glyph_ocr.py train document.pdf --lib custom_library.npz
```

### Training Workflow

1. Navigate through the source pages.
2. Zoom in or out to make glyph selection easier.
3. Drag a box around a character.
4. Enter the character's correct label.
5. Repeat for other characters and variations.
6. Save the trained glyph library automatically.

The interface also provides page navigation, zoom controls, fit-to-width, undo, and the ability to display recognition results.

### OCR Mode

Run recognition against a document using an existing trained glyph library:

```bash
python glyph_ocr.py ocr document.pdf
```

Specify a glyph library:

```bash
python glyph_ocr.py ocr document.pdf --lib glyph_library.npz
```

Save the recognised text to a file:

```bash
python glyph_ocr.py ocr document.pdf --out output.txt
```

Process specific pages:

```bash
python glyph_ocr.py ocr document.pdf --pages 1-3,7
```

This example processes pages 1, 2, 3 and 7.

Process multiple files:

```bash
python glyph_ocr.py ocr document1.pdf document2.png --out output.txt
```

## Command-Line Options

| Option | Description | Default |
|---|---|---|
| `--lib` | Glyph sample library file | `glyph_library.npz` |
| `--dpi` | PDF rendering resolution | `200` |
| `--max-dist` | Maximum acceptable glyph matching distance | `6.0` |
| `--dilate` | Morphological dilation passes used to help join broken strokes | `1` |
| `--pages` | Pages to process, such as `1-3,7` | All pages |
| `--out` | Output text file | Standard output |
| `--debug` | Generate annotated debug images | Disabled |
| `--gap-factor` | Insert spaces based on the gap between glyphs | `0.0` |

## Recognition Method

Each glyph is converted into a standardised image representation before comparison.

### Glyph Normalisation

The selected or detected glyph is:

- Cropped to its bounding box.
- Centred within a square canvas.
- Resized to `32 × 32` pixels.
- Converted into a normalised image feature representation.

The original glyph aspect ratio is also retained.

### Similarity Matching

The program compares each detected glyph against the stored reference samples using a distance calculation combining:

- Normalised image pixel similarity.
- Aspect ratio difference.

The closest matching sample is selected if its distance is within the configured maximum threshold. Otherwise, the character is classified as unknown.

This approach allows the user to build a character recognition library incrementally using examples from the actual documents being processed.

## Training Library

Trained glyph samples are stored in a compressed NumPy archive (`.npz`).

The library contains:

- Normalised glyph images.
- Glyph aspect ratios.
- Character labels.

This allows the trained library to be reused without having to retrain the same characters every time.

## Debugging and Visual Inspection

Debug mode can generate annotated images for each processed page:

```bash
python glyph_ocr.py ocr document.pdf --debug
```

Recognised characters are marked with bounding boxes, while unknown characters are highlighted differently.

Debug images are saved in a `debug/` directory, allowing visual inspection of the segmentation and recognition process.

## Limitations

This is an experimental OCR approach and is not intended to replace established OCR systems for general-purpose document recognition.

Its performance depends on:

- The quality and resolution of the source document.
- Correct glyph selection during training.
- Similarity between trained glyphs and characters in the document.
- Character segmentation accuracy.
- Font consistency.
- Spacing and layout of the text.

Characters with similar visual shapes may be confused, while heavily distorted, overlapping, or incorrectly segmented glyphs may not be recognised reliably.

The tool currently focuses on character-level recognition rather than advanced document layout analysis, linguistic context, or semantic correction.

## Future Development

Potential improvements include:

- Improved character segmentation.
- Better handling of connected and overlapping glyphs.
- Support for more complex document layouts.
- Improved matching algorithms.
- Confidence scoring and recognition review tools.
- Expanded training and correction workflows.
- Additional output formats.
- More robust handling of different fonts and glyph variations.

## Project Purpose

This project was developed as an exploration of computer vision, image processing, OCR, and practical problem-solving.

It addresses a real-world problem encountered when attempting to extract text from documents containing unfamiliar characters: when conventional libraries and AI-based extraction struggle, manually providing reference examples and performing visual similarity matching can offer an alternative approach.

Rather than relying entirely on pre-trained recognition models, this tool gives the user control over what the system learns and recognises.
