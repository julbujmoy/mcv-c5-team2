import os
import shutil
import pandas as pd
import pytesseract
from PIL import Image


# If Tesseract is not in your PATH, specify its location:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Linux/macOS example
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows example

def has_text(image_path, char_threshold=10):
    """
    Returns True if the image contains text exceeding the char_threshold.
    Adjust char_threshold based on your criteria.
    """
    try:
        img = Image.open(image_path)
        # Convert to grayscale for better OCR performance
        gray = img.convert('L')
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(gray)
        # Evaluate if the text length meets the threshold
        return len(extracted_text.strip()) >= char_threshold
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # If there's an error (e.g., file not found, invalid image), treat it as having text
        return True


def filter_images_and_save_csv(
        csv_path,
        images_folder,
        output_csv_path,
        output_images_folder,
        image_extension=".jpg"
):
    """
    Reads the CSV, checks each image for text,
    and writes a new CSV + folder containing images without text.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_images_folder, exist_ok=True)

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Prepare a list for rows that pass the text filter
    filtered_rows = []

    for idx, row in df.iterrows():
        # Example columns: "Image_Name" and "Title"
        image_name = str(row["Image_Name"])

        # Construct the full path to the image
        image_path = os.path.join(images_folder, image_name + image_extension)

        # Check if the file actually exists
        if not os.path.isfile(image_path):
            print(f"File not found, skipping: {image_path}")
            continue

        # Check if the image has text
        if has_text(image_path):
            print(f"Filtering out image with text: {image_name}")
        else:
            # Keep the row
            filtered_rows.append(row)

            # Copy the image to the new folder
            destination_path = os.path.join(output_images_folder, image_name + image_extension)
            shutil.copyfile(image_path, destination_path)

    # Create a new DataFrame and save as CSV
    new_df = pd.DataFrame(filtered_rows)
    new_df.to_csv(output_csv_path, index=False)

    print(f"Filtering complete. {len(filtered_rows)} images without text saved to {output_images_folder}.")
    print(f"New CSV written to {output_csv_path}.")


# Example usage for a single split:
if __name__ == "__main__":
    # Adjust these paths to your own
    original_csv_path = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_data.csv"
    original_images_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test"

    # Output paths for filtered data
    filtered_csv_path = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered.csv"
    filtered_images_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered"

    # Run the filtering
    filter_images_and_save_csv(
        csv_path=original_csv_path,
        images_folder=original_images_folder,
        output_csv_path=filtered_csv_path,
        output_images_folder=filtered_images_folder,
        image_extension=".jpg"
    )


