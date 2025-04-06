import os
import pandas as pd


def remove_unannotated_images(csv_file, images_folder, image_extension=".jpg"):
    """
    Reads the CSV of annotations and removes any image file in the images_folder
    whose base name (without extension) does not appear in the CSV's "Image_Name" column.

    Parameters:
      csv_file (str): Path to the CSV file containing annotations.
      images_folder (str): Path to the folder containing the image files.
      image_extension (str): The extension of the image files (default is ".jpg").
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create a set of image names from the CSV. Here we assume the CSV "Image_Name" column
    # contains the base file names without the extension (as in your dataset class).
    annotated_set = set(df["Image_Name"].astype(str))
    print(f"Found {len(annotated_set)} annotated images in the CSV.")

    removed_count = 0
    total_files = 0

    # Loop over all files in the images folder
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(image_extension):
            total_files += 1
            # Extract the base name (without extension)
            basename = os.path.splitext(filename)[0]

            # If the base name is not in the CSV, remove the file
            if basename not in annotated_set:
                file_path = os.path.join(images_folder, filename)
                try:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Removed unannotated image: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"Processed {total_files} image(s). Removed {removed_count} unannotated image(s).")


if __name__ == "__main__":
    # Update these paths to match your environment
    csv_file = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered.csv"
    images_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered"

    # Remove images that do not have an annotation in the CSV
    remove_unannotated_images(csv_file, images_folder, image_extension=".jpg")
