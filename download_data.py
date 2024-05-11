import zipfile
import os

def unzip_and_extract(zip_file_path, output_dir):
    try:
        # Open the zip file for reading
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all contents to the output directory
            zip_ref.extractall(output_dir)
        
        print("File unzipped and extracted successfully.")
    except Exception as e:
        print(f"Error unzipping and extracting file: {e}")

if __name__ == "__main__":
    # Path to the zip file to unzip
    zip_file_path = "/home/xboril/PA228/data_seg_public.zip?predmet=1553781"

    # Output directory where the contents will be extracted
    output_dir = "/home/xboril/PA228/data"

    # Call the unzip_and_extract function
    unzip_and_extract(zip_file_path, output_dir)
