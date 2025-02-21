import os
import shutil
import glob

def organize_audio_files(directory):
    """
    Organizes audio files in a directory into subfolders based on filename suffixes.

    Files ending with specific suffixes (e.g., -sc.wav, -dk.wav, -lo.wav, -hu.wav, -ti.wav, -ch.wav)
    will be moved to folders named after these suffixes (e.g., 'sc', 'dk', 'lo', 'hu', 'ti', 'ch').

    Args:
        directory (str): The path to the directory containing the audio files.
    """

    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return

    suffixes_folders = {
        "-sc": "sc",
        "-dk": "dk",
        "-lo": "lo",
        "-hu": "hu",
        "-ti": "ti",
        "-ch": "ch",
        "-bp": "bp",
        "-dc": "dc",
        "-bu": "bu"   
    }

    for suffix, folder_name in suffixes_folders.items():
        folder_path = os.path.join(directory, folder_name)
        os.makedirs(folder_path, exist_ok=True) # Create folders if they don't exist

    for filename in os.listdir(directory):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(directory, filename)
            moved = False
            for suffix, folder_name in suffixes_folders.items():
                if filename.lower().endswith(f"{suffix}.wav"):
                    destination_folder = os.path.join(directory, folder_name)
                    destination_path = os.path.join(destination_folder, filename)
                    try:
                        shutil.move(filepath, destination_path)
                        print(f"Moved '{filename}' to '{folder_name}' folder.")
                        moved = True
                        break # Move only once even if multiple suffixes are present (prioritize first match)
                    except Exception as e:
                        print(f"Error moving '{filename}': {e}")
                        moved = True # To prevent default move in case of error
                        break
            if not moved:
                print(f"File '{filename}' does not match any suffix and remains in the main directory.")

if __name__ == "__main__":
    target_directory = input("Enter the directory containing the audio files to organize: ")
    organize_audio_files(target_directory)
    print("File organization complete.")