import urllib.request
import zipfile
import os

def download_and_extract():
    url = "https://targetsound.cs.washington.edu/files/FSDSoundScapes.zip"
    filename = "FSDSoundScapes.zip"
    
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Download completed: {filename}")
        
        print("Extracting...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction completed!")
        
        # 清理zip文件
        os.remove(filename)
        print("Cleanup completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_extract() 