import os

def normalize_extension(ext):
    ext = ext.lower()
    if ext in ['.jpeg', '.jpg']:
        return '.jpg'
    elif ext == '.png':
        return '.png'
    elif ext == '.txt':
        return '.txt'
    else:
        return None  # skip unsupported files

def normalize_filenames_in_sequence(folder='images/train', root_file_name='image'):
    images = []
    for fname in os.listdir(folder):
        base, ext = os.path.splitext(fname)
        normalized_ext = normalize_extension(ext)
        if normalized_ext:
            images.append((fname, normalized_ext))

    images.sort()  # optional, to maintain consistent order

    for idx, (old_name, ext) in enumerate(images, start=1):
        new_name = f"{root_file_name}_{idx}{ext}"
        src = os.path.join(folder, old_name)
        dst = os.path.join(folder, new_name)
        print(f"Renaming {src} -> {dst}")
        os.rename(src, dst)
