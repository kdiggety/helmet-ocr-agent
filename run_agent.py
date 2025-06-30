from helmet_sticker_ocr_pipeline import run_pipeline

if __name__ == "__main__":
    df = run_pipeline("images/train/")
    print(df)
