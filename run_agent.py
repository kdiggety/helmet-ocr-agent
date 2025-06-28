from ocr_pipeline import run_pipeline

if __name__ == "__main__":
    df = run_pipeline("photos")
    print(df)