"""
PDF をダウンロードしてテキストファイルに変換するスクリプト。
Usage: python test/pdf_to_text.py <URL> [--out <output_path>]
"""
import sys
import argparse
import requests
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

def read_pdf(url, temp_path="./data/temp.pdf"):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
    with open(temp_path, "wb") as f:
        f.write(requests.get(url, headers=headers).content)
    laparams = LAParams()
    laparams.word_margin = 0.2
    text = extract_text(temp_path, laparams=laparams)
    text_list = text.split("\n")
    text_list = [x.strip() for x in text_list]
    text_lines = []
    for line in text_list:
        if line != "" and not line.isdigit():
            text_lines.append(line)
    return " ".join(text_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="PDF の URL")
    parser.add_argument("--out", type=str, default=None, help="出力テキストファイルパス")
    args = parser.parse_args()
    print(f"Downloading {args.url} ...")
    text = read_pdf(args.url)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data") / (Path(args.url).stem + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved to {out_path} ({len(text)} chars)")

if __name__ == "__main__":
    main()
