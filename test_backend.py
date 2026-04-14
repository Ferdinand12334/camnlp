from hoo import run_collection

df = run_collection(
    keyword="AI",
    platform="YouTube",
    language="English",
    max_records=5
)

print(df.head())