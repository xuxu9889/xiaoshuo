from store import search_chunks, search_facts

def probe(q):
    print(f"\n== Query: {q} ==")
    fs = search_facts(q, k=3)
    cs = search_chunks(q, k=3)
    print("Facts:")
    for f in fs: print(" -", f["id"], f["summary"][:60])
    print("Chunks:")
    for c in cs: print(" -", c["id"], c["text"][:60])

if __name__ == "__main__":
    probe("转职条件")
    probe("营养舱")
    probe("初始武器")
