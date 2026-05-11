import requests

BASE = "http://127.0.0.1:5001"

TESTS = [
    ("BER", "CDG", "2026-02-15"),
    ("BER", "LHR", "2026-03-10"),
    ("FRA", "JFK", "2026-04-20"),
    ("MUC", "BCN", "2026-02-05"),
    ("HAM", "AMS", "2026-02-25"),
]

def post(path, payload):
    r = requests.post(BASE + path, json=payload, timeout=30)
    return r.status_code, r.json()

def main():
    for o, d, dt in TESTS:
        print("=" * 70)
        print(f"Route: {o}->{d} date={dt}")

        code, out = post("/model1_curve", {"origin": o, "destination": d, "date": dt})
        print("MODEL1_CURVE:", code)
        print(out)

        code, out = post("/model2", {"origin": o, "destination": d, "date": dt})
        print("MODEL2:", code)
        print(out)

if __name__ == "__main__":
    main()
