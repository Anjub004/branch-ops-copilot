"""Create store.db: a small, fictional retail database for the stock agent (Milestone 2).

    python make_data.py

Deterministic (fixed random seed), so tests and evals always see the same numbers.
Everything here is made up: branches, products, prices and sales.
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "store.db"
AS_OF = date(2026, 10, 27)   # "today" inside the dataset
DAYS = 28

BRANCHES = [("B01", "Lusaka Central", "Lusaka"), ("B02", "Kitwe", "Copperbelt"), ("B03", "Ndola", "Copperbelt"),
            ("B04", "Livingstone", "Southern"), ("B05", "Kabwe", "Central"), ("B06", "Chipata", "Eastern")]

# sku, name, department, unit price (kwacha), perishable, base units sold per day at an average branch
PRODUCTS = [
    ("1001", "Long-life milk 1L", "Dairy", 32.50, 0, 40), ("1002", "Fresh milk 2L", "Dairy", 45.00, 1, 30),
    ("1003", "Plain yoghurt 500g", "Dairy", 28.00, 1, 18), ("1004", "Cheddar cheese 400g", "Dairy", 95.00, 1, 8),
    ("1005", "Butter 500g", "Dairy", 72.00, 1, 10),
    ("2001", "White bread 700g", "Bakery", 24.00, 1, 60), ("2002", "Brown bread 700g", "Bakery", 26.00, 1, 35),
    ("2003", "Bread rolls 6-pack", "Bakery", 18.00, 1, 25), ("2004", "Chocolate muffin", "Bakery", 15.00, 1, 20),
    ("3001", "Bananas per kg", "Fresh Produce", 22.00, 1, 45), ("3002", "Tomatoes per kg", "Fresh Produce", 30.00, 1, 35),
    ("3003", "Onions 2kg bag", "Fresh Produce", 38.00, 1, 20), ("3004", "Apples per kg", "Fresh Produce", 48.00, 1, 15),
    ("4001", "Mealie meal 25kg", "Grocery", 285.00, 0, 22), ("4002", "Rice 2kg", "Grocery", 62.00, 0, 18),
    ("4003", "Cooking oil 2L", "Grocery", 120.00, 0, 16), ("4004", "Sugar 2kg", "Grocery", 55.00, 0, 20),
    ("4005", "Baked beans 410g", "Grocery", 21.00, 0, 25), ("4006", "Pasta 500g", "Grocery", 24.00, 0, 14),
    ("5001", "Bottled water 500ml", "Beverages", 8.00, 0, 70), ("5002", "Cola 2L", "Beverages", 35.00, 0, 30),
    ("5003", "Orange juice 1L", "Beverages", 42.00, 0, 15), ("5004", "Instant coffee 200g", "Beverages", 110.00, 0, 6),
    ("6001", "Dishwashing liquid 750ml", "Household", 39.00, 0, 10), ("6002", "Toilet paper 9-pack", "Household", 78.00, 0, 12),
    ("6003", "Laundry powder 2kg", "Household", 99.00, 0, 9), ("6004", "Bleach 1.5L", "Household", 33.00, 0, 8),
]
BRANCH_SIZE = {"B01": 1.6, "B02": 1.0, "B03": 1.1, "B04": 0.8, "B05": 0.6, "B06": 0.7}

# A few hand-placed situations, so the agent has something interesting to find.
STOCK_OVERRIDES = {("B02", "2001"): 0, ("B02", "1001"): 12, ("B03", "1002"): 15, ("B04", "4001"): 9, ("B01", "5001"): 60}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE branches (branch_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, region TEXT NOT NULL);
CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL,
                       unit_price REAL NOT NULL, perishable INTEGER NOT NULL);
CREATE TABLE stock (branch_id TEXT NOT NULL REFERENCES branches, sku TEXT NOT NULL REFERENCES products,
                    qty_on_hand INTEGER NOT NULL, PRIMARY KEY (branch_id, sku));
CREATE TABLE sales (sale_date TEXT NOT NULL, branch_id TEXT NOT NULL REFERENCES branches,
                    sku TEXT NOT NULL REFERENCES products, qty INTEGER NOT NULL);
CREATE TABLE purchase_orders (po_id TEXT PRIMARY KEY, branch_id TEXT NOT NULL REFERENCES branches,
                              sku TEXT NOT NULL REFERENCES products, qty INTEGER NOT NULL,
                              order_date TEXT NOT NULL, expected_date TEXT NOT NULL,
                              status TEXT NOT NULL CHECK (status IN ('open', 'received', 'cancelled')));
CREATE INDEX idx_sales ON sales (branch_id, sku, sale_date);
"""


def build(path=DB_PATH, seed=42):
    rng = random.Random(seed)
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO meta VALUES ('as_of', ?)", (AS_OF.isoformat(),))
    con.executemany("INSERT INTO branches VALUES (?, ?, ?)", BRANCHES)
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", [p[:5] for p in PRODUCTS])

    sales, stock = [], []
    for branch_id, _, _ in BRANCHES:
        for sku, _, _, _, _, base in PRODUCTS:
            rate = base * BRANCH_SIZE[branch_id]
            for d in range(DAYS, 0, -1):
                day = AS_OF - timedelta(days=d)
                weekend = 1.3 if day.weekday() >= 5 else 1.0
                sales.append((day.isoformat(), branch_id, sku, max(0, round(rng.gauss(rate * weekend, rate * 0.2)))))
            cover_days = rng.uniform(2, 20)
            qty = STOCK_OVERRIDES.get((branch_id, sku), round(rate * cover_days))
            stock.append((branch_id, sku, qty))
    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", sales)
    con.executemany("INSERT INTO stock VALUES (?, ?, ?)", stock)

    pos, n = [], 1
    for branch_id, _, _ in BRANCHES:
        for sku in rng.sample([p[0] for p in PRODUCTS if not p[4]], 4):  # non-perishables come by PO
            order = AS_OF - timedelta(days=rng.randint(2, 20))
            expected = order + timedelta(days=rng.randint(5, 12))
            status = rng.choice(["open", "open", "received", "cancelled"])
            pos.append((f"PO{n:04d}", branch_id, sku, rng.randint(50, 400), order.isoformat(), expected.isoformat(), status))
            n += 1
    pos.append((f"PO{n:04d}", "B02", "1001", 300, (AS_OF - timedelta(days=10)).isoformat(),
                (AS_OF - timedelta(days=3)).isoformat(), "open"))  # an overdue milk order at Kitwe
    con.executemany("INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?, ?)", pos)
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print("Wrote", build())
