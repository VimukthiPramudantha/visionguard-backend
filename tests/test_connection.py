from sqlalchemy import text
from app.database import engine

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("DB Connected:", result.scalar())

except Exception as e:
    print("Failed:", e)