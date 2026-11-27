import sqlite3
from contextlib import closing, nullcontext
from pathlib import Path
from time import sleep
from typing import Protocol
from unicodedata import decimal

from src.schemas.retrieval import VehicleContractContext


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fleet_mock.db"


class VehicleRepository(Protocol):
    def get_by_license_plate(self, license_plate: str) -> VehicleContractContext | None: ...


def normalize_license_plate(plate: str) -> str:
    return "".join(str(decimal(character)) for character in plate if character.isdecimal())


_SEED_RECORDS = (
    ("482-91-302", "F-1001", "Demo Fleet A", "fleet_smb", "Hyundai Tucson", 2024,
     "operational_leasing", "active", True, True, 1500, "מרכז שירות פתח תקווה"),
    ("329-47-615", "F-1002", "Demo Driver A", "private", "Kia Picanto", 2023,
     "private_leasing", "active", True, False, 1900, "מרכז שירות ראשון לציון"),
    ("671-28-904", "F-1003", "Demo Driver B", "private", "Nissan Qashqai", 2025,
     "manufacturer_warranty", "active", False, False, 0, "מרכז שירות חיפה"),
    ("54-762-19", "F-1004", "Demo Driver C", "private", "Renault Clio", 2019,
     "trade_in_warranty", "expired", False, False, 0, "מרכז שירות באר שבע"),
    ("917-36-284", "F-1005", "Demo Fleet B", "fleet_enterprise", "Mazda CX-5", 2024,
     "operational_leasing", "active", True, True, 1200, "מרכז שירות נתניה"),
)


class SQLiteVehicleRepository:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        max_retries: int = 2,
        retry_backoff: float = 0.02,
    ) -> None:
        if max_retries < 0 or retry_backoff < 0:
            raise ValueError("Retry count and backoff must be nonnegative.")
        self.db_path = str(db_path)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._memory_connection = None
        if self.db_path == ":memory:":
            self._memory_connection = self._connect()
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=0.25, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _connection(self):
        if self._memory_connection is not None:
            return nullcontext(self._memory_connection)
        return closing(self._connect())

    def _initialize(self) -> None:
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    normalized_plate TEXT PRIMARY KEY,
                    license_plate TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    customer_name TEXT NOT NULL,
                    customer_segment TEXT NOT NULL,
                    vehicle_make_model TEXT NOT NULL,
                    vehicle_year INTEGER NOT NULL,
                    contract_type TEXT NOT NULL,
                    contract_status TEXT NOT NULL,
                    includes_towing INTEGER NOT NULL,
                    includes_replacement_car INTEGER NOT NULL,
                    deductible_nis INTEGER NOT NULL,
                    preferred_service_center TEXT NOT NULL
                )
            """)
            if connection.execute("SELECT 1 FROM vehicles LIMIT 1").fetchone() is None:
                connection.executemany("""
                    INSERT INTO vehicles (
                        normalized_plate, license_plate, customer_id, customer_name,
                        customer_segment, vehicle_make_model, vehicle_year, contract_type,
                        contract_status, includes_towing, includes_replacement_car,
                        deductible_nis, preferred_service_center
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(normalize_license_plate(record[0]), *record) for record in _SEED_RECORDS])

    def _lookup(self, normalized_plate: str) -> VehicleContractContext | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM vehicles WHERE normalized_plate = ?", (normalized_plate,)
            ).fetchone()
        return VehicleContractContext.model_validate(dict(row)) if row is not None else None

    def get_by_license_plate(self, license_plate: str) -> VehicleContractContext | None:
        normalized_plate = normalize_license_plate(license_plate)
        if not normalized_plate:
            return None
        for attempt in range(self.max_retries + 1):
            try:
                return self._lookup(normalized_plate)
            except (sqlite3.OperationalError, TimeoutError):
                if attempt == self.max_retries:
                    raise
                sleep(self.retry_backoff * 2**attempt)
