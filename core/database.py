"""SQLite decision database for DRS sessions and review evidence."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from config.settings import DECISIONS_DIR

try:
    from sqlalchemy import (
        Boolean,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        create_engine,
        select,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
except Exception:  # pragma: no cover
    create_engine = None
    declarative_base = None
    sessionmaker = None


if declarative_base is not None:
    Base = declarative_base()
else:
    Base = object


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    date = Column(String, nullable=False)
    venue = Column(String, default="")
    team_bat = Column(String, default="")
    team_bowl = Column(String, default="")
    innings = Column(Integer, default=1)


class DecisionRecord(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    appeal_type = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    ball_impact_x = Column(Float)
    ball_impact_y = Column(Float)
    ball_impact_z = Column(Float)
    projected_hit_stumps = Column(Boolean, default=False)
    pitch_contact_zone = Column(String, default="")
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)


class CameraEventRecord(Base):
    __tablename__ = "camera_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    camera_id = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    event_type = Column(String, nullable=False)
    dropped_frames = Column(Integer, default=0)


class TrackingFrameRecord(Base):
    __tablename__ = "tracking_frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), nullable=False)
    frame_num = Column(Integer, nullable=False)
    ball_x = Column(Float)
    ball_y = Column(Float)
    velocity_mps = Column(Float)
    direction_deg = Column(Float)


class DatabaseManager:
    """Stores DRS decisions and evidence with a small SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if create_engine is None or sessionmaker is None:
            raise RuntimeError("SQLAlchemy is required for DatabaseManager")
        self.db_path = Path(db_path) if db_path else DECISIONS_DIR / "drs.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.SessionLocal = sessionmaker(self.engine, future=True)
        self.run_migrations()

    def run_migrations(self) -> None:
        Base.metadata.create_all(self.engine)

    def ensure_session(self, session_id: str, **metadata: Any) -> None:
        with self.SessionLocal() as db:
            existing = db.get(SessionRecord, session_id)
            if existing is None:
                db.add(
                    SessionRecord(
                        id=session_id,
                        date=str(metadata.get("date", time.strftime("%Y-%m-%d"))),
                        venue=str(metadata.get("venue", "")),
                        team_bat=str(metadata.get("team_bat", "")),
                        team_bowl=str(metadata.get("team_bowl", "")),
                        innings=int(metadata.get("innings", 1)),
                    )
                )
                db.commit()

    def log_decision(self, appeal_data: dict[str, Any]) -> int:
        session_id = str(appeal_data.get("session_id", "default"))
        self.ensure_session(session_id)
        with self.SessionLocal() as db:
            record = DecisionRecord(
                timestamp=float(appeal_data.get("timestamp", time.time())),
                appeal_type=str(appeal_data.get("appeal_type", "LBW")),
                decision=str(appeal_data.get("decision", "REVIEW_INCONCLUSIVE")),
                confidence=float(appeal_data.get("confidence", 0.0)),
                ball_impact_x=_optional_float(appeal_data.get("ball_impact_x")),
                ball_impact_y=_optional_float(appeal_data.get("ball_impact_y")),
                ball_impact_z=_optional_float(appeal_data.get("ball_impact_z")),
                projected_hit_stumps=bool(appeal_data.get("projected_hit_stumps", False)),
                pitch_contact_zone=str(appeal_data.get("pitch_contact_zone", "")),
                session_id=session_id,
            )
            db.add(record)
            db.flush()
            for frame in appeal_data.get("tracking_frames", []):
                db.add(
                    TrackingFrameRecord(
                        decision_id=record.id,
                        frame_num=int(frame.get("frame_num", 0)),
                        ball_x=_optional_float(frame.get("ball_x")),
                        ball_y=_optional_float(frame.get("ball_y")),
                        velocity_mps=_optional_float(frame.get("velocity_mps")),
                        direction_deg=_optional_float(frame.get("direction_deg")),
                    )
                )
            db.commit()
            return int(record.id)

    def get_session_decisions(self, session_id: str) -> list[dict[str, Any]]:
        with self.SessionLocal() as db:
            rows = db.execute(select(DecisionRecord).where(DecisionRecord.session_id == session_id)).scalars()
            return [_decision_to_dict(row) for row in rows]

    def get_accuracy_stats(self) -> dict[str, Any]:
        with self.SessionLocal() as db:
            decisions = list(db.execute(select(DecisionRecord)).scalars())
        if not decisions:
            return {"total_decisions": 0, "average_confidence": 0.0, "inconclusive": 0}
        return {
            "total_decisions": len(decisions),
            "average_confidence": sum(row.confidence for row in decisions) / len(decisions),
            "inconclusive": sum(1 for row in decisions if row.decision == "REVIEW_INCONCLUSIVE"),
        }

    def export_session_csv(self, session_id: str, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.get_session_decisions(session_id)
        with output.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys()) if rows else ["id", "session_id", "decision"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decision_to_dict(row: DecisionRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "appeal_type": row.appeal_type,
        "decision": row.decision,
        "confidence": row.confidence,
        "ball_impact_x": row.ball_impact_x,
        "ball_impact_y": row.ball_impact_y,
        "ball_impact_z": row.ball_impact_z,
        "projected_hit_stumps": row.projected_hit_stumps,
        "pitch_contact_zone": row.pitch_contact_zone,
        "session_id": row.session_id,
    }
