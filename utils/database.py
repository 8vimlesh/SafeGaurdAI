"""Database models and helper functions for PPE Safety System.

Defines SQLAlchemy ORM models for `violations` and `detection_logs` and
provides helper functions to log events and query recent statistics.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import safeguardai.config as config

Base = declarative_base()


class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String, nullable=True)
    zone = Column(String, nullable=True)
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    screenshot_path = Column(String, nullable=True)
    resolved = Column(Boolean, default=False)


class DetectionLog(Base):
    __tablename__ = "detection_logs"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String, nullable=True)
    total_people = Column(Integer, default=0)
    violation_count = Column(Integer, default=0)
    compliance_pct = Column(Float, default=0.0)


# Engine and session
ENGINE = create_engine(config.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=ENGINE)


def init_db() -> None:
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=ENGINE)


def log_violation(
    camera_id: str, zone: str, class_name: str, confidence: float, screenshot_path: str, resolved: bool = False
) -> Violation:
    """Insert a violation record and return the created object."""
    session = SessionLocal()
    v = Violation(
        camera_id=camera_id,
        zone=zone,
        class_name=class_name,
        confidence=confidence,
        screenshot_path=screenshot_path,
        resolved=resolved,
    )
    session.add(v)
    session.commit()
    session.refresh(v)
    session.close()
    return v


def log_detection(camera_id: str, total_people: int, violation_count: int) -> DetectionLog:
    """Insert a detection summary record and return the created object."""
    session = SessionLocal()
    compliance_pct = 0.0
    if total_people > 0:
        compliance_pct = max(0.0, 100.0 * (1.0 - (violation_count / float(total_people))))
    dl = DetectionLog(
        camera_id=camera_id, total_people=total_people, violation_count=violation_count, compliance_pct=compliance_pct
    )
    session.add(dl)
    session.commit()
    session.refresh(dl)
    session.close()
    return dl


def log_detection_log(camera_id: str, total_people: int, violation_count: int) -> DetectionLog:
    """Alias for legacy detection logging API."""
    return log_detection(camera_id, total_people, violation_count)


def get_recent_violations(limit: int = 10) -> List[Violation]:
    """Return the most recent `limit` violations."""
    session = SessionLocal()
    rows = session.query(Violation).order_by(Violation.timestamp.desc()).limit(limit).all()
    session.close()
    return rows


def get_compliance_stats() -> Dict[str, float]:
    """Return simple compliance statistics (average compliance_pct and counts)."""
    session = SessionLocal()
    total = session.query(DetectionLog).count()
    avg_pct = 0.0
    if total > 0:
        avg = session.query(func.avg(DetectionLog.compliance_pct)).scalar()
        avg_pct = float(avg or 0.0)
    session.close()
    return {"records": total, "avg_compliance_pct": avg_pct}
