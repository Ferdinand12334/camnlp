# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import bcrypt

Base = declarative_base()
engine = create_engine("sqlite:///camnlp.db")
SessionLocal = sessionmaker(bind=engine)


# ── MODELS ────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    user_id       = Column(Integer, primary_key=True)
    username      = Column(String, unique=True)
    password_hash = Column(String)
    role          = Column(String)
    email         = Column(String)
    created_at    = Column(DateTime, default=datetime.utcnow)

class TextData(Base):
    __tablename__   = "text_data"
    data_id         = Column(Integer, primary_key=True)
    content         = Column(Text)
    source          = Column(String)
    platform        = Column(String)
    language        = Column(String)
    collection_date = Column(DateTime)

class Topic(Base):
    __tablename__ = "topics"
    topic_id      = Column(Integer, primary_key=True)
    topic_name    = Column(String)
    keywords      = Column(Text)
    frequency     = Column(Integer, default=0)

class AnalysisResult(Base):
    __tablename__   = "analysis_results"
    id              = Column(Integer, primary_key=True)
    data_id         = Column(Integer, ForeignKey("text_data.data_id"))
    sentiment_label = Column(String)
    polarity_score  = Column(Float)
    topic_id        = Column(Integer, ForeignKey("topics.topic_id"))
    analysis_date   = Column(DateTime)

class Report(Base):
    __tablename__ = "reports"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer)
    title         = Column(String)
    content       = Column(Text)
    format        = Column(String)
    created_at    = Column(DateTime, default=datetime.utcnow)


# ── SEED DATA ─────────────────────────────────────────────────────────────────
DEFAULT_TOPICS = [
    (1, "Security",   "security,attack,protest,crisis,conflict,army,soldier,violence,clashes"),
    (2, "Economy",    "economy,price,trade,inflation,business,market,money,investment,jobs"),
    (3, "Education",  "school,student,education,university,teacher,reform,learning,exam"),
    (4, "Healthcare", "health,hospital,doctor,medicine,disease,clinic,nurse,treatment,vaccine"),
    (5, "Politics",   "government,president,election,policy,minister,parliament,law,corruption"),
]

DEFAULT_USERS = [
    ("admin",   "admin123",   "Administrator"),
    ("analyst", "analyst123", "Analyst"),
]


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def get_session():
    return SessionLocal()


def init_db():
    Base.metadata.create_all(engine)
    session = get_session()

    # ✅ Seed default users
    if not session.query(User).first():
        for username, password, role in DEFAULT_USERS:
            pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            session.add(User(username=username, password_hash=pw_hash, role=role))

    # ✅ Seed topics (this was missing — root cause of your crash!)
    if not session.query(Topic).first():
        for topic_id, name, keywords in DEFAULT_TOPICS:
            session.add(Topic(
                topic_id=topic_id,
                topic_name=name,
                keywords=keywords,
                frequency=0,
            ))

    session.commit()
    session.close()


def verify_user(username, password):
    session = get_session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        return user
    return None