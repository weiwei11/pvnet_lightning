# @author: ww
from datetime import datetime

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker
# from sqlalchemy.orm.session import Session
from contextlib import contextmanager

BaseModel = declarative_base()
Session = sessionmaker()


@contextmanager
def open_session():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_table(engine, base_model_class=BaseModel):
    base_model_class.metadata.create_all(engine)


def bind_session(engine, session_class=Session):
    session_class.configure(bind=engine)


def create_database(path):
    engine = sqlalchemy.create_engine(path)
    return engine


class EvaluationRecord(BaseModel):
    __tablename__ = 'relocation_evaluation_record'

    id = Column(Integer, primary_key=True)
    record_time = Column('record time', DateTime, default=datetime.now, onupdate=datetime.now)
    dataset_name = Column('dataset', String(30))
    scene_name = Column('scene', String(30))
    method = Column('method', String(256))
    mean_r = Column('mean R', Float)
    mean_t = Column('mean T', Float)
    median_r = Column('median R', Float)
    median_t = Column('median T', Float)
    ratio_5degree_5cm = Column('5° 5cm', Float)
    semantic_classification = Column('semantic classification', Float)

# engine = sqlalchemy.create_engine('sqlite:///evaluation_record.db')
# create_all_table(engine)
# bind_session(engine)
# with storage.open_session() as session:
#     session.add(storage.EvaluationRecord(dataset_name=data_name, scene_name=scene_name, method=method,
#                                          mean_r=mean_r, mean_t=mean_t, median_r=median_r, median_t=median_t, ratio_5degree_5cm=ratio))
