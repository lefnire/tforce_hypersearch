import time, json, re
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgres://lefnire:lefnire@127.0.0.1/hypersearch")


def setup_runs_table():
    conn = engine.connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs
        (
            id SERIAL NOT NULL,
            hypers JSONB NOT NULL,
            reward_avg DOUBLE PRECISION NOT NULL,
            flag VARCHAR(16),
            rewards DOUBLE PRECISION[],
            agent VARCHAR(64) DEFAULT 'ppo_agent'::CHARACTER VARYING NOT NULL
        );
    """)
