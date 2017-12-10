import time, json, re
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgres://lefnire:lefnire@127.0.0.1/hypersearch")


def setup_runs_table():
    conn = engine.connect()
    conn.execute("""
        create table if not exists runs
        (
            id serial not null,
            hypers jsonb not null,
            reward_avg double precision not null,
            flag varchar(16),
            rewards double precision[],
            agent varchar(64) default 'ppo_agent'::character varying not null
        );
    """)
