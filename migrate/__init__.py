import json

import psycopg


def main():
    key_to_tex = json.load(open("data/symbols.json"))
    key_to_tex = {x["id"]: x["command"][1:] for x in key_to_tex}

    tex_to_typ = json.load(open("data/default.json"))["commands"]
    tex_to_typ = {k: v["alias"] for k, v in tex_to_typ.items() if v.get("alias")}

    key_to_typ = {k: tex_to_typ[v] for k, v in key_to_tex.items() if v in tex_to_typ}

    conn = psycopg.connect("dbname=detypify")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE key_to_typ (
            id serial PRIMARY KEY,
            key text,
            typ text)
    """)
    cur.executemany(
        "INSERT INTO key_to_typ(key, typ) VALUES (%s, %s)",
        key_to_typ.items(),
    )
    cur.execute("""
        CREATE TABLE typ_samples (
            id serial PRIMARY KEY,
            typ text,
            strokes json)
    """)
    cur.execute("""
        INSERT INTO typ_samples(typ, strokes)
        SELECT typ, strokes
        FROM samples
        JOIN key_to_typ
        ON samples.key = key_to_typ.key
    """)
    conn.commit()
