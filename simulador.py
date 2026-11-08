import math
import random
import sqlite3
import sys
import time
from datetime import datetime

DB = 'monitoreo.db'
INTERVALO = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

MAQUINAS = [
    ('M1', 'Torno 1', 'L'),
    ('M2', 'Torno 2', 'M'),
    ('M3', 'Fresadora 1', 'H'),
    ('M4', 'Fresadora 2', 'L'),
    ('M5', 'Taladro 1', 'M'),
]
DEGRADA = 'M4'  # esta maquina se degrada poco a poco


def limitar(x, minimo, maximo):
    return max(minimo, min(maximo, x))


con = sqlite3.connect(DB, timeout=10)
con.execute('PRAGMA journal_mode=WAL')
con.execute('CREATE TABLE IF NOT EXISTS maquinas (id TEXT PRIMARY KEY, nombre TEXT, tipo TEXT)')
con.execute('CREATE TABLE IF NOT EXISTS lecturas (id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'maquina_id TEXT, ts TEXT, air REAL, proc REAL, rpm REAL, torque REAL, wear REAL)')
con.executemany('INSERT OR IGNORE INTO maquinas VALUES (?, ?, ?)', MAQUINAS)
con.commit()

estado = {mid: {'wear': random.randint(0, 150), 'd': 0.0, 'aire': random.uniform(299, 301)}
          for mid, _, _ in MAQUINAS}
estado[DEGRADA]['wear'] = 150

print(f"Simulador iniciado: {len(MAQUINAS)} maquinas, una lectura cada {INTERVALO} s. Ctrl+C para detener.")
t = 0
try:
    while True:
        t += 1
        ahora = datetime.now().isoformat(timespec='seconds')
        filas = []
        for mid, nombre, tipo in MAQUINAS:
            e = estado[mid]
            e['wear'] += 1  # el desgaste de la herramienta sube con el uso
            torque = random.gauss(40, 7)
            rpm = random.gauss(1540, 120)
            if mid == DEGRADA:
                e['d'] += 1 / 60  # nivel de degradacion: 0 = sana, 1 = muy degradada
                if e['d'] > 1.05:  # mantenimiento: vuelve a empezar
                    e['d'] = 0.0
                    e['wear'] = 0
                torque += 28 * e['d']
                rpm -= 250 * e['d']
            elif e['wear'] > 200:  # cambio de herramienta
                e['wear'] = 0
            aire = e['aire'] + 0.8 * math.sin(t / 40) + random.gauss(0, 0.2)
            proceso = aire + 10 + random.gauss(0, 0.3)
            filas.append((mid, ahora,
                          round(limitar(aire, 295, 305), 1),
                          round(limitar(proceso, 305, 314), 1),
                          round(limitar(rpm, 1000, 3000)),
                          round(limitar(torque, 0, 80), 1),
                          e['wear']))
        con.executemany('INSERT INTO lecturas (maquina_id, ts, air, proc, rpm, torque, wear) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?)', filas)
        con.commit()
        m4 = [f for f in filas if f[0] == DEGRADA][0]
        print(f"[{ahora[-8:]}] {len(filas)} lecturas | Fresadora 2: torque {m4[5]} Nm, {m4[4]} rpm, desgaste {m4[6]} min")
        time.sleep(INTERVALO)
except KeyboardInterrupt:
    print("\nSimulador detenido.")
finally:
    con.close()
