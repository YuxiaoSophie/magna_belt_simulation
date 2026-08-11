import json
import socket
import time

import pyspacemouse


HOST = "127.0.0.1"
PORT = 5005


print("[Mac] Opening SpaceMouse...")

device = pyspacemouse.open()

if device is None:
    raise RuntimeError("Could not open SpaceMouse")

print("[Mac] SpaceMouse connected:", device)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"[Mac] Waiting for Newton connection on {HOST}:{PORT}...")


try:
    while True:

        conn, addr = server.accept()

        print("[Mac] Newton connected!")

        try:
            while True:

                s = device.read()

                msg = {
                    "x": float(s.x),
                    "y": float(s.y),
                    "z": float(s.z),

                    "roll": float(s.roll),
                    "pitch": float(s.pitch),
                    "yaw": float(s.yaw),

                    "buttons": list(s.buttons),
                }

                packet = (
                    json.dumps(msg) + "\n"
                ).encode("utf-8")

                conn.sendall(packet)

                time.sleep(0.01)

        except (
            BrokenPipeError,
            ConnectionResetError,
        ):
            print("[Mac] Newton disconnected.")

        finally:
            conn.close()

except KeyboardInterrupt:
    print("\n[Mac] stopping")

finally:
    device.close()
    server.close()