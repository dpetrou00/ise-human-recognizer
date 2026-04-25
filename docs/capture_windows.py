"""
Run this script with native Windows Python (not WSL2) to stream webcam frames
to the WSL2 processing pipeline over TCP.

Windows is the server (it provides the webcam feed); WSL2 connects as the client.

Setup:
    pip install opencv-python

Usage (Windows PowerShell or CMD):
    python src\\capture_windows.py
"""

import struct
import socket
import cv2

HOST = "0.0.0.0"
PORT = 5000


def stream_frames(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f"Waiting for WSL2 client on port {PORT}...")

        conn, addr = srv.accept()
        print(f"Connected: {addr} — streaming. Press Ctrl+C to stop.")

        try:
            with conn:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame, exiting.")
                        break

                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    data = buf.tobytes()
                    conn.sendall(struct.pack(">I", len(data)) + data)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except (BrokenPipeError, ConnectionResetError):
            print("WSL2 client disconnected.")
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_frames()
