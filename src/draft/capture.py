import struct
import socket
import numpy as np
import cv2

# WSL2 NAT mode: reach the Windows host via the gateway IP, not 127.0.0.1.
# Find this with: ip route | grep default
HOST = "172.27.48.1"
PORT = 5000


def _recv_exactly(conn: socket.socket, n: int) -> bytes:
    """Returns the next n bytes from conn.

    Args:
        conn (socket.socket): The socket connection to read from.
        n (int): The number of bytes to read and return.
    
    Returns:
        bytes: The n bytes requested.
    
    Raises:
        ConnectionError: If the connection closes cleanly.
    """
    buf = b""

    # We grow buf until we receive exactly n bytes.
    while len(buf) < n:

        # Conn blocks until it receives a number of bytes in the range [1,n-len(buf)].
        chunk = conn.recv(n - len(buf))

        # If an empty packet is received, the connection closed cleanly.
        if not chunk:
            raise ConnectionError("Server disconnected")
        
        # Append the latest packet's payload to buf.
        buf += chunk
    return buf


def receive_frames():
    """Yields frames received by the TCP client from the Windows webcam server.

    Yields:
        cv2.typing.MatLike or None: The received frame from the Windows webcam server.
    """

    # Initializes socket as a TCP connection IPv4.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        # The address to connect needs to be sent as a tuple of (HOST, PORT).
        sock.connect((HOST, PORT))
        print(f"Connected to Windows camera server at {HOST}:{PORT}")

        # Yield frames
        while True:

            # Receive exactly the first 4 bytes from the socket connection. This integer is the number of bytes representing the frame.
            raw_len = _recv_exactly(sock, 4)

            # Convert the 4 bytes to the unsigned integer bytes size of the frame.
            frame_len = struct.unpack(">I", raw_len)[0]

            # Receive the exact number of bytes representing the frame.
            data = _recv_exactly(sock, frame_len)

            # Decode the raw frame bytes into a color image (MatLike). 
            # cv2.imdecode() cannot process raw bytes so the bytes are converted to a numpy array of integers first.
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Skip frames not decoded properly.
            if frame is None:
                continue

            yield frame

# Show the live video frames received if the file is run directly.
if __name__ == "__main__":
    for frame in receive_frames():
        cv2.imshow("WSL2 Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
