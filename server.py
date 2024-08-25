import asyncio
import queue
import socket
import threading

from TTS import tts_queue, launch_tts
from SVC import launch_svc

print("This is the tts-svc backend server for GTASA")
print("AI audio generation program")
print("==========Don't close me===========")


async def process_request(request_queue):
    while True:
        message, conn = request_queue.get()
        try:
            content, speaker, cn = message.split(';', 2)
            cn = True if cn == "cn" else False
            tts_queue.put((content, cn, speaker, conn))
        except RuntimeError as e:
            print(e, conn)
        request_queue.task_done()


def handle_client(conn, address, request_queue):
    while True:
        try:
            message = conn.recv(1024).decode()
            if message:
                request_queue.put((message, conn))
                print("====================================================\n"
                      + "Connection from GTASA asi process:" + str(address))
        except Exception as e:
            print(e, conn)


def work():
    launch_tts()
    launch_svc()
    print('Launch TTS and SVC')

    host = '127.0.0.1'
    port = 65432

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(6)
    print("Waiting for connection...")

    request_queue = queue.Queue()
    processing_thread = threading.Thread(target=asyncio.run, args=(process_request(request_queue),))
    processing_thread.daemon = True
    processing_thread.start()

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, address, request_queue))
        client_thread.daemon = True
        client_thread.start()


if __name__ == '__main__':
    asyncio.run(work())
