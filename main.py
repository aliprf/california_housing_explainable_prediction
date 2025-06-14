import threading

from client.client import build_ui
from server.server import app


def run_flask()-> None:
    app.run(port=5000, debug=False, use_reloader=False)


def run_gradio() -> None:
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    t1 = threading.Thread(target=run_flask)
    t1.start()
    run_gradio()
