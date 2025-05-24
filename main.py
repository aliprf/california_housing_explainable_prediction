
import threading
from server.server import app  
from client.client import build_ui

def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

def run_gradio():
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    t1 = threading.Thread(target=run_flask)
    t1.start()
    run_gradio() 