"""
App grafico (Tkinter) para rodar o modelo treinado sobre capturas ao vivo da
tela do usuario, com checkboxes para escolher quais classes mostrar
(Hemacia/Leucocito/Plaqueta).

Uso: python app.py
"""
import ctypes
import queue
import sys
import threading
import tkinter as tk
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detection_core import (  # noqa: E402
    CLASS_LABELS_PT,
    CLASSES,
    MODEL_PATH,
    PLATELET_CLASS_NAME,
    RBC_CLASS_NAME,
    draw_detections,
    estimate_rbc_size,
    load_font,
    load_model,
    merge_platelet_detections,
    platelet_tile_scan,
    primary_detections,
)

OUTPUT_DIR = PROJECT_ROOT / "Imagens analisadas"
CANVAS_BG = "#1e1e1e"
LIVE_CAPTURE_INTERVAL_MS = 2000  # tempo entre o fim de uma analise e a proxima captura

MONITORINFOF_PRIMARY = 0x1


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def list_monitors() -> list[dict]:
    """Enumera os monitores fisicos via user32 (ctypes -- sem dependencia
    nova). Cada item tem 'bbox' (esquerda, topo, direita, baixo) nas mesmas
    coordenadas de tela usadas por ImageGrab.grab(all_screens=True) e
    'primary' (bool)."""
    monitors: list[dict] = []

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT), ctypes.c_ssize_t
    )

    def _callback(hmonitor, _hdc, _rect_ptr, _data):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append({
                "bbox": (r.left, r.top, r.right, r.bottom),
                "primary": bool(info.dwFlags & MONITORINFOF_PRIMARY),
            })
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_callback), 0)
    return monitors


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CARJIM-IA - Deteccao de Celulas")
        self.root.geometry("1000x700")
        self.root.minsize(600, 400)

        self.model = None
        self.font = None
        self.device = None

        self.msg_queue: queue.Queue = queue.Queue()

        self._last_image: Image.Image | None = None
        self._last_detections: list | None = None
        self._last_had_platelet_scan = False
        self._last_annotated_full: Image.Image | None = None
        self._last_capture_at: datetime | None = None
        self._photo_image = None  # referencia forte -- sem isso o Tk descarta e o canvas fica em branco

        self._live_running = False
        self._capture_after_id = None
        self.monitor_options = self._build_monitor_options()
        self.monitor_var = tk.StringVar(value=self.monitor_options[0]["label"])

        # Um checkbox por classe da taxonomia (detection_core.CLASSES), na
        # mesma ordem. Todos marcados por padrao.
        self.show_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in CLASSES
        }
        self.class_checks: dict[str, ttk.Checkbutton] = {}

        self.status_var = tk.StringVar(value="Carregando modelo...")

        self._build_widgets()
        self._start_model_load()
        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # opcoes de tela (monitor a capturar)
    # ------------------------------------------------------------------

    def _build_monitor_options(self) -> list[dict]:
        try:
            monitors = list_monitors()
        except Exception:  # noqa: BLE001 - enumeracao de monitor nao pode derrubar o app
            monitors = []

        # principal primeiro, depois as demais na ordem devolvida pelo Windows
        monitors.sort(key=lambda mon: not mon["primary"])

        options = []
        for index, mon in enumerate(monitors, start=1):
            left, top, right, bottom = mon["bbox"]
            width, height = right - left, bottom - top
            suffix = " (principal)" if mon["primary"] else ""
            options.append({
                "label": f"Tela {index}{suffix} - {width}x{height}",
                "bbox": mon["bbox"],
            })

        options.append({"label": "Todas as telas", "bbox": None})
        return options

    def _selected_monitor_bbox(self):
        selected_label = self.monitor_var.get()
        for option in self.monitor_options:
            if option["label"] == selected_label:
                return option["bbox"]
        return None

    # ------------------------------------------------------------------
    # construcao da interface
    # ------------------------------------------------------------------

    def _build_widgets(self):
        # linha 1: um checkbox por classe (sao 7, ficam apertados junto dos
        # botoes numa linha so -- por isso as classes ganham a propria linha)
        classes_row = ttk.Frame(self.root, padding=(8, 8, 8, 2))
        classes_row.pack(side="top", fill="x")
        for name in CLASSES:
            check = ttk.Checkbutton(
                classes_row,
                text=CLASS_LABELS_PT.get(name, name),
                variable=self.show_vars[name],
                command=self._on_checkbox_toggle,
            )
            check.pack(side="left", padx=4)
            self.class_checks[name] = check

        # linha 2: acoes
        actions_row = ttk.Frame(self.root, padding=(8, 2, 8, 8))
        actions_row.pack(side="top", fill="x")

        self.capture_button = ttk.Button(
            actions_row, text="Iniciar Captura ao Vivo", command=self._on_toggle_live_capture, state="disabled"
        )
        self.capture_button.pack(side="left", padx=4)

        self.save_button = ttk.Button(
            actions_row, text="Salvar como...", command=self._on_save, state="disabled"
        )
        self.save_button.pack(side="left", padx=4)

        ttk.Label(actions_row, text="Tela:").pack(side="left", padx=(12, 2))
        self.monitor_combo = ttk.Combobox(
            actions_row,
            textvariable=self.monitor_var,
            values=[option["label"] for option in self.monitor_options],
            state="readonly",
            width=26,
        )
        self.monitor_combo.pack(side="left", padx=4)

        status_label = ttk.Label(self.root, textvariable=self.status_var, padding=(8, 4))
        status_label.pack(side="top", fill="x")

        self.canvas = tk.Canvas(self.root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render())

    # ------------------------------------------------------------------
    # carregamento do modelo (thread de fundo, ao abrir o app)
    # ------------------------------------------------------------------

    def _start_model_load(self):
        threading.Thread(target=self._model_load_worker, daemon=True).start()

    def _model_load_worker(self):
        try:
            if not MODEL_PATH.exists():
                self.msg_queue.put((
                    "model_error",
                    f"Modelo nao encontrado em {MODEL_PATH}.\nRode antes scripts/03_train.py.",
                ))
                return
            model, device = load_model()
            font = load_font()
            self.msg_queue.put(("model_ready", model, device, font))
        except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer falha na UI
            self.msg_queue.put(("model_error", str(exc)))

    # ------------------------------------------------------------------
    # captura de tela ao vivo + inferencia (thread de fundo, em loop)
    # ------------------------------------------------------------------

    def _on_toggle_live_capture(self):
        if self._live_running:
            self._stop_live_capture()
        else:
            self._start_live_capture()

    def _start_live_capture(self):
        self._live_running = True
        self.capture_button.config(text="Parar Captura ao Vivo")
        self._run_capture_cycle()

    def _stop_live_capture(self):
        self._live_running = False
        self.capture_button.config(text="Iniciar Captura ao Vivo")
        if self._capture_after_id is not None:
            self.root.after_cancel(self._capture_after_id)
            self._capture_after_id = None
        self.status_var.set("Captura ao vivo parada")

    def _run_capture_cycle(self):
        if not self._live_running:
            return

        try:
            image = self._grab_screen_hidden()
        except Exception as exc:  # noqa: BLE001
            self._stop_live_capture()
            messagebox.showerror("Erro ao capturar a tela", str(exc))
            return

        self._last_capture_at = datetime.now()
        self.status_var.set("Analisando captura da tela...")

        want_platelet = self.show_vars[PLATELET_CLASS_NAME].get()
        threading.Thread(
            target=self._inference_worker, args=(image, want_platelet), daemon=True
        ).start()

    def _grab_screen_hidden(self) -> Image.Image:
        """Torna a janela do app invisivel (sem minimizar/perder foco de
        quem esta usando outra janela) para nao capturar a si mesma, tira o
        print da tela escolhida no combobox e restaura a visibilidade."""
        bbox = self._selected_monitor_bbox()
        self.root.attributes("-alpha", 0.0)
        self.root.update()
        try:
            # all_screens=True sempre -- e o que faz o ImageGrab entender
            # coordenadas de monitores secundarios (que podem ser negativas)
            # tanto pro bbox de uma tela especifica quanto pra tela toda.
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
        finally:
            self.root.attributes("-alpha", 1.0)
        return image.convert("RGB")

    def _schedule_next_capture(self):
        if not self._live_running:
            return
        self._capture_after_id = self.root.after(LIVE_CAPTURE_INTERVAL_MS, self._run_capture_cycle)

    def _inference_worker(self, image: Image.Image, want_platelet: bool):
        try:
            detections = primary_detections(self.model, image)
            had_platelet_scan = False

            if want_platelet:
                detections, had_platelet_scan = self._add_platelet_scan(image, detections)

            self.msg_queue.put(("inference_done", image, detections, had_platelet_scan))
        except Exception as exc:  # noqa: BLE001
            self.msg_queue.put(("inference_error", str(exc)))

    def _add_platelet_scan(self, image: Image.Image, detections: list):
        """Roda o reforco de plaqueta (passada em recortes) e mescla no
        resultado. Devolve (detections_mescladas, rodou_ou_nao)."""
        platelet_class_id = next(
            (cid for cid, name in self.model.names.items() if name == PLATELET_CLASS_NAME), None
        )
        rbc_class_id = next(
            (cid for cid, name in self.model.names.items() if name == RBC_CLASS_NAME), None
        )
        rbc_size_px = estimate_rbc_size(detections, rbc_class_id) if rbc_class_id is not None else None

        if platelet_class_id is None or not rbc_size_px:
            return detections, False

        tile_platelets = platelet_tile_scan(self.model, image, platelet_class_id, rbc_size_px)
        merged = merge_platelet_detections(detections, tile_platelets, platelet_class_id)
        return merged, True

    # ------------------------------------------------------------------
    # checkboxes
    # ------------------------------------------------------------------

    def _on_checkbox_toggle(self):
        if self._last_image is None:
            return

        # Em captura ao vivo o proximo ciclo ja roda com o estado atual dos
        # checkboxes em poucos segundos -- nao vale a pena disparar mais uma
        # thread de reforco so pra essa imagem, que ja vai ficar obsoleta.
        if self._live_running:
            self._render()
            return

        # Caso especial: o usuario marcou Plaqueta, mas a imagem em cache
        # foi processada com Plaqueta desmarcada -- nao existe caixa de
        # plaqueta pra mostrar ainda, entao roda so o reforco (nao repete
        # a deteccao inteira) sob demanda.
        if self.show_vars[PLATELET_CLASS_NAME].get() and not self._last_had_platelet_scan:
            self.status_var.set("Processando...")
            self._set_controls_enabled(False)
            threading.Thread(target=self._platelet_addon_worker, daemon=True).start()
            return

        self._render()

    def _platelet_addon_worker(self):
        try:
            detections, had_platelet_scan = self._add_platelet_scan(
                self._last_image, self._last_detections
            )
            self.msg_queue.put(("platelet_addon_done", detections, had_platelet_scan))
        except Exception as exc:  # noqa: BLE001
            self.msg_queue.put(("inference_error", str(exc)))

    # ------------------------------------------------------------------
    # fila de mensagens (consumida na thread principal)
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                message = self.msg_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_message(self, message):
        kind = message[0]

        if kind == "model_ready":
            _, model, device, font = message
            self.model = model
            self.device = device
            self.font = font
            self.status_var.set("Pronto")
            self.capture_button.config(state="normal")

        elif kind == "model_error":
            _, error_msg = message
            self.status_var.set("Erro ao carregar o modelo")
            messagebox.showerror("Erro ao carregar o modelo", error_msg)

        elif kind == "inference_done":
            _, image, detections, had_platelet_scan = message
            self._last_image = image
            self._last_detections = detections
            self._last_had_platelet_scan = had_platelet_scan
            self._set_controls_enabled(True)
            self._render()
            self._schedule_next_capture()

        elif kind == "platelet_addon_done":
            _, detections, had_platelet_scan = message
            self._last_detections = detections
            self._last_had_platelet_scan = had_platelet_scan
            self._set_controls_enabled(True)
            self._render()

        elif kind == "inference_error":
            _, error_msg = message
            self.status_var.set("Pronto" if self._last_image is not None else "Captura ao vivo parada")
            self._set_controls_enabled(True)
            messagebox.showerror("Erro ao processar captura", error_msg)
            self._schedule_next_capture()

    # ------------------------------------------------------------------
    # desenho / filtragem (sincrono, sem thread -- rapido)
    # ------------------------------------------------------------------

    def _selected_class_names(self) -> set:
        return {name for name, var in self.show_vars.items() if var.get()}

    def _render(self):
        if self._last_image is None or self.model is None:
            return

        selected = self._selected_class_names()
        filtered = [det for det in self._last_detections if self.model.names[det[0]] in selected]

        # nunca desenhar em cima da imagem ja anotada -- sempre parte de
        # uma copia limpa da imagem original
        annotated = draw_detections(self._last_image.copy(), filtered, self.model.names, self.font)
        self._last_annotated_full = annotated

        self._show_on_canvas(annotated)
        self._update_status_counts(filtered)

    def _show_on_canvas(self, annotated: Image.Image):
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        img_w, img_h = annotated.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        display_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))

        display_image = annotated.resize(display_size, Image.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(display_image)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo_image, anchor="center")

    def _update_status_counts(self, filtered: list):
        counts = {name: 0 for name in CLASSES}
        for det in filtered:
            name = self.model.names[det[0]]
            if name in counts:
                counts[name] += 1

        # so mostra as classes marcadas, para a linha de status nao virar
        # uma parede de "0" com as 7 classes
        selected = self._selected_class_names()
        parts = [
            f"{CLASS_LABELS_PT.get(name, name)}: {counts[name]}"
            for name in CLASSES
            if name in selected
        ]
        self.status_var.set("Concluido - " + ", ".join(parts))

    # ------------------------------------------------------------------
    # salvar
    # ------------------------------------------------------------------

    def _on_save(self):
        if self._last_annotated_full is None or self._last_capture_at is None:
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = self._last_capture_at.strftime("%Y%m%d_%H%M%S")
        default_name = f"captura_{timestamp}_analisada.png"
        path_str = filedialog.asksaveasfilename(
            title="Salvar como",
            initialdir=str(OUTPUT_DIR),
            initialfile=default_name,
            defaultextension=".png",
        )
        if not path_str:
            return

        try:
            self._last_annotated_full.save(path_str)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro ao salvar", str(exc))

    # ------------------------------------------------------------------
    # utilitarios
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        # capture_button fica de fora -- precisa continuar clicavel durante
        # a analise para o usuario poder parar a captura ao vivo a qualquer
        # momento.
        state = "normal" if enabled else "disabled"
        for widget in (self.save_button, *self.class_checks.values()):
            widget.config(state=state)

    def _on_close(self):
        self._stop_live_capture()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        App(root)
        root.mainloop()
    except Exception as exc:  # noqa: BLE001 - garante que erros inesperados aparecam pro usuario
        messagebox.showerror("Erro inesperado", str(exc))
        raise


if __name__ == "__main__":
    main()
