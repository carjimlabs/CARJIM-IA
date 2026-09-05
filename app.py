"""
App grafico (Tkinter) para rodar o modelo treinado numa imagem escolhida
pelo usuario, com checkboxes para escolher quais classes mostrar
(Hemacia/Leucocito/Plaqueta).

Uso: python app.py
"""
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detection_core import (  # noqa: E402
    CLASS_LABELS_PT,
    CLASSES,
    MODEL_PATH,
    PLATELET_CLASS_NAME,
    RBC_CLASS_NAME,
    VALID_EXTENSIONS,
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
        self._current_image_path: Path | None = None
        self._photo_image = None  # referencia forte -- sem isso o Tk descarta e o canvas fica em branco

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

        self.select_button = ttk.Button(
            actions_row, text="Selecionar Imagem...", command=self._on_select_image, state="disabled"
        )
        self.select_button.pack(side="left", padx=4)

        self.save_button = ttk.Button(
            actions_row, text="Salvar como...", command=self._on_save, state="disabled"
        )
        self.save_button.pack(side="left", padx=4)

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
    # selecao de imagem e inferencia (thread de fundo, por clique)
    # ------------------------------------------------------------------

    def _on_select_image(self):
        extensions = " ".join(f"*{ext}" for ext in sorted(VALID_EXTENSIONS))
        path_str = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", extensions), ("Todos os arquivos", "*.*")],
        )
        if not path_str:
            return

        image_path = Path(path_str)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro ao abrir imagem", str(exc))
            return

        self._current_image_path = image_path
        self.status_var.set("Processando...")
        self._set_controls_enabled(False)

        want_platelet = self.show_vars[PLATELET_CLASS_NAME].get()
        threading.Thread(
            target=self._inference_worker, args=(image, want_platelet), daemon=True
        ).start()

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
            self.select_button.config(state="normal")

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

        elif kind == "platelet_addon_done":
            _, detections, had_platelet_scan = message
            self._last_detections = detections
            self._last_had_platelet_scan = had_platelet_scan
            self._set_controls_enabled(True)
            self._render()

        elif kind == "inference_error":
            _, error_msg = message
            self.status_var.set("Pronto" if self._last_image is not None else "Selecione uma imagem")
            self._set_controls_enabled(True)
            messagebox.showerror("Erro ao processar imagem", error_msg)

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
        if self._last_annotated_full is None or self._current_image_path is None:
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"{self._current_image_path.stem}_analisado{self._current_image_path.suffix}"
        path_str = filedialog.asksaveasfilename(
            title="Salvar como",
            initialdir=str(OUTPUT_DIR),
            initialfile=default_name,
            defaultextension=self._current_image_path.suffix or ".png",
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
        state = "normal" if enabled else "disabled"
        for widget in (self.select_button, self.save_button, *self.class_checks.values()):
            widget.config(state=state)


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
