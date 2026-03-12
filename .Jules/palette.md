## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2026-03-12 - [Tkinter Tooltip UX Delays]
**Learning:** Instant hover tooltips in dense dashboards (`<Enter>` triggers immediately) cause aggressive flashing and UI obstruction as the user moves their mouse. They also linger annoyingly.
**Action:** Always implement a delay (e.g., 500ms via `after`) before showing Tkinter tooltips, ensure `after_cancel` is called on `<Leave>`, and bind `<ButtonPress>` to dismiss them so they don't block interaction.