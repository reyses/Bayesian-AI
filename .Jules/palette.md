## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2026-02-23 - [Tkinter Tooltip UX in High-Density Dashboards]
**Learning:** Immediate display of tooltips on `<Enter>` in high-density Tkinter dashboards (like `visualization/dashboard.py`) causes aggressive UI flashing and frequently obstructs underlying interactive elements before the user intends to view the information.
**Action:** Always introduce a delay (e.g., ~500ms) using `widget.after()` for tooltip generation, ensure the scheduled event is properly cancelled via `widget.after_cancel()` if the mouse leaves before the timeout, and provide a `<ButtonPress>` binding to easily dismiss the tooltip if it does obstruct the UI.
