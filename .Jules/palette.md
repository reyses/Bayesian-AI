## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2025-05-21 - [Delayed Tooltips for High-Density Dashboards]
**Learning:** Instant tooltips triggered by `<Enter>` events in high-density dashboards cause aggressive UI flashing and obstruction as users navigate across dense visual elements.
**Action:** Always implement a ~500ms delay using `.after()` before showing tooltips, cancel the timer with `.after_cancel()` if the mouse leaves early, and bind `<ButtonPress>` to dismiss tooltips immediately, ensuring a smoother user experience.
