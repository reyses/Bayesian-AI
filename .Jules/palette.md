## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2025-05-21 - [Tkinter Tooltip Delay and Dismissal]
**Learning:** In high-density dashboards, immediate tooltips cause aggressive flashing and UI obstruction. If a tooltip appears instantly on hover, it distracts users moving their mouse across the interface. Furthermore, tooltips that remain visible during interaction (clicks) can obscure the very elements the user is trying to interact with.
**Action:** Always implement a ~500ms delay before showing tooltips, ensure scheduled show events are cancelled (`after_cancel`) if the mouse leaves early, and bind `<ButtonPress>` to immediately dismiss the tooltip and its pending schedule.
