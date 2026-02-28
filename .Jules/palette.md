## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2026-02-12 - [Tkinter Tooltip Fallback]
**Learning:** `bbox("insert")` on `ttk.Label` raises `AttributeError` or returns `None`. A robust tooltip implementation must catch these exceptions and fallback to `winfo_rootx/y` relative positioning.
**Action:** Always wrap `bbox("insert")` in a try-except block when implementing tooltips that might be attached to non-text widgets.

## 2026-02-24 - [Mocking Matplotlib Subplots]
**Learning:** When adding features that instantiate `plt.subplots`, existing tests mocking `tkinter` will fail if `plt.subplots` is not also mocked and configured to return a tuple `(fig, ax)`.
**Action:** Always patch `plt.subplots` in tests for classes that create plots in `__init__`, ensuring it returns valid mock objects.

## 2024-05-19 - Desktop App Tooltip Interaction Pattern
**Learning:** Tooltips in dense desktop UIs (like Tkinter dashboards) that trigger instantly on hover create an aggressive, flashing experience. Furthermore, users need a way to dismiss them if they obscure underlying elements or interact with the screen.
**Action:** Always add a ~500ms debounce/delay to tooltip visibility in high-density dashboards, and ensure click events cancel the active hover state.
