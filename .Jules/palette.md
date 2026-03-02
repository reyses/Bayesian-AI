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

## 2026-02-24 - [Testing Matplotlib Default Globals]
**Learning:** When adding features that use global constants like `DEFAULT_CHART_DPI`, existing tests might fail with `NameError` if the constant isn't defined or imported.
**Action:** Always handle missing global constants with fallback mechanisms or proper imports to ensure tests pass without unhandled exceptions.

## 2026-02-24 - [High-Density Dashboard Tooltips]
**Learning:** Instant tooltips in high-density dashboards cause aggressive flashing and UI obstruction, leading to poor UX.
**Action:** Use a ~500ms delay before showing tooltips (`after`) and include a `<ButtonPress>` binding to instantly dismiss them. Ensure scheduled events are cancelled (`after_cancel`) if the mouse leaves before the timeout.
