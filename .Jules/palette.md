## 2025-05-20 - [Tkinter Tooltip Positioning]
**Learning:** `ttk.Button` widgets do not support `bbox("insert")` for caret positioning. Tooltips must be positioned relative to the widget using `winfo_rootx()`, `winfo_rooty()`, and `winfo_height()` instead.
**Action:** Use `winfo_root*` methods for all widget-relative popups in Tkinter to ensure compatibility across widget types.

## 2026-02-12 - [Mocking Tkinter Widgets Identity]
**Learning:** When mocking `tkinter` with `MagicMock`, distinct widget instantiations (e.g., `tk.Label()`) return the *same* mock object by default. This breaks identity checks (`id(w1) != id(w2)`) and shared state assumptions in tests.
**Action:** Always use `side_effect=lambda *a, **k: MagicMock()` or a factory function when mocking widget classes to ensure each instantiation returns a unique mock object.

## 2026-03-03 - [Treeview Sorting Interaction]
**Learning:** `ttk.Treeview` headers do not support binding via `bind("<Button-1>")` directly on the header region easily. The standard way is `tree.heading(col, command=lambda: ...)`.
**Action:** Always use the `command` option in `tree.heading()` for sort interactions, and remember to capture the loop variable with `lambda c=col:` to avoid late-binding issues.
