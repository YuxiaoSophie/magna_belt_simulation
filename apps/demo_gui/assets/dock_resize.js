// Drag the viewer dock's left edge (.dock__handle) to set --dock-w; double-click resets.
// Delegated listeners: Dash renders the layout after this script runs.
(function () {
  var KEY = "demo_gui.dock_w", root = document.documentElement, dragging = false, cur = null;
  function set(px) { root.style.setProperty("--dock-w", px + "px"); }
  function handle(e) { return e.target.closest && e.target.closest(".dock__handle"); }
  try {
    var w = parseInt(localStorage.getItem(KEY), 10);
    if (w > 0) set(w);
  } catch (err) {}
  document.addEventListener("pointerdown", function (e) {
    var h = handle(e);
    if (!h || e.button !== 0) return;
    dragging = true;
    cur = null;
    document.body.classList.add("dock-resizing");
    try { h.setPointerCapture(e.pointerId); } catch (err) {}
    e.preventDefault();
  });
  document.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    var max = 0.7 * window.innerWidth;
    cur = Math.round(Math.min(Math.max(window.innerWidth - e.clientX, 320), max));
    set(cur);
  });
  function end() {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("dock-resizing");
    if (cur) { try { localStorage.setItem(KEY, String(cur)); } catch (err) {} }
  }
  document.addEventListener("pointerup", end);
  document.addEventListener("pointercancel", end);
  document.addEventListener("dblclick", function (e) {
    if (!handle(e)) return;
    root.style.removeProperty("--dock-w");
    try { localStorage.removeItem(KEY); } catch (err) {}
  });
})();
