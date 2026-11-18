/* v2 region-box editor for the optional Gradio fallback. */
(function () {
  if (window.__rfV2Ready) { if (window.__rfPending) { window.rfLoadPayload(window.__rfPending); window.__rfPending = null; } return; }
  window.__rfV2Ready = true;
  var S = { payload: null, selected: null, active: null, zoom: 1, drag: null, preview: null };
  window.__rfBoxes = [];
  window.__rfAction = {};

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]; }); }
  function color(g) { return g === 'bottom' ? 'var(--rf-bottom)' : 'var(--rf-top)'; }
  function box(id) { for (var i = 0; i < window.__rfBoxes.length; i++) if (window.__rfBoxes[i].id === id) return window.__rfBoxes[i]; return null; }
  function region(key) { var rs = (S.payload && S.payload.schema && S.payload.schema.regions) || []; for (var i = 0; i < rs.length; i++) if (rs[i].key === key) return rs[i]; return null; }
  function setText(sel, text) { var el = document.querySelector(sel); if (el) el.textContent = text; }
  function toast(msg) {
    var stage = document.querySelector('.rf-stage'); if (!stage) return;
    var old = document.querySelector('.rf-toast'); if (old) old.remove();
    var el = document.createElement('div'); el.className = 'rf-toast'; el.textContent = msg;
    stage.appendChild(el); setTimeout(function () { el.remove(); }, 2500);
  }
  function action(name, data) {
    var payload = {name: name}; if (data) Object.keys(data).forEach(function (k) { payload[k] = data[k]; });
    window.__rfAction = payload;
    var btn = document.querySelector('#rf-action'); if (btn) btn.click();
  }
  function commit() { var b = document.querySelector('#rf-sync'); if (b) b.click(); }

  window.rfLoadPayload = function (json) {
    try { S.payload = JSON.parse(json || '{}'); } catch (e) { return; }
    var p = S.payload;
    window.__rfBoxes = (p.boxes || []).map(function (b) {
      return {id: b.id, garment: b.garment, region: b.region, side: b.side || 'center',
              label: b.label, xyxy: b.xyxy.slice(), hidden: !!b.hidden};
    });
    S.selected = window.__rfBoxes.length ? window.__rfBoxes[0].id : null;
    var rs = (p.schema && p.schema.regions) || [];
    if (!S.active || !region(S.active)) S.active = rs.length ? rs[0].key : null;
    setText('#rf-header-image', (p.index + 1) + ' / ' + p.total + '  •  ' + p.image_name);
    setText('#rf-header-progress-text', p.annotated + '/' + p.total);
    var bar = document.querySelector('#rf-progress-bar'); if (bar) bar.style.width = (p.total ? 100 * p.annotated / p.total : 0) + '%';
    renderAll();
  };
  window.__rfPending = null;
  window.rfLoadColor = function (json) {
    if (!json || json === '{}') return;
    var data = {}; try { data = JSON.parse(json); } catch (e) { return; }
    if (data.color) {
      var garment = data.garment || 'top';
      upsert(garment, 'color', 'center', data.color, data.xyxy || [0, 0, 100, 100]);
      commit(); renderAll(); toast('HSV colour: ' + data.color);
    }
  };
  function upsert(garment, reg, side, value, xyxy) {
    var existing = null;
    window.__rfBoxes.forEach(function (b) { if (b.garment === garment && b.region === reg && b.side === side) existing = b; });
    if (existing) { existing.label = value; existing.xyxy = xyxy.slice(); return existing; }
    var b = {id: 'b' + Date.now().toString(36) + Math.floor(Math.random() * 1e4).toString(36),
             garment: garment, region: reg, side: side, label: value, xyxy: xyxy.slice(), hidden: false};
    window.__rfBoxes.push(b); return b;
  }

  function renderAll() { if (!S.payload) return; renderSidebar(); renderCanvas(); renderEditor(); }

  function renderSidebar() {
    var host = document.querySelector('#rf-panel-classes'); if (!host) return;
    var rs = (S.payload.schema && S.payload.schema.regions) || [];
    var html = '<div class="rf-section-title">Regions</div>';
    rs.forEach(function (r, i) {
      var b = null;
      window.__rfBoxes.forEach(function (x) { if (x.garment === r.garment && x.region === r.region && x.side === r.side) b = x; });
      html += '<div class="rf-class-row ' + (S.active === r.key ? 'active' : '') + '" data-action="region" data-key="' + esc(r.key) + '">' +
        '<span class="rf-dot" style="background:' + color(r.garment) + '"></span><span>' + (i + 1 <= 9 ? (i + 1) + ' ' : '') + esc(r.title) + '</span>' +
        '<span class="rf-count">' + (b ? esc(b.label) : '—') + '</span></div>';
    });
    host.innerHTML = html;
    var reg = document.querySelector('#rf-panel-regions');
    var lines = window.__rfBoxes.map(function (b, i) { return (i + 1) + ': ' + esc(b.garment) + '.' + esc(b.region) + ' = ' + esc(b.label); });
    if (reg) reg.innerHTML = '<div class="rf-section-title">Boxes</div><div class="rf-hint">' + (lines.join('<br>') || 'No boxes yet. Pick a region and drag.') + '</div>';
  }

  function rfPos(e) {
    var svg = document.querySelector('#rf-svg'), p = S.payload, r = svg.getBoundingClientRect();
    return [Math.max(0, Math.min(p.width, (e.clientX - r.left) / r.width * p.width)),
            Math.max(0, Math.min(p.height, (e.clientY - r.top) / r.height * p.height))];
  }

  function renderCanvas() {
    var p = S.payload, img = document.querySelector('#rf-image'), wrap = document.querySelector('#rf-canvas-wrap'), svg = document.querySelector('#rf-svg');
    if (!img || !wrap || !svg) return;
    img.src = p.image_src || '';
    wrap.style.width = (p.width * S.zoom) + 'px'; wrap.style.height = (p.height * S.zoom) + 'px';
    svg.setAttribute('viewBox', '0 0 ' + p.width + ' ' + p.height);
    var h = 8 / S.zoom, parts = [];
    window.__rfBoxes.forEach(function (b) {
      if (b.hidden) return;
      var sel = b.id === S.selected, c = color(b.garment), x1 = b.xyxy[0], y1 = b.xyxy[1], x2 = b.xyxy[2], y2 = b.xyxy[3];
      parts.push('<g data-box="' + esc(b.id) + '"><rect class="rf-rect' + (sel ? ' selected' : '') + '" x="' + x1 + '" y="' + y1 +
        '" width="' + (x2 - x1) + '" height="' + (y2 - y1) + '" stroke="' + c + '" vector-effect="non-scaling-stroke" style="pointer-events:all"></rect>');
      var text = b.region + ': ' + b.label;
      parts.push('<rect x="' + x1 + '" y="' + Math.max(0, y1 - 17 / S.zoom) + '" width="' + (text.length * 7 + 10) / S.zoom + '" height="' + (16 / S.zoom) + '" fill="' + c + '"></rect>');
      parts.push('<text class="rf-label" x="' + (x1 + 3 / S.zoom) + '" y="' + Math.max(12 / S.zoom, y1 - 4 / S.zoom) + '" style="font-size:' + (11 / S.zoom) + 'px">' + esc(text) + '</text>');
      if (sel) [[x1,y1,'nw'],[x2,y1,'ne'],[x1,y2,'sw'],[x2,y2,'se']].forEach(function (hd) {
        parts.push('<rect class="rf-handle" data-handle="' + hd[2] + '" x="' + (hd[0] - h/2) + '" y="' + (hd[1] - h/2) + '" width="' + h + '" height="' + h + '" stroke="' + c + '" vector-effect="non-scaling-stroke" style="pointer-events:all"></rect>');
      });
      parts.push('</g>');
    });
    if (S.preview) { var q = S.preview; parts.push('<rect x="' + Math.min(q[0],q[2]) + '" y="' + Math.min(q[1],q[3]) + '" width="' + Math.abs(q[2]-q[0]) + '" height="' + Math.abs(q[3]-q[1]) + '" stroke="#ef4444" stroke-width="2" stroke-dasharray="6 4" fill="#ef444422" vector-effect="non-scaling-stroke"></rect>'); }
    svg.innerHTML = parts.join('');
    svg.onpointerdown = function (e) {
      var handle = e.target.getAttribute && e.target.getAttribute('data-handle');
      var group = e.target.closest && e.target.closest('[data-box]'), id = group && group.getAttribute('data-box');
      if (handle && id) { S.drag = {type:'resize', id:id, corner:handle}; e.preventDefault(); return; }
      if (id) { S.selected = id; renderEditor(); renderCanvas(); return; }
      if (S.active) { var p0 = rfPos(e); S.drag = {type:'draw', start:p0}; S.preview = [p0[0],p0[1],p0[0],p0[1]]; }
    };
    svg.onpointermove = function (e) {
      if (!S.drag) return; var p1 = rfPos(e);
      if (S.drag.type === 'draw') { S.preview = [S.drag.start[0], S.drag.start[1], p1[0], p1[1]]; renderCanvas(); return; }
      var b = box(S.drag.id); if (!b) return; var o = b.xyxy.slice(), c = S.drag.corner;
      if (c.indexOf('w') >= 0) o[0] = Math.min(o[2] - 4, p1[0]);
      if (c.indexOf('e') >= 0) o[2] = Math.max(o[0] + 4, p1[0]);
      if (c.indexOf('n') >= 0) o[1] = Math.min(o[3] - 4, p1[1]);
      if (c.indexOf('s') >= 0) o[3] = Math.max(o[1] + 4, p1[1]);
      b.xyxy = o.map(Math.round); renderCanvas();
    };
    svg.onpointerup = function (e) {
      if (!S.drag) return; var d = S.drag, p1 = rfPos(e); S.drag = null; S.preview = null;
      if (d.type === 'resize') { commit(); renderAll(); return; }
      var x1 = Math.round(Math.min(d.start[0], p1[0])), y1 = Math.round(Math.min(d.start[1], p1[1]));
      var x2 = Math.round(Math.max(d.start[0], p1[0])), y2 = Math.round(Math.max(d.start[1], p1[1]));
      if (x2 - x1 < 8 || y2 - y1 < 8) { renderCanvas(); return; }
      var r = region(S.active); if (!r) return;
      var b = upsert(r.garment, r.region, r.side, r.classes[0], [x1, y1, x2, y2]);
      S.selected = b.id;
      if (r.region === 'garment') action('color', {garment: r.garment, xyxy: [x1,y1,x2,y2]});
      commit(); renderAll();
    };
  }

  function renderEditor() {
    var host = document.querySelector('#rf-editor-host'); if (!host) return;
    var b = box(S.selected);
    if (!b) { host.innerHTML = ''; return; }
    var r = null;
    ((S.payload && S.payload.schema && S.payload.schema.regions) || []).forEach(function (x) {
      if (x.garment === b.garment && x.region === b.region && x.side === b.side) r = x;
    });
    var values = r ? r.classes : [b.label];
    host.innerHTML = '<div class="rf-editor"><div class="rf-editor-head"><span class="rf-dot" style="background:' + color(b.garment) + '"></span>' + esc(b.garment) + ' · ' + esc(b.region) + '</div>' +
      '<label>Label</label><select data-action="label">' + values.map(function (v) { return '<option' + (v === b.label ? ' selected' : '') + '>' + esc(v) + '</option>'; }).join('') + '</select>' +
      '<div class="rf-editor-actions"><button class="rf-btn danger" data-action="remove">Remove</button><button class="rf-btn primary" data-action="close">Save + Close</button></div></div>';
  }

  document.addEventListener('click', function (e) {
    var el = e.target.closest('[data-action]'); if (!el) return;
    var name = el.getAttribute('data-action');
    if (name === 'region') { S.active = el.getAttribute('data-key'); renderSidebar(); }
    else if (name === 'remove') { window.__rfBoxes = window.__rfBoxes.filter(function (b) { return b.id !== S.selected; }); S.selected = null; commit(); renderAll(); }
    else if (name === 'close') { S.selected = null; renderEditor(); renderCanvas(); }
    else if (name === 'undo' || name === 'redo' || name === 'reset' || name === 'prev' || name === 'next' || name === 'skip' || name === 'submit') action(name);
  });
  document.addEventListener('change', function (e) {
    if (e.target.getAttribute && e.target.getAttribute('data-action') === 'label') {
      var b = box(S.selected); if (b) { b.label = e.target.value; commit(); renderSidebar(); }
    }
  });
  document.addEventListener('keydown', function (e) {
    var tag = (document.activeElement && document.activeElement.tagName) || '';
    if (['INPUT','SELECT','TEXTAREA'].indexOf(tag) >= 0) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { action(e.shiftKey ? 'redo' : 'undo'); e.preventDefault(); return; }
    if (/^[1-9]$/.test(e.key)) { var rs = (S.payload.schema.regions || []); var r = rs[Number(e.key) - 1]; if (r) { S.active = r.key; renderSidebar(); } }
    else if (e.key === 'Delete') { window.__rfBoxes = window.__rfBoxes.filter(function (b) { return b.id !== S.selected; }); S.selected = null; commit(); renderAll(); }
    else if (e.key === 'ArrowLeft') action('prev');
    else if (e.key === 'ArrowRight') action('next');
    else if (e.key === 'Enter') { action('submit'); e.preventDefault(); }
  });

  window.rfInitEditor = function () {
    if (window.__rfPending) { window.rfLoadPayload(window.__rfPending); window.__rfPending = null; }
    requestAnimationFrame(function () { var p = S.payload; if (!p) return; var scroll = document.querySelector('.rf-scroll');
      S.zoom = Math.min((scroll.clientWidth - 44) / p.width, (scroll.clientHeight - 44) / p.height, 1); renderCanvas(); });
  };
})();
