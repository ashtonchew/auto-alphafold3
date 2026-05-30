/* Auto-AlphaFold3 demo UI — trajectory interactivity.
   Attaches hover tooltips + click-to-select rings to an SVG line chart with
   id="trajChart". Illustrative points; shared by the board and the trajectory
   panel. No dependencies. */
(function () {
  var svg = document.getElementById('trajChart');
  if (!svg) return;
  var NS = 'http://www.w3.org/2000/svg';
  var pts = (window.TRAJ_POINTS) || [
    {x:60,y:199,t:'T001',s:0.321,st:'provisional'},
    {x:94,y:214,t:'T002',s:0.318,st:'provisional'},
    {x:127,y:175,t:'T003',s:0.326,st:'provisional'},
    {x:161,y:151,t:'T004',s:0.331,st:'provisional'},
    {x:195,y:223,t:'T005',s:0.316,st:'provisional'},
    {x:228,y:122,t:'T006',s:0.337,st:'provisional'},
    {x:262,y:190,t:'T007',s:0.323,st:'provisional'},
    {x:296,y:142,t:'T008',s:0.333,st:'provisional'},
    {x:330,y:209,t:'T009',s:0.319,st:'provisional'},
    {x:363,y:113,t:'T010',s:0.339,st:'provisional'},
    {x:397,y:233,t:'T011',s:0.314,st:'provisional'},
    {x:431,y:94,t:'T012',s:0.343,st:'confirmed'},
    {x:464,y:194,t:'T013',s:0.322,st:'provisional'},
    {x:498,y:127,t:'T014',s:0.336,st:'provisional'},
    {x:532,y:218,t:'T015',s:0.317,st:'provisional'},
    {x:565,y:166,t:'T016',s:0.328,st:'provisional'},
    {x:599,y:204,t:'T017',s:0.320,st:'provisional'},
    {x:633,y:137,t:'T018',s:0.334,st:'provisional'},
    {x:666,y:228,t:'T019',s:0.315,st:'provisional'},
    {x:700,y:185,t:'T020',s:0.324,st:'provisional'}
  ];
  var COLORS = {confirmed:'#7fee64', killed:'#ff9ea1', provisional:'#9a9a9a'};
  function ring(r, op) {
    var c = document.createElementNS(NS, 'circle');
    c.setAttribute('r', r); c.setAttribute('fill', 'none');
    c.setAttribute('stroke', '#7fee64'); c.setAttribute('stroke-width', '1.6');
    c.setAttribute('opacity', op); c.setAttribute('pointer-events', 'none');
    svg.appendChild(c); return c;
  }
  var selRing = ring(10, '0'), hoverRing = ring(9, '0');
  var tip = document.createElement('div'); tip.className = 'pt-tip'; document.body.appendChild(tip);
  function place(c, p) { c.setAttribute('cx', p.x); c.setAttribute('cy', p.y); }
  // Default selection: the confirmed best if any, else the highest-scoring point.
  var selected = pts[0];
  pts.forEach(function (p) {
    var better = (p.st === 'confirmed') !== (selected.st === 'confirmed')
      ? p.st === 'confirmed'
      : p.s > selected.s;
    if (better) selected = p;
  });
  place(selRing, selected); selRing.setAttribute('opacity', '0.85');
  function moveTip(el) { var r = el.getBoundingClientRect(); tip.style.left = (r.left + r.width / 2) + 'px'; tip.style.top = r.top + 'px'; }
  pts.forEach(function (p) {
    var hit = document.createElementNS(NS, 'circle');
    hit.setAttribute('cx', p.x); hit.setAttribute('cy', p.y); hit.setAttribute('r', '15');
    hit.setAttribute('fill', 'transparent'); hit.setAttribute('class', 'pt');
    hit.addEventListener('mouseenter', function () {
      place(hoverRing, p); hoverRing.setAttribute('opacity', '1');
      tip.innerHTML = '<b>' + p.t + '</b> · <span style="color:' + COLORS[p.st] + '">' + p.st + '</span> · <span class="num">' + p.s.toFixed(3) + '</span>';
      tip.classList.add('show'); moveTip(hit);
    });
    hit.addEventListener('mousemove', function () { moveTip(hit); });
    hit.addEventListener('mouseleave', function () { hoverRing.setAttribute('opacity', '0'); tip.classList.remove('show'); });
    hit.addEventListener('click', function () { selected = p; place(selRing, p); selRing.setAttribute('opacity', '0.85'); });
    svg.appendChild(hit);
  });
})();
