/* Auto-AlphaFold3 demo UI — trajectory interactivity.
   Attaches hover tooltips + click-to-select rings to an SVG line chart with
   id="trajChart". Illustrative points; shared by the board and the trajectory
   panel. No dependencies. */
(function () {
  var svg = document.getElementById('trajChart');
  if (!svg) return;
  var NS = 'http://www.w3.org/2000/svg';
  var pts = (window.TRAJ_POINTS) || [
    {x:60,y:254,t:'T001',s:0.325,st:'provisional'},
    {x:91,y:267,t:'T003',s:0.318,st:'killed'},
    {x:138,y:242,t:'T006',s:0.331,st:'provisional'},
    {x:185,y:218,t:'T009',s:0.344,st:'confirmed'},
    {x:247,y:228,t:'T013',s:0.339,st:'confirmed'},
    {x:325,y:194,t:'T018',s:0.357,st:'confirmed'},
    {x:388,y:209,t:'T022',s:0.349,st:'killed'},
    {x:466,y:166,t:'T027',s:0.372,st:'confirmed'},
    {x:528,y:150,t:'T031',s:0.381,st:'confirmed'},
    {x:606,y:118,t:'T036',s:0.398,st:'confirmed'},
    {x:669,y:105,t:'T039',s:0.405,st:'confirmed'},
    {x:700,y:92,t:'T042',s:0.412,st:'confirmed'}
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
  var selected = pts[pts.length - 1];
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
