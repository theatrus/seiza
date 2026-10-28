const short = v => new Intl.NumberFormat('en', {maximumSignificantDigits:3, notation:Math.abs(v)>=10000?'compact':'standard'}).format(v);
function chart(id, {xmin,xmax,ymin,ymax,xlabel,ylabel,time=false,integer=false}) {
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
 svg.setAttribute('viewBox','0 0 600 280');
 svg.setAttribute('role','img');
 svg.setAttribute('aria-label',ylabel+' versus '+xlabel);
 function el(tag,attrs,text) {
  const e=document.createElementNS(svg.namespaceURI,tag);
  for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
  if(text!=null)e.textContent=text;
  svg.append(e);return e;
 }
 if(xmin===xmax){xmin-=time?1000:.5;xmax+=time?1000:.5;}
 if(ymin===ymax){const pad=Math.max(Math.abs(ymin)*.05,.05);ymin-=pad;ymax+=pad;}
 const x=v=>64+510*(v-xmin)/(xmax-xmin),y=v=>225-190*(v-ymin)/(ymax-ymin);
 for(let i=0;i<=4;i++){
  const value=ymin+(ymax-ymin)*i/4,yy=y(value);
  el('line',{x1:64,x2:574,y1:yy,y2:yy,stroke:'var(--grid)','stroke-width':1});
  el('text',{x:54,y:yy+4,fill:'var(--muted)','font-size':11,'text-anchor':'end'},Math.abs(value)>=10000||Math.abs(ymax-ymin)<1e-8?fmt(value):new Intl.NumberFormat('en',{maximumFractionDigits:Math.min(12,Math.max(0,Math.ceil(-Math.log10((ymax-ymin)/4))))}).format(value));
 }
 const ticks=new Set();
 for(let i=0;i<=4;i++){
  let value=xmin+(xmax-xmin)*i/4;
  if(integer)value=Math.round(value);
  if(ticks.has(value)||value<xmin||value>xmax)continue;
  ticks.add(value);
  const label=time?new Date(value).toISOString().slice(xmax-xmin>=86400000?0:11,xmax-xmin>=86400000?10:19):short(value);
  el('text',{x:x(value),y:246,fill:'var(--muted)','font-size':11,'text-anchor':'middle'},label);
 }
 el('text',{x:64,y:18,fill:'var(--muted)','font-size':11},ylabel);
 el('text',{x:319,y:272,fill:'var(--muted)','font-size':11,'text-anchor':'middle'},xlabel);
 (typeof id==='string'?$(id):id).replaceChildren(svg);
 return {el,x,y};
}
function dot(c,x,y,color,text) {
 const point=c.el('circle',{cx:c.x(x),cy:c.y(y),r:3.7,fill:color,stroke:'var(--surface)','stroke-width':1.3});
 const title=document.createElementNS(point.namespaceURI,'title');title.textContent=text;point.append(title);
}
